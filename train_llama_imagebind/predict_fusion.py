import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
from peft import PeftModel
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm  # 引入进度条
import json
from transformers import set_seed

from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data
from model.projector import MMInputProjector
from model.fusion import CrossAttentionFusion

CONFIG = {
    "base_model_path": "/root/huggingface/llama/Meta-Llama-3-8B-Instruct", 
    "vision_model_path": "/root/huggingface/imagebind/imagebind_huge.pth", 
    "checkpoint_dir": "./checkpoints/llama_mintrec2_fusion/checkpoint-2313", 
    "test_data_path": "/root/user/xyh/Datasets/MIntRec2/MIntRec2_test.json", 
    "output_file": "./eval/mintrec2_predictions_fusion.json",
    "device": "cuda"
}

# ==========================================
# 2. 推理类定义
# ==========================================
class IntentPredictor:
    def __init__(self, model_dir, base_model_path, vision_model_path, device="cuda"):
        
        self.device = device

        print(f"Loading Base Qwen Model from {base_model_path}...")
        # A. 加载 Base LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            trust_remote_code=True
        )
        
        # B. 加载 LoRA 权重
        print(f"Loading LoRA adapters from {model_dir}...")
        self.llm = PeftModel.from_pretrained(base_model, model_dir)
        self.llm.eval() # 切换到评估模式
        
        # C. 加载 Vision Encoder
        print("Loading ImageBind...")
        self.encoder = imagebind_model.imagebind_huge(pretrained=False)
        self.encoder.load_state_dict(torch.load(CONFIG['vision_model_path'], map_location="cpu"))
        self.encoder.eval().to(device)

        # D. 加载 Projector
        print("Loading Projector weights...")
        llm_dim = self.llm.config.hidden_size
        vis_dim = 1024
        
        self.projector = MMInputProjector(input_dim=vis_dim, output_dim=llm_dim).to(device)
        # 加载训练好的 projector.pt
        projector_path = os.path.join(model_dir, "projector.pt")
        if os.path.exists(projector_path):
            self.projector.load_state_dict(torch.load(projector_path, map_location=device))
        else:
            raise FileNotFoundError(f"Projector weights not found at {projector_path}")
        self.projector.eval()
        self.projector.to(dtype=torch.bfloat16)
        
        # E. 【新增】加载 Fusion Module
        print("Loading Cross-Attention Fusion weights...")
        self.fusion_module = CrossAttentionFusion(
            hidden_size=llm_dim,
            num_heads=32, # 需与训练时一致
            dropout=0.0   # 预测时 dropout 设为 0 也没关系，因为会 eval()
        ).to(device)
        
        fusion_path = os.path.join(model_dir, "fusion_module.pt")
        if os.path.exists(fusion_path):
            self.fusion_module.load_state_dict(torch.load(fusion_path, map_location=device))
        else:
            # 如果没找到，如果是刚开始调试，可能需要检查路径；或者暂时注释掉报错
            raise FileNotFoundError(f"Fusion weights not found at {fusion_path}")
        self.fusion_module.eval()
        self.fusion_module.to(dtype=torch.bfloat16)

        print("Model loaded successfully!")

    def predict(self, video_path, audio_path, text_utterance):
        imagebind_inputs = {
            ModalityType.VISION: data.load_and_transform_video_data([video_path], self.device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([audio_path], self.device)
        }
        with torch.no_grad():
            multimodal_embeds = self.encoder(imagebind_inputs)

            vision_embeds = multimodal_embeds[ModalityType.VISION]
            audio_embeds = multimodal_embeds[ModalityType.AUDIO]

            vision_embeds = vision_embeds.unsqueeze(1)
            audio_embeds = audio_embeds.unsqueeze(1)

            mm_inputs = torch.cat([vision_embeds, audio_embeds], dim=1)
            mm_inputs = mm_inputs.to(dtype=torch.bfloat16)
            mm_embeds = self.projector(mm_inputs)

        # 2. 准备文本 Prompt
        # 构造 prompt，注意这里没有 Answer，只有 User 的提问
        instruction = (
                f"Analyze the user intent based on the video and the following text: '{text_utterance}'. "
                f"Answer with the label only."
            )
        messages = [{"role": "user", "content": instruction}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        with torch.no_grad():
            # 获取文本 Embedding
            # 注意：llm 是 PeftModel，可以通过 get_input_embeddings 获取
            text_embeds = self.llm.get_input_embeddings()(input_ids)
            
            #fusion
            fusion_output = self.fusion_module(text_embeds, mm_embeds)

            # 3. 拼接 [Image, Text]
            inputs_embeds = torch.cat([mm_embeds, fusion_output, text_embeds], dim=1)
            
            # 4. 生成 Attention Mask
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

            # 5. 生成 (Generate)
            generate_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=10,      # 意图标签通常很短
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=terminators
            )

        # 6. 解码
        output_text = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        return output_text.strip()

# ==========================================
# 3. 使用示例
# ==========================================
def evaluate():
    set_seed(42) #设种子
    
    # 1. 检查路径
    if not os.path.exists(CONFIG['test_data_path']):
        raise FileNotFoundError(f"Test data not found: {CONFIG['test_data_path']}")
    
    # 2. 初始化模型
    predictor = IntentPredictor(
        model_dir=CONFIG['checkpoint_dir'],
        base_model_path=CONFIG['base_model_path'],
        vision_model_path=CONFIG['vision_model_path'],
        device=CONFIG['device']
    )

    # 3. 加载数据
    print(f"Loading test data from {CONFIG['test_data_path']}...")
    with open(CONFIG['test_data_path'], 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 调试：只测前5条 (正式跑时注释掉下面这行)
    # test_data = test_data[:5] 

    results = []

    print("Starting inference...")
    # 使用 tqdm 显示进度条
    for item in tqdm(test_data, desc="Predicting"):
        video_path = item['video_path']
        audio_path = item['audio_path']
        text = item['text']
        true_label = item['label']

        try:
            # 进行预测
            predicted_label = predictor.predict(
                video_path=video_path,
                audio_path=audio_path,
                text_utterance=text
            )
            
            # 保存结果
            result_item = {
                "label": true_label,
                "predict": predicted_label,
            }
            results.append(result_item)
            
        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            results.append({
                "video_path": video_path,
                "error": str(e)
            })

    # 4. 保存结果到文件
    os.makedirs(os.path.dirname(CONFIG['output_file']), exist_ok=True)
    with open(CONFIG['output_file'], 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 5. 打印统计信息
    print("\n" + "="*30)
    print(f"Results saved to: {CONFIG['output_file']}")
    print("="*30)

if __name__ == "__main__":
    evaluate()
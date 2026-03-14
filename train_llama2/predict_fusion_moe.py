import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    SiglipVisionModel, 
    SiglipImageProcessor
)
from peft import PeftModel
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm  # 引入进度条
import json
from transformers import set_seed

from model.projector import MMInputProjector
from model.moe_vision_adapter import VisionMoEAdapter
from model.fusion import CrossAttentionFusion

CONFIG = {
    "base_model_path": "/root/huggingface/llama/Meta-Llama-3-8B-Instruct", 
    "vision_model_path": "/root/huggingface/google/siglip-so400m-patch14-384", 
    "checkpoint_dir": "./checkpoints/llama_mintrec_fusion_moe/checkpoint-501", 
    "test_data_path": "/root/user/xyh/Datasets/MIntRec/MIntRec_test.json", 
    "output_file": "./eval/mintrec_predictions_fusion_moe.json",
    "device": "cuda",
    "num_frames": 4
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
        print(f"Loading Vision Encoder from {vision_model_path}...")
        self.vision_encoder = SiglipVisionModel.from_pretrained(vision_model_path, torch_dtype=torch.bfloat16).to(device)
        self.vision_processor = SiglipImageProcessor.from_pretrained(vision_model_path)
        self.vision_encoder.eval()

        # D. 加载 Projector
        print("Loading Projector weights...")
        llm_dim = self.llm.config.hidden_size
        vis_dim = self.vision_encoder.config.hidden_size
        
        self.projector = MMInputProjector(input_dim=vis_dim, output_dim=llm_dim).to(device)
        # 加载训练好的 projector.pt
        projector_path = os.path.join(model_dir, "projector.pt")
        if os.path.exists(projector_path):
            self.projector.load_state_dict(torch.load(projector_path, map_location=device))
        else:
            raise FileNotFoundError(f"Projector weights not found at {projector_path}")
        self.projector.eval()
        self.projector.to(dtype=torch.bfloat16)

        #E. 加载moe_adapter
        self.vision_moe = VisionMoEAdapter(input_dim=llm_dim, num_experts=4, top_k=2).to(device).to(torch.bfloat16)
        # 加载训练好的 vision_moe.pt
        moe_path = os.path.join(model_dir, "vision_moe.pt")
        if os.path.exists(moe_path):
            self.vision_moe.load_state_dict(torch.load(moe_path, map_location=device))
        else:
            raise FileNotFoundError(f"MoE Adapter weights not found at {moe_path}")
        self.vision_moe.eval()

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

    def _load_frames(self, video_path, num_frames=4):
        """读取视频帧，与训练时逻辑保持一致"""
        frames = []
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return [Image.new('RGB', (224, 224)) for _ in range(num_frames)]
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
             return [Image.new('RGB', (224, 224)) for _ in range(num_frames)]

        indices = np.linspace(0, total_frames-1, num_frames).astype(int)
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            else:
                frames.append(Image.new('RGB', (224, 224)))
        cap.release()
        
        # 补全
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
            
        return frames[:num_frames]

    def predict(self, video_path, text_utterance, num_frames=4):
        # 1. 准备视觉特征
        frames = self._load_frames(video_path, num_frames)
        vision_inputs = self.vision_processor(images=frames, return_tensors="pt")
        pixel_values = vision_inputs.pixel_values.to(self.device).to(torch.bfloat16)
        
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)

        b, t, c, h, w = pixel_values.shape
        pixel_values_flat = pixel_values.view(-1, c, h, w)
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values_flat)
            image_embeds = vision_outputs.last_hidden_state  
                
            vis_seq_len = image_embeds.shape[1]
            vis_dim = image_embeds.shape[2]
            image_embeds = image_embeds.view(b, t * vis_seq_len, vis_dim)

            image_embeds = self.projector(image_embeds)

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
            fusion_output = self.fusion_module(text_embeds, image_embeds)
            
        #插入moe
        batch_size, total_seq, hidden_dim = image_embeds.shape
        image_embeds_flat = image_embeds.view(-1, hidden_dim)
        with torch.no_grad():
            image_embeds_flat = self.vision_moe(image_embeds_flat)
            image_embeds = image_embeds_flat.view(batch_size, total_seq, hidden_dim)

            # 3. 拼接 [Image, Text]
            inputs_embeds = torch.cat([image_embeds, fusion_output, text_embeds], dim=1)
            
            # 4. 生成 Attention Mask
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

            # 5. 生成 (Generate)
            generate_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=10,      # 意图标签通常很短
                do_sample=False,        # 预测通常用贪婪搜索以保证确定性
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
        text = item['text']
        true_label = item['label']

        try:
            # 进行预测
            predicted_label = predictor.predict(
                video_path=video_path,
                text_utterance=text,
                num_frames=CONFIG['num_frames']
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
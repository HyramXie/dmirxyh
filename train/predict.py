import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration
from safetensors.torch import load_file # 必须导入这个来加载 safetensors

# 导入你定义的模型类
from model.qwen2_5vl_moe import Qwen2_5_VL_MIntRec

# --- 配置 ---
CONFIG = {
    # "moe_weights_name": "./checkpoints/mintrec_moe/vision_moe_final.pt",          # MoE 权重文件名
    "test_data_path": "/root/user/xyh/LLaMA-Factory-main/data/MELD_test.json",  # 测试集路径
    "checkpoint_dir": "./checkpoints/meld_moe/checkpoint-2622", # 训练保存的目录
    "output_file": "./eval/meld_predictions.json",    # 结果保存路径
    "model_path": "/root/huggingface/qwen/Qwen2.5-VL-7B-Instruct"
}

class MIntRecTestDataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item['messages']
        text = messages[0]["content"]
        label = messages[1]["content"] # 假设 label 是意图字符串
        
        video_path = item["videos"][0]
        
        # 构造 Prompt (注意：这里没有 Assistant 的回复，只有 User 的提问)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 64 * 28 * 28, # 控制分辨率以节省显存
                        "min_pixels": 32 * 28 * 28,
                        "fps": 1.0, # 抽帧率
                    },
                    {"type": "text", "text": text}
                ],
            }
        ]
        
        # 预处理文本
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 预处理视觉
        image_inputs, video_inputs = process_vision_info(messages)
        
        return {
            "text_input": text_input,
            "images": image_inputs,
            "videos": video_inputs,
            "label": label
        }

def load_trained_model():
    print("Loading base model structure...")
    # 1. 初始化空模型
    # 这一步会加载 Qwen 底座 (4-bit) + 随机初始化的 LoRA + 随机初始化的 MoE
    # 注意：device_map="auto" 或 "cuda" 取决于你的显存，单卡建议用 "cuda"
    model_wrapper = Qwen2_5_VL_MIntRec(device="cuda")
    
    # 2. 定位权重文件
    # 你的目录里有 model.safetensors，所以我们要加载它
    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "model.safetensors")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ 找不到权重文件: {checkpoint_path}")

    print(f"Loading full checkpoint from {checkpoint_path}...")
    
    # 3. 加载权重
    # 使用 safetensors 库加载
    state_dict = load_file(checkpoint_path)
    
    # 4. 将权重灌入模型
    # strict=False 是为了忽略一些不匹配的 key (比如 Qwen 底座里的某些 buffer)
    # 只要 LoRA (lora_A, lora_B) 和 MoE (vision_moe) 的 key 能对上就行
    missing_keys, unexpected_keys = model_wrapper.load_state_dict(state_dict, strict=False)
    
    print(f"✅ Weights loaded successfully.")
    print(f"   Missing keys: {len(missing_keys)} (Expected if base model is frozen)")
    print(f"   Unexpected keys: {len(unexpected_keys)}")
    
    # 切换到评估模式
    model_wrapper.eval()
    
    return model_wrapper


def run_prediction():
    # 1. 准备 Processor (必须与训练时参数一致)
    model_path = CONFIG["model_path"]
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

    # 2. 加载模型
    model_wrapper = load_trained_model()
    # 提取核心模型用于 generate (Qwen2_5_VL_MIntRec 是 nn.Module，没有 generate 方法)
    # core_model 是 PeftModel，它有 generate 方法
    # 此时 core_model.visual.forward 已经被我们 Patch 过了，所以 generate 会自动用到 MoE
    inference_model = model_wrapper.core_model 

    # 3. 准备数据
    test_dataset = MIntRecTestDataset(CONFIG["test_data_path"], processor)
    # Batch size 设为 1 最稳，视频推理显存占用大
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    results = []

    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # 因为 batch_size=1，这里直接取第一个
            item = batch[0]
            
            # 准备输入
            inputs = processor(
                text=[item["text_input"]],
                images=item["images"],
                videos=item["videos"],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # 生成
            generated_ids = inference_model.generate(
                **inputs, 
                max_new_tokens=16,  # 标签通常很短
                do_sample=False     # 贪婪解码，保证结果确定性
            )
            
            # 解码
            # generated_ids 包含了 input 的部分，我们需要截断
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            pred_label = output_text.strip()
            gold_label = item["label"]
            
            results.append({
                "predict": pred_label,
                "label": gold_label
            })

    # 4. 保存结果
    with open(CONFIG["output_file"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {CONFIG['output_file']}")

if __name__ == "__main__":
    run_prediction()
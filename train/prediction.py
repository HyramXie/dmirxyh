import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# 导入你定义的模型类
from model_provider import Qwen2_5_VL_MIntRec

# --- 配置 ---
CONFIG = {
    "test_data_path": "./MIntRec/test.json",  # 测试集路径
    "video_dir": "./MIntRec/video",           # 视频目录
    "checkpoint_dir": "./checkpoints/mintrec_lora_moe", # 训练保存的目录
    "moe_weights_name": "vision_moe_final.pt",          # MoE 权重文件名
    "output_file": "test_predictions.csv",    # 结果保存路径
    "image_max_pixels": 256 * 28 * 28,        # 必须与训练时一致
    "video_max_pixels": 128 * 28 * 28,        # 必须与训练时一致
}

class MIntRecTestDataset(Dataset):
    def __init__(self, json_path, video_dir, processor):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.video_dir = video_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = item['video_id']
        text = item['text']
        label = item['label'] # 测试集通常也有 label 用于计算指标，如果没有则忽略
        
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        
        # 构造 Prompt (注意：这里没有 Assistant 的回复，只有 User 的提问)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": CONFIG["video_max_pixels"],
                        "fps": 1.0, 
                    },
                    {"type": "text", "text": f"Analyze the speaker's intent: '{text}'. Answer with label."}
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
            "video_id": video_id,
            "text_input": text_input,
            "images": image_inputs,
            "videos": video_inputs,
            "label": label
        }

def load_trained_model():
    print("Loading base model and initializing MoE...")
    # 1. 初始化模型 (这一步会加载 Base Model + 4bit 量化 + 随机初始化的 LoRA/MoE)
    # 注意：这里加载的是基础模型的权重，LoRA 和 MoE 还是初始状态
    model_wrapper = Qwen2_5_VL_MIntRec(device="cuda")
    
    print("Loading trained LoRA weights...")
    # 2. 加载训练好的 LoRA 权重覆盖掉随机初始化的 LoRA
    # model_wrapper.core_model 已经是 PeftModel 了
    model_wrapper.core_model.load_adapter(CONFIG["checkpoint_dir"], adapter_name="default")
    
    print("Loading trained MoE weights...")
    # 3. 加载训练好的 MoE 权重
    moe_path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["moe_weights_name"])
    if os.path.exists(moe_path):
        state_dict = torch.load(moe_path, map_location="cuda")
        model_wrapper.vision_moe.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"MoE weights not found at {moe_path}")
        
    model_wrapper.eval()
    return model_wrapper

def run_prediction():
    # 1. 准备 Processor (必须与训练时参数一致)
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_path, 
        min_pixels=256, 
        max_pixels=CONFIG["image_max_pixels"]
    )

    # 2. 加载模型
    model_wrapper = load_trained_model()
    # 提取核心模型用于 generate (Qwen2_5_VL_MIntRec 是 nn.Module，没有 generate 方法)
    # core_model 是 PeftModel，它有 generate 方法
    # 此时 core_model.visual.forward 已经被我们 Patch 过了，所以 generate 会自动用到 MoE
    inference_model = model_wrapper.core_model 

    # 3. 准备数据
    test_dataset = MIntRecTestDataset(CONFIG["test_data_path"], CONFIG["video_dir"], processor)
    # Batch size 设为 1 最稳，视频推理显存占用大
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    results = []
    correct_count = 0
    total_count = 0

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
            
            # 简单统计准确率
            if pred_label.lower() == gold_label.lower():
                correct_count += 1
            total_count += 1
            
            results.append({
                "video_id": item["video_id"],
                "prediction": pred_label,
                "ground_truth": gold_label,
                "is_correct": pred_label.lower() == gold_label.lower()
            })

    # 4. 保存结果
    df = pd.DataFrame(results)
    df.to_csv(CONFIG["output_file"], index=False)
    
    acc = correct_count / total_count if total_count > 0 else 0
    print(f"Inference Done! Accuracy: {acc:.4f}")
    print(f"Results saved to {CONFIG['output_file']}")

if __name__ == "__main__":
    run_prediction()
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class MIntRecDataset(Dataset):
    def __init__(self, data_json_path, tokenizer):
        """
        data_json_path: 包含 [{"video_path": "...", "text": "...", "label": "..."}] 的列表
        """
        self.data = self._load_json(data_json_path)
        self.tokenizer = tokenizer

    def _load_json(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 准备视频输入
        video_path = item["video_path"]
        audio_path = item["audio_path"]
        
        # 2. 准备文本输入
        input_text = item['text']
        label_text = item['label']
        
        instruction = (
            f"Analyze the user intent based on the video and the following text: '{input_text}'. "
            f"Answer with the label only."
        )

        messages = [
            {"role": "user", "content": instruction}
        ]
        
        # 3. 生成 input_ids (Prompt + Label)
        
        # A. 仅对 Prompt 进行编码 (add_generation_prompt=True 会加上 Assistant 的引导符)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids

        # 加上 eot
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        label_ids_only = self.tokenizer(label_text, add_special_tokens=False).input_ids
        target_ids = label_ids_only + [eot_id]

        # 4. 拼接
        input_ids = prompt_ids + target_ids
        
        # 5. 制作 Labels (Masking)
        # Prompt 部分设为 -100，Label 部分保留
        context_length = len(prompt_ids)
        labels = [-100] * context_length + target_ids

                
        # 转换为 Tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "video_path": video_path,
            "audio_path": audio_path
        }

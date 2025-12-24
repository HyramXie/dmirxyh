import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class MIntRecDataset(Dataset):
    def __init__(self, data_json_path, tokenizer, image_processor, num_frames=4):
        """
        data_json_path: 包含 [{"video_path": "...", "text": "...", "label": "..."}] 的列表
        """
        self.data = self._load_data(data_json_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_frames = num_frames

    def _load_data(self, path):
        # 如果是真实文件请使用 json.load
        with open(path, 'r') as f:
            return json.load(f)

    def _load_frames(self, video_path):
        frames = []
        if not os.path.exists(video_path):
            # 如果找不到文件，生成黑帧 (为了代码不报错)
            return [Image.new('RGB', (224, 224)) for _ in range(self.num_frames)]
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
             return [Image.new('RGB', (224, 224)) for _ in range(self.num_frames)]

        # 均匀采样
        indices = np.linspace(0, total_frames-1, self.num_frames).astype(int)
        
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
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
            
        return frames[:self.num_frames]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 准备视频输入
        video_path = item["video_path"]
        frames = self._load_frames(video_path) # List[PIL.Image]
        
        # Siglip Processor 处理图片列表
        # return: {'pixel_values': [T, C, H, W]}
        vision_inputs = self.image_processor(images=frames, return_tensors="pt")
        pixel_values = vision_inputs.pixel_values # [T, 3, H, W]
        
        # 2. 准备文本输入
        messages = [
            {"role": "user", "content": item['text']},
            {"role": "assistant", "content": item['label']}
        ]
        
        # 3. 使用 apply_chat_template 生成文本
        # 技巧：我们需要知道“提问部分”有多长，以便 mask 掉 labels
        # A. 获取完整对话的 input_ids
        full_input_ids = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False, 
            return_tensors="pt"
        ).squeeze(0) # [Seq_Len]

        # B. 获取“Prompt部分”（User输入 + Assistant引导符）的长度
        # Qwen 的 template 会在 user 结束后自动加上 <|im_start|>assistant\n
        user_only_messages = messages[:-1] # 去掉 assistant 的回复
        prompt_input_ids = self.tokenizer.apply_chat_template(
            user_only_messages, 
            tokenize=True, 
            add_generation_prompt=True, # 关键：加上这句会自动补全 <|im_start|>assistant\n
            return_tensors="pt"
        ).squeeze(0)
        
        prompt_len = prompt_input_ids.shape[0]

        # 4. 处理 Labels (Masking)
        labels = full_input_ids.clone()
        # 将 Prompt 部分（包括 System, User, 和 Assistant 的引导符）设为 -100
        labels[:prompt_len] = -100
        
        # 5. 获取 Attention Mask
        attention_mask = torch.ones_like(full_input_ids)
        
        return {
            "input_ids": full_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values
        }


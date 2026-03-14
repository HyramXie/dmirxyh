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
        self.data = self._load_json(data_json_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_frames = num_frames
        
        # 定义意图标签集合，用于提示模型
        label_path = os.path.join("/root/user/xyh/train_llama/data", data_json_path.split("Datasets/")[1].split("/")[0] + ".json")
        self.intent_labels = self._load_json(label_path)
        self.labels_str = ", ".join(self.intent_labels)
        print(f"Loaded {self.intent_labels} intent labels.")

    def _load_json(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
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

        # # 加上 eot
        # eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # label_ids_only = self.tokenizer(label_text, add_special_tokens=False).input_ids
        # target_ids = label_ids_only + [eot_id]

        # B. 对 Label 进行编码 (关键：手动加上 EOS)
        # 注意：这里 label 前面可能需要一个空格，取决于 tokenizer，但通常直接编码即可
        target_text = label_text + self.tokenizer.eos_token
        target_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids

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
            "pixel_values": pixel_values
        }


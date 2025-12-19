import json
import torch
from torch.utils.data import Dataset
from transformers import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info 

class MIntRecDataset(Dataset):
    def __init__(self, json_path, video_dir, processor):
        """
        json_path: MIntRec 的标注文件 (train.json)
        video_dir: 视频文件夹路径
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.video_dir = video_dir
        self.processor = processor
        
        # MIntRec 意图标签列表 (示例)
        self.intents = [
            "complain", "praise", "apologize", "thank", "criticize", 
            "agree", "disagree", "ask", "answer", "taunt" # 请根据实际数据集补充完整
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item['messages']
        text = messages[0]["content"]
        label = messages[1]["content"] # 假设 label 是意图字符串
        
        video_path = item["videos"][0]
        
        # 构造对话格式
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420, # 控制分辨率以节省显存
                        "fps": 1.0, # 抽帧率
                    },
                    {"type": "text", "text": f"Analyze the speaker's intent in the video and text: '{text}'. Answer with the intent label directly."}
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}]
            }
        ]
        
        # 预处理输入 (Input)
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # 提取 Vision Info
        image_inputs, video_inputs = process_vision_info(messages)
        
        return {
            "text_input": text_input,
            # "images": image_inputs,
            "videos": video_inputs,
            "messages": messages # 保留原始 msg 用于 debug 或 collate
        }
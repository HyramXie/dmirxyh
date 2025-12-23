import os
from torch.utils.data import Dataset
from PIL import Image

class MIntRecDataset(Dataset):
    def __init__(self, data_json_path, tokenizer, image_processor, num_frames=4):
        """
        data_json_path: 包含 [{"video_id": "...", "text": "...", "label": "..."}] 的列表
        """
        self.data = self._load_data(data_json_path)
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_frames = num_frames
        
        # MIntRec Intent Labels (根据实际情况修改)
        self.intents = [
            'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 'Care', 'Agree', 
            'Taunt', 'Flaunt', 'Joke', 'Oppose', 'Comfort', 'Inform', 'Advise', 
            'Arrange', 'Introduce', 'Leave', 'Prevent', 'Greeting', 'Ask for help'
        ]

    def _load_data(self, path):
        # 如果是真实文件请使用 json.load
        # 这里为了演示，生成伪数据
        if not os.path.exists(path):
            print("Warning: Data file not found. Using dummy data.")
            return [{"video_id": "dummy.mp4", "text": "Why did you do that?", "label": "Complain"}] * 10
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
        # 假设 video_id 包含后缀，如 'train_001.mp4'
        video_path = os.path.join(self.video_dir, item['video_id'])
        frames = self._load_frames(video_path) # List[PIL.Image]
        
        # Siglip Processor 处理图片列表
        # return: {'pixel_values': [T, C, H, W]}
        vision_inputs = self.image_processor(images=frames, return_tensors="pt")
        pixel_values = vision_inputs.pixel_values # [T, 3, H, W]
        
        # 2. 准备文本输入
        prompt = f"<|im_start|>user\nAnalyze the video and text to determine the intent.\nText: {item['text']}\n<|im_end|>\n<|im_start|>assistant\nThe intent is {item['label']}<|im_end|>"
        
        # Tokenize
        # 注意：这里我们手动拼接 prompt 和 answer，在 collator 里再处理 padding
        encodings = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encodings.input_ids.squeeze(0)
        attention_mask = encodings.attention_mask.squeeze(0)
        
        # Labels: 计算 Loss
        # 简单的做法：Label = Input_ids, 但将 Prompt 部分设为 -100
        labels = input_ids.clone()
        
        # 找到 "The intent is " 的位置来截断 label (简化处理，实际可以使用更复杂的掩码逻辑)
        # 这里简单假设只训练最后几个 token（意图标签）
        # 更严谨的做法是分段 tokenize prompt 和 answer
        
        # 重新构建更稳健的 mask 逻辑：
        user_prompt = f"<|im_start|>user\nAnalyze the video and text to determine the intent.\nText: {item['text']}\n<|im_end|>\n<|im_start|>assistant\nThe intent is "
        prompt_ids = self.tokenizer(user_prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        labels[:len(prompt_ids)] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values
        }


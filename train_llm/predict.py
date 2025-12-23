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

# ==========================================
# 1. 必须复制 Projector 定义以加载权重
# ==========================================
class MMInputProjector(nn.Module):
    def __init__(self, input_dim=1152, output_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. 推理类定义
# ==========================================
class IntentPredictor:
    def __init__(self, 
                 model_dir, 
                 base_model_path="Qwen/Qwen2.5-7B-Instruct", 
                 vision_model_path="google/siglip-so400m-patch14-384", 
                 device="cuda"):
        
        self.device = device
        print(f"Loading Base Qwen Model from {base_model_path}...")
        
        # A. 加载 Base LLM
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
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
        self.vision_encoder = SiglipVisionModel.from_pretrained(vision_model_path).to(device)
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
        pixel_values = vision_inputs.pixel_values.to(self.device) # [T, 3, H, W]
        
        with torch.no_grad():
            # Vision Encoder
            # [T, C, H, W] -> [T, Seq, Dim]
            vision_outputs = self.vision_encoder(pixel_values)
            image_embeds = vision_outputs.last_hidden_state
            
            # Flatten & Project
            # [T, Seq, Dim] -> [1, T*Seq, Dim] -> [1, T*Seq, LLM_Dim]
            T, S, D = image_embeds.shape
            image_embeds = image_embeds.view(1, T * S, D) 
            image_embeds = self.projector(image_embeds)

        # 2. 准备文本 Prompt
        # 构造 prompt，注意这里没有 Answer，只有 User 的提问
        prompt = f"<|im_start|>user\nAnalyze the video and text to determine the intent.\nText: {text_utterance}\n<|im_end|>\n<|im_start|>assistant\nThe intent is"
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            # 获取文本 Embedding
            # 注意：llm 是 PeftModel，可以通过 get_input_embeddings 获取
            text_embeds = self.llm.get_input_embeddings()(input_ids)
            
            # 3. 拼接 [Image, Text]
            inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            
            # 4. 生成 Attention Mask
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

            # 5. 生成 (Generate)
            generate_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=10,      # 意图标签通常很短
                do_sample=False,        # 预测通常用贪婪搜索以保证确定性
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 6. 解码
        output_text = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        return output_text.strip()

# ==========================================
# 3. 使用示例
# ==========================================
if __name__ == "__main__":
    # 配置路径
    # 这里指向 train.py 训练完保存的目录
    MODEL_DIR = "output_qwen_siglip" 
    
    # 你的测试数据
    test_video = "mintrec_videos/test_video_001.mp4"
    test_text = "I really appreciate your help with this project."
    
    # 初始化
    try:
        predictor = IntentPredictor(model_dir=MODEL_DIR)
        
        # 预测
        print("-" * 30)
        print(f"Video: {test_video}")
        print(f"Text: {test_text}")
        
        intent = predictor.predict(test_video, test_text)
        
        print("-" * 30)
        print(f"Predicted Intent: {intent}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error: {e}")
        print("请确保已运行 train.py 并在 output_qwen_siglip 中生成了模型文件。")
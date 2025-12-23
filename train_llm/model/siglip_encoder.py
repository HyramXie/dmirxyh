import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipImageProcessor

class SiglipEncoder(nn.Module):
    def __init__(self, model_name="google/siglip-so400m-patch14-384", device="cuda"):
        super().__init__()
        self.device = device
        # 加载 SIGLIP 视觉部分
        self.model = SiglipVisionModel.from_pretrained(model_name).to(device)
        self.processor = SiglipImageProcessor.from_pretrained(model_name)
        
        # 冻结视觉编码器
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        """
        images: List of PIL Images or Tensor [B, T, C, H, W] for video frames
        这里假设输入已经是预处理好的 Tensor [B, C, H, W] 代表视频的关键帧或平均帧
        """
        # 如果输入是像素值
        if isinstance(images, torch.Tensor):
            pixel_values = images.to(self.device)
        else:
            # 如果是图片列表，使用 processor 处理
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            # SIGLIP 输出通常是 [B, Seq_len, Hidden_dim]
            # 我们取 pooler_output 或者 序列平均，这里取 last_hidden_state
            image_embeds = outputs.last_hidden_state # [B, 729, 1152] 
        
        return image_embeds
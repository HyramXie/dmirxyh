import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class QwenWithSiglip(nn.Module):
    def __init__(self, 
                 llm_path="Qwen/Qwen2.5-7B-Instruct", 
                 vision_path="google/siglip-so400m-patch14-384",
                 device="cuda"):
        super().__init__()
        self.device = device
        
        # 1. 加载 LLM
        print("Loading Qwen...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, 
            torch_dtype=torch.bfloat16, # 建议使用 bf16
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        
        # 2. 加载 Vision Encoder
        print("Loading SIGLIP...")
        self.vision_encoder = SiglipEncoder(model_name=vision_path, device=device)
        
        # 3. 加载 Projector
        print("Initializing Projector...")
        # 获取 LLM 的 hidden size 和 Vision 的 hidden size
        llm_hidden_size = self.llm.config.hidden_size
        vision_hidden_size = self.vision_encoder.model.config.hidden_size
        self.projector = MMInputProjector(input_dim=vision_hidden_size, output_dim=llm_hidden_size).to(device)

        # 4. 配置训练参数（冻结策略）
        self._set_trainable_params()
        
    def _set_trainable_params(self):
        # A. 冻结 LLM 原生参数
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # B. 冻结 Vision Encoder (已在类内部处理，这里再次确认)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # C. 激活 Projector 参数
        for param in self.projector.parameters():
            param.requires_grad = True
            
        # D. 应用 LoRA 到 LLM
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            # Qwen 的核心模块
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        """
        input_ids: [B, Seq_Len] 文本
        pixel_values: [B, C, H, W] 图像/视频帧
        labels: [B, Seq_Len]
        """
        # 1. 获取视觉特征 [B, ViT_Seq, Vis_Dim]
        image_embeds = self.vision_encoder(pixel_values) 
        
        # 2. 投影到 LLM 空间 [B, ViT_Seq, LLM_Dim]
        image_embeds = self.projector(image_embeds)
        
        # 3. 获取文本 Embeddings [B, Text_Seq, LLM_Dim]
        # 注意：这里需要通过 llm.model.embed_tokens 获取，因为 llm 现在包裹了 LoRA
        # Qwen2 的结构通常是 model.model.embed_tokens
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 4. 拼接策略: [Image, Text]
        # 这里的实现方式是简单的拼接。更复杂的做法是使用特殊的 <image> token 占位并替换
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # 5. 调整 Attention Mask
        # 视觉部分的 mask 全是 1
        vision_mask = torch.ones((image_embeds.shape[0], image_embeds.shape[1]), device=self.device)
        attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        
        # 6. 调整 Labels
        # 视觉部分的 label 设为 -100 (不计算 loss)
        if labels is not None:
            vision_labels = torch.full((image_embeds.shape[0], image_embeds.shape[1]), -100, device=self.device)
            labels = torch.cat([vision_labels, labels], dim=1)
        
        # 7. 前向传播
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
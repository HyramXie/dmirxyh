import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data
from model.projector import MMInputProjector

class QwenWithSiglip(nn.Module):
    def __init__(self, llm_path="Qwen/Qwen2.5-7B-Instruct", vision_path="google/siglip-so400m-patch14-384", device="cuda"):
        super().__init__()
        self.device = device
        
        # 1. 加载 LLM
        print("Loading Qwen...")
        print("Configuring QLoRA (4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 # 计算时使用 bf16
        )
        
        # 2. 加载 LLM (带量化配置)
        print(f"Loading Qwen (4-bit): {llm_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, 
            quantization_config=bnb_config, # <--- 关键：应用量化
            device_map="auto",              # <--- 关键：自动分配设备
            torch_dtype=torch.bfloat16
        )
        self.llm = prepare_model_for_kbit_training(self.llm)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)     
        # Llama-3 必须手动设置 pad_token，通常使用 eos_token 作为 pad
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id    

        # 2. 加载 ImageBind encoder
        print("Loading ImageBind...")
        self.encoder = imagebind_model.imagebind_huge(pretrained=False)
        self.encoder.load_state_dict(torch.load(vision_path, map_location="cpu"))
        self.encoder.eval().to(device)
        
        # 3. 加载 Projector
        print("Initializing Projector...")
        # 获取 LLM 的 hidden size 和 Vision 的 hidden size
        llm_hidden_size = self.llm.config.hidden_size
        self.projector = MMInputProjector(input_dim=1024, output_dim=llm_hidden_size).to(device)

        # 4. 配置训练参数（冻结策略）
        self._set_trainable_params()
        
    def _set_trainable_params(self):
        # A. 冻结 LLM 原生参数
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # B. 冻结 Vision Encoder (已在类内部处理，这里再次确认)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # C. 激活 Projector 参数
        for param in self.projector.parameters():
            param.requires_grad = True
            
        # D. 应用 LoRA 到 LLM
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            # Qwen 的核心模块
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.llm = get_peft_model(self.llm, peft_config)

        self.llm.gradient_checkpointing_enable()
        self.llm.config.use_cache = False  # 必须关，否则无效

        self.llm.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, video_paths, audio_paths, labels=None):
        imagebind_inputs = {
            ModalityType.VISION: data.load_and_transform_video_data(video_paths, self.device),
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device)
        }
        with torch.no_grad():
            multimodal_embeds = self.encoder(imagebind_inputs)

        vision_embeds = multimodal_embeds[ModalityType.VISION]
        audio_embeds = multimodal_embeds[ModalityType.AUDIO]

        vision_embeds = vision_embeds.unsqueeze(1)
        audio_embeds = audio_embeds.unsqueeze(1)

        mm_inputs = torch.cat([vision_embeds, audio_embeds], dim=1)
        
        # 2. 投影到 LLM 空间 [B, ViT_Seq, LLM_Dim]
        mm_embeds = self.projector(mm_inputs)
        
        # 3. 获取文本 Embeddings [B, Text_Seq, LLM_Dim]
        # 注意：这里需要通过 llm.model.embed_tokens 获取，因为 llm 现在包裹了 LoRA
        # Qwen2 的结构通常是 model.model.embed_tokens
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 4. 拼接策略: [Image, Text]
        # 这里的实现方式是简单的拼接。更复杂的做法是使用特殊的 <image> token 占位并替换
        inputs_embeds = torch.cat([mm_embeds, text_embeds], dim=1)
        
        # 5. 调整 Attention Mask
        # 视觉部分的 mask 全是 1
        vision_mask = torch.ones((mm_embeds.shape[0], mm_embeds.shape[1]), device=self.device)
        attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        
        # 6. 调整 Labels
        # 视觉部分的 label 设为 -100 (不计算 loss)
        if labels is not None:
            vision_labels = torch.full((mm_embeds.shape[0], mm_embeds.shape[1]), -100, device=self.device)
            labels = torch.cat([vision_labels, labels], dim=1)
        
        # 7. 前向传播
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from model.siglip_encoder import SiglipEncoder
from model.projector import MMInputProjector
from model.moe_vision_adapter import VisionMoEAdapter
from model.fusion import CrossAttentionFusion

class LLaMAWithSiglip(nn.Module):
    def __init__(self, llm_path="LLaMA/Meta-Llama-3-8B-Instruct", vision_path="google/siglip-so400m-patch14-384", device="cuda", use_moe=False, use_fusion=False):
        super().__init__()
        self.device = device
        self.use_moe = use_moe
        self.use_fusion = use_fusion
        
        # 1. 加载 LLM (4-bit QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, 
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.llm = prepare_model_for_kbit_training(self.llm)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 2. 视觉编码器与投影层
        self.vision_encoder = SiglipEncoder(model_name=vision_path, device=device)
        llm_hidden_size = self.llm.config.hidden_size
        vision_hidden_size = self.vision_encoder.model.config.hidden_size
        self.projector = MMInputProjector(input_dim=vision_hidden_size, output_dim=llm_hidden_size).to(device)

        # 3. 条件加载 MoE 模块
        if self.use_moe:
            self.vision_moe = VisionMoEAdapter(input_dim=llm_hidden_size, num_experts=2, top_k=2).to(self.llm.device).to(torch.bfloat16)

        # 4. 条件加载融合模块
        if self.use_fusion:
            self.fusion_module = CrossAttentionFusion(
                hidden_size=llm_hidden_size,
                num_heads=32,
                dropout=0.1,
                max_frames=4
            ).to(self.llm.device).to(torch.bfloat16)

        self._set_trainable_params()
        
    def _set_trainable_params(self):
        # 冻结基础模型
        for param in self.llm.parameters(): param.requires_grad = False
        for param in self.vision_encoder.parameters(): param.requires_grad = False
            
        # 激活投影层
        for param in self.projector.parameters(): param.requires_grad = True
            
        # 根据开关激活对应模块
        if self.use_moe:
            for param in self.vision_moe.parameters(): param.requires_grad = True
        
        if self.use_fusion:
            for param in self.fusion_module.parameters(): param.requires_grad = True
            
        # 配置 LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.gradient_checkpointing_enable()
        self.llm.config.use_cache = False

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        b, t, c, h, w = pixel_values.shape
        pixel_values_flat = pixel_values.view(-1, c, h, w)
        
        # 视觉特征提取
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values_flat)
            image_embeds = vision_outputs if isinstance(vision_outputs, torch.Tensor) else vision_outputs.last_hidden_state
        
        vis_seq_len, vis_dim = image_embeds.shape[1], image_embeds.shape[2]
        image_embeds = image_embeds.view(b, t * vis_seq_len, vis_dim)
        
        # 1. 基础投影
        image_embeds = self.projector(image_embeds)

        # 2. 条件执行 MoE
        if self.use_moe:
            batch_size, total_seq, hidden_dim = image_embeds.shape
            image_embeds_flat = image_embeds.view(-1, hidden_dim)
            image_embeds_flat = self.vision_moe(image_embeds_flat)
            image_embeds = image_embeds_flat.view(batch_size, total_seq, hidden_dim)

        # 3. 获取文本 Embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # 4. 条件执行 Fusion 并构造序列
        inputs_list = [image_embeds]
        mask_list = [torch.ones((image_embeds.shape[0], image_embeds.shape[1]), device=self.device)]
        label_list = [torch.full((image_embeds.shape[0], image_embeds.shape[1]), -100, device=self.device)] if labels is not None else []

        if self.use_fusion:
            fusion_output = self.fusion_module(text_embeds=text_embeds, video_embeds=image_embeds)
            inputs_list.append(fusion_output)
            mask_list.append(attention_mask) # 融合层使用文本 mask
            if labels is not None:
                label_list.append(torch.full((b, fusion_output.shape[1]), -100, device=self.device))

        # 拼接文本部分
        inputs_list.append(text_embeds)
        mask_list.append(attention_mask)
        if labels is not None:
            label_list.append(labels)

        # 5. 合并
        inputs_embeds = torch.cat(inputs_list, dim=1)
        attention_mask = torch.cat(mask_list, dim=1)
        if labels is not None:
            labels = torch.cat(label_list, dim=1)
        
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
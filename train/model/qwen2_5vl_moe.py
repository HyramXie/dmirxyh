import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from model.moe_vision_adapter import VisionMoEAdapter

class Qwen2_5_VL_MIntRec(nn.Module):
    def __init__(self, model_path="/root/huggingface/qwen/Qwen2.5-VL-7B-Instruct", device="cuda"):
        super().__init__()
        # 1. 加载基础模型
        self.core_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.core_model.gradient_checkpointing_enable()
        
        # 2. 配置 LoRA (微调 LLM)
        lora_config = LoraConfig(
            r=8,  # 修改为 8
            lora_alpha=16, # 通常 alpha = 2 * r
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
            task_type="CAUSAL_LM",
            lora_dropout=0.05
        )
        self.core_model = get_peft_model(self.core_model, lora_config)
        
        # 3. 初始化 MoE (全参数训练)
        hidden_size = self.core_model.config.hidden_size
        self.vision_moe = VisionMoEAdapter(input_dim=hidden_size).to(device).to(torch.bfloat16)
        
        # 4. Monkey Patch: 劫持 visual.forward
        # 我们需要在 Visual Encoder 输出后，进入 LLM 之前拦截数据
        self.original_visual_forward = self.core_model.visual.forward
        self.core_model.visual.forward = self.new_visual_forward
        
        self.print_trainable_parameters()



# # ... (MoEGate 和 VisionMoEAdapter 类保持不变，不需要动) ...
# # 请保留上面的 MoEGate 和 VisionMoEAdapter 代码

# class Qwen2_5_VL_MIntRec(nn.Module):
#     def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda"):
#         super().__init__()
        
#         # --- 核心修改 1: 配置 4-bit 量化 ---
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,            # 开启 4bit 加载
#             bnb_4bit_quant_type="nf4",    # 使用 nf4 格式（性能最好）
#             bnb_4bit_compute_dtype=torch.bfloat16, # 计算时还原为 bf16
#             bnb_4bit_use_double_quant=True # 二次量化节省显存
#         )

#         # 1. 加载基础模型 (带量化配置)
#         # 注意: device_map="auto" 或者是具体的 device
#         self.core_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             model_path, 
#             quantization_config=bnb_config, # <--- 传入量化配置
#             device_map=device,              # 让 bitsandbytes 自动管理设备
#             trust_remote_code=True 
#         )
        
#         # --- 核心修改 2: 预处理 kbit 模型 ---
#         # 这步非常关键，它会冻结原参数并把 Norm 层转为 fp32 以保证稳定
#         self.core_model = prepare_model_for_kbit_training(self.core_model)

#         # 2. 配置 LoRA
#         lora_config = LoraConfig(
#             r=8,
#             lora_alpha=16,
#             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
#             task_type="CAUSAL_LM",
#             lora_dropout=0.05
#         )
#         self.core_model = get_peft_model(self.core_model, lora_config)
        
#         # 3. 初始化 MoE (这部分保持 BF16，因为它参数量小且需要训练)
#         hidden_size = self.core_model.config.hidden_size
#         self.vision_moe = VisionMoEAdapter(input_dim=hidden_size).to(device).to(torch.bfloat16)
        
#         # 4. Monkey Patch 视觉层
#         self.original_visual_forward = self.core_model.visual.forward
#         self.core_model.visual.forward = self.new_visual_forward
        
#         self.print_trainable_parameters()

    def new_visual_forward(self, *args, **kwargs):
        # 执行原始视觉编码
        # 输出通常是 [Total_Tokens, Dim]
        visual_outputs = self.original_visual_forward(*args, **kwargs)
        
        # 通过 MoE
        moe_outputs = self.vision_moe(visual_outputs)
        
        return moe_outputs

    def forward(self, **kwargs):
        return self.core_model(**kwargs)

    def print_trainable_parameters(self):
        trainable = 0
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        print(f"trainable params: {trainable/1e6:.2f}M || all params: {all_param/1e6:.2f}M || trainable%: {100 * trainable / all_param:.2f}")
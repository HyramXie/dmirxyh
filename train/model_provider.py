import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import get_peft_model, LoraConfig

class MoEGate(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return weights, indices

class VisionMoEAdapter(nn.Module):
    """
    插入到 Visual Encoder 之后的 MoE 适配器
    用于控制模态输入，动态选择最相关的视觉特征专家
    """
    def __init__(self, input_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 专家网络：使用瓶颈结构减少参数量
        hidden_dim = input_dim * 4
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        


    def forward(self, x):
        # x: [Batch*Seq, Dim] (Qwen Flatten后的视觉特征)
        identity = x
        weights, indices = self.router(x) # [N, TopK]
        
        # 简单的循环实现 MoE 计算 (Batch * Seq * TopK)
        # 生产环境可用 scatter/gather 优化
        out = torch.zeros_like(x)
        flat_weights = weights # [N, K]
        flat_indices = indices # [N, K]
        
        for k in range(self.top_k):
            k_weights = flat_weights[:, k].unsqueeze(1) # [N, 1]
            k_indices = flat_indices[:, k]             # [N]
            
            for e_idx in range(self.num_experts):
                mask = (k_indices == e_idx)
                if mask.any():
                    expert_out = self.experts[e_idx](x[mask])
                    out[mask] += expert_out * k_weights[mask]
        
        # 残差连接 + 门控控制
        # 如果视觉信息噪音大，router可以通过权重调整，gate_scale也可以学习调节幅度
        return self.norm(identity + out * self.gate_scale)

class Qwen2_5_VL_MIntRec(nn.Module):
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda"):
        super().__init__()
        # 1. 加载基础模型
        self.core_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
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
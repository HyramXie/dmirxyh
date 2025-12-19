import torch
import torch.nn as nn
from model.moe_gate import MoEGate

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
        # 门控
        self.router = MoEGate(input_dim, num_experts, top_k)
        # 残差连接的 LayerNorm
        self.norm = nn.LayerNorm(input_dim)
        # 门控系数：学习视觉特征是否需要被保留
        self.gate_scale = nn.Parameter(torch.ones(1) * 0.1)


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
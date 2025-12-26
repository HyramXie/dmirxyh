import torch
import torch.nn as nn
import torch
import torch.nn as nn
from model.moe_gate import MoEGate

class MoEProjector(nn.Module):
    """
    MoE Projector: 将 Vision Encoder 的特征
    投影到 LLM hidden space（如 4096）
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        num_experts=4,
        top_k=2,
        use_residual=False   # projector 默认不强制 residual
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_residual = use_residual

        # ⭐ 专家网络（MLP projector）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim)
            )
            for _ in range(num_experts)
        ])

        # ⭐ Router 仍然基于 vision feature
        self.router = MoEGate(input_dim, num_experts, top_k)

        # ⭐ 输出空间的 LayerNorm
        self.norm = nn.LayerNorm(output_dim)

        # ⭐ 控制 MoE 注入强度（非常重要）
        self.gate_scale = nn.Parameter(torch.ones(1) * 0.1)

        # 如果你真的想 residual（一般不推荐）
        if use_residual:
            self.residual_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        x: [B, N, input_dim]  (视觉 token)
        return: [B, N, output_dim]
        """
        B, N, D = x.shape
        x_flat = x.view(B * N, D)      # ⭐ token-level

        # router
        weights, indices = self.router(x_flat)  # [T, K]

        # 输出 buffer（dtype 对齐）
        out_flat = torch.zeros(
            x_flat.size(0),
            self.experts[0][-1].out_features,
            device=x.device,
            dtype=x.dtype
        )

        for k in range(self.top_k):
            k_weights = weights[:, k].unsqueeze(1)   # [T, 1]
            k_indices = indices[:, k]                # [T]

            for e in range(self.num_experts):
                mask = (k_indices == e)
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])
                    out_flat[mask] += expert_out * k_weights[mask]

        out = out_flat.view(B, N, -1)

        # 可选 residual
        if self.use_residual:
            out = out + self.residual_proj(x)

        # scale + norm
        return self.norm(out * self.gate_scale)

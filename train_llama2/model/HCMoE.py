import torch
import torch.nn as nn

class HCExpert(nn.Module):
    def __init__(self, dim_v):
        super().__init__()
        # 专家由两层线性层组成，中间包含 GELU [cite: 144, 147]
        self.w_prime = nn.Linear(dim_v, dim_v)
        self.w = nn.Linear(dim_v, dim_v)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(dim_v)

    def forward(self, E):
        # 对应论文公式 (1) 和 (2) [cite: 144, 147]
        # V = σ(W · GELU(W' · E + b') + b)
        x = self.w_prime(E)
        x = self.gelu(x)
        x = self.w(x)
        return self.layer_norm(x)

class GatingNetwork(nn.Module):
    def __init__(self, dim_v, num_heads=8):
        super().__init__()
        # 核心在于多头自注意力 [cite: 105, 149]
        self.attention = nn.MultiheadAttention(embed_dim=dim_v, num_heads=num_heads, batch_first=True)
        
        # 论文结构图中显示该块由 Linear + ReLU + LN 组成，且重复两次 [cite: 114, 284]
        self.block = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.ReLU(),
            nn.LayerNorm(dim_v),
            nn.Linear(dim_v, dim_v),
            nn.ReLU(),
            nn.LayerNorm(dim_v)
        )
        
        # 最后通过线性层和 Softmax 输出 2 个专家的权重评分 [cite: 96, 150]
        self.gate_projector = nn.Linear(dim_v, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, E):
        # Attention(E) [cite: 150]
        attn_output, _ = self.attention(E, E, E)
        x = self.block(attn_output)
        
        # 计算逻辑回归权重，并在最后一个维度取平均值或对特定 token 评分
        # 论文指出权重是输入依赖的 [cite: 149]
        logits = self.gate_projector(x) # [batch, tokens, 2]
        weights = self.softmax(logits) 
        return weights

class HybridCompressor(nn.Module):
    def __init__(self, dim_v):
        super().__init__()
        self.emotion_expert = HCExpert(dim_v)
        self.general_expert = HCExpert(dim_v)
        self.gating_network = GatingNetwork(dim_v)

    def forward(self, E):
        # 1. 专家计算
        v_emo = self.emotion_expert(E) # [cite: 143]
        v_gen = self.general_expert(E) # [cite: 145]
        
        # 2. 门控评分计算 (G)
        weights = self.gating_network(E)
        g_emotion = weights[..., 0:1] # 获取第一个评分作为 G [cite: 152]
        g_general = weights[..., 1:2] # 对应 (1-G) [cite: 152]
        
        # 3. 最终融合公式: Vout = G · Vemo + (1-G) · Vgen [cite: 152]
        v_out = g_emotion * v_emo + g_general * v_gen
        return v_out
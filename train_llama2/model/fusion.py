import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, max_frames=4):
        """
        hidden_size: LLM 的 hidden size (如 4096)
        num_heads: 注意力头数
        max_frames: 预设的最大帧数，用于初始化时序编码
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 1. 【新增】时序位置编码 (Temporal Position Embedding)
        # 为每一帧学习一个特定的向量，让模型知道视频帧的先后顺序
        self.temporal_embedding = nn.Parameter(torch.zeros(1, max_frames, 1, hidden_size))
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        
        # 2. 第一层：交叉注意力 (Cross-Attention)
        # 对应架构图底层：Query 来自文本，Key/Value 来自视频
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        
        # 3. 第二层：自注意力 (Self-Attention)
        # 对应架构图中层：对融合后的特征进一步提取上下文
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # 4. 第三层：MLP Projector (Feed-Forward)
        # 对应架构图顶层：非线性特征映射
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, text_embeds, video_embeds, num_frames=4, video_mask=None):
        """
        text_embeds: [Batch, Text_Seq, Dim]
        video_embeds: [Batch, Total_Vis_Seq, Dim] (通常是 T * Patches)
        num_frames: 当前输入的实际帧数
        """
        B, Total_Vis_Seq, D = video_embeds.shape
        patches_per_frame = Total_Vis_Seq // num_frames
        
        # --- Step 1: 注入时序感知 ---
        # 将视频展回 [B, T, Patches, D] 以便加上时序偏置
        vis_features = video_embeds.view(B, num_frames, patches_per_frame, D)
        # 加上对应帧的 temporal embedding
        vis_features = vis_features + self.temporal_embedding[:, :num_frames, :, :]
        # 展回 [B, T*Patches, D] 用于计算
        vis_features = vis_features.view(B, -1, D)

        # --- Step 2: Cross Attention (文本采样视频) ---
        # 残差来源: text_embeds
        attn_out, _ = self.cross_attn(
            query=text_embeds,
            key=vis_features,
            value=vis_features
        )
        x = self.norm1(text_embeds + attn_out) # 对应图中第一个 Add 节点

        # --- Step 3: Self Attention (特征内聚) ---
        # 残差来源: x (Cross Attention 的输出)
        self_attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x
        )
        x = self.norm2(x + self_attn_out) # 对应图中第二个 Add 节点

        # --- Step 4: MLP Projector ---
        # 最终特征映射
        mlp_out = self.mlp(x)
        output = self.norm3(x + mlp_out) # 标准 Transformer 结构通常在 MLP 也有残差

        return output
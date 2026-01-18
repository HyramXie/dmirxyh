import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        """
        hidden_size: LLM 的 hidden size (例如 4096)
        num_heads: 注意力头数
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Multihead Attention
        # batch_first=True 表示输入输出都是 [Batch, Seq, Dim]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 2. LayerNorm & Dropout (用于残差连接和稳定训练)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 3. 门控系数 (可选)：学习融合的比例，初始设为 0 可以让训练更稳定
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, text_embeds, video_embeds, text_mask=None, video_mask=None):
        """
        text_embeds: [Batch, Text_Seq, Dim]  -> 作为 Query
        video_embeds: [Batch, Video_Seq, Dim] -> 作为 Key, Value
        """
        # MultiheadAttention 的 forward 参数: query, key, value
        # key_padding_mask: 用于屏蔽 padding 部分 (True 表示屏蔽)
        # 注意: nn.MultiheadAttention 的 mask 逻辑是 True 代表被 mask 掉
        
        # 处理 mask: 如果传入的是 [B, Seq] 的 0/1 mask (1有效)，需要转反
        key_padding_mask = None
        if video_mask is not None:
            # 假设 video_mask 1是有效，0是padding -> 变成 True(屏蔽) / False(保留)
            key_padding_mask = (video_mask == 0)
            
        attn_output, _ = self.cross_attn(
            query=text_embeds,
            key=video_embeds,
            value=video_embeds,
            key_padding_mask=key_padding_mask
        )
        
        # 残差连接 + 门控
        # 我们希望输出的是“基于文本关注到的视频信息”
        # 这里可以直接返回 attn_output，或者加个残差（这就变成了 Text 增强）
        # 既然你要“三者合并”，这里直接返回 pure attention output 比较清晰
        
        return self.dropout(self.norm(attn_output))
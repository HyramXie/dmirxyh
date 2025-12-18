# coding=utf-8
# Copyright 2025.
# VisionMoEAdapter for Qwen2.5-VL within LLaMA-Factory
# ------------------------------------------------------
# 用于在 Qwen2.5-VL 的视觉-语言融合阶段引入 Mixture-of-Experts 模块
# 以增强模态对齐与鲁棒性。

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionMoEAdapter(nn.Module):
    """
    VisionMoEAdapter
    ----------------
    在 Qwen2.5-VL 的多模态融合阶段插入 MoE 结构，
    根据模态强度（文本/图像主导程度）动态选择专家。

    参数:
        base_model: 已加载的 Qwen2.5-VL 模型实例
        hidden_dim: 模型隐藏维度（通常 4096 或 3584）
        num_experts: 专家数量 (默认 3)
        top_k: 每次激活专家数 (默认 2)
        dropout: dropout 比例
    """

    def __init__(self, base_model, hidden_dim=4096, num_experts=3, top_k=2, dropout=0.1):
        super().__init__()
        self.base_model = base_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Router 输入维度 = hidden_dim + 1 (模态提示)
        self.gate = nn.Linear(hidden_dim + 1, num_experts)

        # 定义三个专家：文本主导、视觉主导、冲突检测
        self.experts = nn.ModuleList([
            self._build_text_expert(hidden_dim, dropout),
            self._build_vision_expert(hidden_dim, dropout),
            self._build_conflict_expert(hidden_dim, dropout)
        ])

    # ===== Expert 定义 =====
    def _build_text_expert(self, dim, dropout):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def _build_vision_expert(self, dim, dropout):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def _build_conflict_expert(self, dim, dropout):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    # ===== Forward =====
    def forward(self, **kwargs):
        """
        前向传播：调用原模型获取 hidden states，
        然后根据模态强度为 Router 提供提示，并激活若干专家。
        """
        outputs = self.base_model(**kwargs, output_hidden_states=True)
        h = outputs.hidden_states[-1]     # [B, S, D]

        # ===== ① 模态特征分离 =====
        # Qwen2.5-VL 的输入序列通常前部分为图像 token
        vision_len = int(h.size(1) * 0.3)  # 约30%为视觉 token
        h_vision = h[:, :vision_len, :]
        h_text = h[:, vision_len:, :]

        # ===== ② 计算模态强度 =====
        text_strength = torch.mean(torch.abs(h_text), dim=(1, 2), keepdim=True)
        vision_strength = torch.mean(torch.abs(h_vision), dim=(1, 2), keepdim=True)
        modality_ratio = text_strength / (vision_strength + 1e-8)
        modality_ratio = modality_ratio.expand(-1, h.size(1), -1)  # [B,S,1]

        # ===== ③ Router gating =====
        router_input = torch.cat([h, modality_ratio], dim=-1)      # [B,S,D+1]
        gate_logits = self.gate(router_input)                      # [B,S,E]
        gate_scores = F.softmax(gate_logits, dim=-1)

        # ===== ④ Top-k 稀疏激活 =====
        top_val, top_idx = torch.topk(gate_scores, self.top_k, dim=-1)
        top_val = top_val / (top_val.sum(dim=-1, keepdim=True) + 1e-8)
        moe_out = torch.zeros_like(h)

        for i in range(self.top_k):
            idx = top_idx[..., i]
            val = top_val[..., i].unsqueeze(-1)
            out_i = torch.stack([self.experts[j](h[b]) for b, j in enumerate(idx[:, 0])])
            moe_out += val * out_i

        outputs.last_hidden_state = moe_out
        # === ⑤ 保存专家激活信息，用于可视化 ===
        if not hasattr(self, "router_logs"):
            self.router_logs = []

        # 记录每条样本的平均激活强度
        avg_gate = gate_scores.mean(dim=1).detach().cpu().numpy()  # [B, num_experts]
        for i in range(avg_gate.shape[0]):
            self.router_logs.append({
                "sample_idx": len(self.router_logs),
                "gate_scores": avg_gate[i].tolist()
            })

        return outputs

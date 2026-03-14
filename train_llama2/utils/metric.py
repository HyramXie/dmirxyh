import numpy as np
import torch
from sklearn.metrics import accuracy_score

def preprocess_logits_for_metrics(logits, labels):
    """
    【显存优化】
    在 GPU 上瞬间完成 Argmax，把巨大的 float32 Logits 变成小小的 int64 索引。
    """
    if isinstance(logits, tuple):
        # 兼容 Hugging Face 模型输出 (logits, past_key_values)
        logits = logits[0]
    
    # 直接取最大值索引 (argmax)，保留序列长度
    # 输出形状: [batch_size, seq_len]
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    """
    【最终稳健版】
    包含：格式转换 + 长度自动对齐 + 掩码过滤
    """
    predictions, labels = eval_pred
    
    # 1. 确保数据是 numpy 数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # ====================================================
    # 🚨 核心修复：强制对齐长度 (解决 54 vs 52 报错)
    # ====================================================
    # 获取两个矩阵在 dim=1 (序列长度) 上的最小值
    # 无论谁长谁短，都截取到公共长度
    min_len = min(predictions.shape[1], labels.shape[1])
    
    predictions = predictions[:, :min_len]
    labels = labels[:, :min_len]

    # ====================================================
    # 2. 正常的 Flatten & Mask 逻辑
    # ====================================================
    
    # 创建掩码：找出所有 label 不等于 -100 的位置
    mask = labels != -100
    
    # 利用掩码提取有效数据
    valid_preds = predictions[mask]
    valid_labels = labels[mask]
    
    # 极端情况兜底
    if len(valid_labels) == 0:
        return {"accuracy": 0.0}

    # 计算准确率
    acc = accuracy_score(valid_labels, valid_preds)
    
    return {"accuracy": acc}
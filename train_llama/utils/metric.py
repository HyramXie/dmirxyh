import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted") 
    
    return {
        "accuracy": acc,
        "f1": f1
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    在显存里就把 Logits 变成预测的 Token ID，
    避免把巨大的 Logits 矩阵传回 CPU。
    """
    if isinstance(logits, tuple):
        # 兼容 Hugging Face 模型输出 (logits, past_key_values)
        logits = logits[0]
    
    # 直接取最大值索引 (argmax)
    return logits.argmax(dim=-1)
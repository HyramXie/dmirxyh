from transformers import Qwen2_5_VLForConditionalGeneration
from llamafactory.model.model_utils.qwen2vl_moe_adapter import VisionMoEAdapter
import torch

# 1️⃣ 加载原始模型
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/root/huggingface/qwen/Qwen2.5-VL-7B-Instruct",
    trust_remote_code=True
)

# 2️⃣ 用MoE包一层
model = VisionMoEAdapter(base_model, hidden_dim=4097, num_experts=3, top_k=2)

# ✅ 3️⃣ 保存配置到json（方便训练脚本读取）
save_dir = "./pretrained/qwen2_5vl_7b_moe_adapter"
import os, json
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, "moe_adapter.pt"))
json.dump({
    "base_model": "/root/huggingface/qwen/Qwen2.5-VL-7B-Instruct",
    "num_experts": 3,
    "top_k": 2,
    "hidden_dim": 4096
}, open(os.path.join(save_dir, "moe_config.json"), "w"))

print(f"✅ MoE Adapter 权重已保存至 {save_dir}")

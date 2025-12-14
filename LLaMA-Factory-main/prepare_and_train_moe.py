# CUDA_VISIBLE_DEVICES=2 python prepare_and_train_moe.py
from transformers import Qwen2_5_VLForConditionalGeneration
from llamafactory.model.model_utils.qwen2vl_moe_adapter import VisionMoEAdapter
import torch, json, os, subprocess

# === 1️⃣ 先加载原模型 + 注入 MoE ===
base_model_path = "/root/huggingface/qwen/Qwen2.5-VL-7B-Instruct"
moe_path = "./pretrained/qwen2_5vl_7b_moe_adapter"

config = json.load(open(os.path.join(moe_path, "moe_config.json")))
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_path, trust_remote_code=True)

# 注意：这里使用 4097 作为 hidden_dim 以匹配保存的检查点
model = VisionMoEAdapter(base_model,
                         hidden_dim=4097,  # 修改为与检查点匹配的维度
                         num_experts=config["num_experts"],
                         top_k=config["top_k"])
model.load_state_dict(torch.load(os.path.join(moe_path, "moe_adapter.pt")))

print("✅ MoE 层加载完成，开始训练 ...")

# === 2️⃣ 启动 LLaMA-Factory 训练 ===
subprocess.run([
    "llamafactory-cli", "train", "62/qwen2vl_lora_sft.yaml"
])

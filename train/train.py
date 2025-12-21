import torch
import argparse
from transformers import TrainingArguments, Trainer, Qwen2_5_VLProcessor
from model.qwen2_5vl_moe import Qwen2_5_VL_MIntRec
from data.mintrec_dataset import MIntRecDataset
from data.data_collator import DataCollatorForQwenMIntRec

def train():
    parser = argparse.ArgumentParser()
    # --- 路径相关参数 ---
    parser.add_argument("--model_path", type=str, default="/root/huggingface/qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data_path", type=str, default="/root/user/xyh/LLaMA-Factory-main/data/MIntRec_train.json")
    parser.add_argument("--video_dir", type=str, default="/root/user/xyh/Datasets/MIntRec/raw_data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/mintrec_moe")

    # --- 训练超参数 ---
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    
    # --- 日志与保存 ---
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    
    # --- 硬件与性能 ---
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader num workers")
    args = parser.parse_args()

    # 1. 初始化 Processor
    model_path = args.model_path
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1280*28*28)

    # 2. 初始化数据集
    train_dataset = MIntRecDataset(
        json_path=args.data_path,
        video_dir=args.video_dir,
        processor=processor
    )
    
    # 3. 初始化模型 (Qwen + LoRA + MoE)
    model = Qwen2_5_VL_MIntRec(model_path=model_path, device="cuda")
    model.train()

    # 4. 训练参数
    training_args = TrainingArguments(
        ### Output
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_steps=args.save_steps,
        overwrite_output_dir=True,
        report_to="none",

        ### Train config
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=1.0e-4, 
        lr_scheduler_type="cosine",
        bf16=True,                      # 强烈建议开启 BF16
        warmup_ratio=0.1,
        ddp_timeout=180000000,
        
        ### Hardware
        dataloader_num_workers=args.num_workers,

        ### Other
        remove_unused_columns=False, # 必须False
    )

    # 5. 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForQwenMIntRec(processor)
    )

    print("Starting training with Qwen2.5-VL + MoE + LoRA...")
    trainer.train()

if __name__ == "__main__":
    train()
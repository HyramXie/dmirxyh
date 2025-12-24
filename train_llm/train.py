import os
import torch
import argparse
from transformers import (
    Trainer, 
    TrainingArguments
)
from transformers import set_seed
from model.qwen_siglip import QwenWithSiglip
from data.mintrec_dataset import MIntRecDataset
from data.data_collator import DataCollator  


def train():
    set_seed(42) #设种子

    parser = argparse.ArgumentParser()
    # --- 路径相关参数 ---
    parser.add_argument("--llm_path", type=str, default="/root/huggingface/qwen/Qwen3-8B")
    parser.add_argument("--vision_path", type=str, default="/root/huggingface/google/siglip-so400m-patch14-384")
    parser.add_argument("--data_path", type=str, default="/root/user/xyh/Datasets/MIntRec/MIntRec_train.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/qwen_mintrec_base")
    parser.add_argument("--num_frames", type=int, default=4)

    # --- 训练超参数 ---
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    # --- 日志与保存 ---
    parser.add_argument("--logging_steps", type=int, default=10)
    
    # --- 硬件与性能 ---
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader num workers")
    args = parser.parse_args()

    # 1. 初始化模型
    model = QwenWithSiglip(llm_path=args.llm_path, vision_path=args.vision_path, device="cuda")
    
    # 2. 准备数据
    dataset = MIntRecDataset(
        data_json_path=args.data_path,
        tokenizer=model.tokenizer,
        image_processor=model.vision_encoder.processor,
        num_frames=args.num_frames # 显存不够可以设为 1 或 2，显存大可以设为 8
    )
    
    # 3. 训练参数
    training_args = TrainingArguments(
        ### Output
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_strategy="no",
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
    
    data_collator = DataCollator(model.tokenizer)

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    print("Starting training...")
    trainer.train()
    
    # 5. 保存
    model.llm.save_pretrained(args.output_dir)
    model.tokenizer.save_pretrained(args.output_dir)
    torch.save(model.projector.state_dict(), os.path.join(args.output_dir, "projector.pt"))
    print(f"Model and projector saved to {args.output_dir}")

if __name__ == "__main__":
    train()
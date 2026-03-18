import os
import torch
import argparse
from transformers import (
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from transformers import set_seed
from model.qwen_siglip import QwenWithSiglip
from utils.multimodal_trainer import MultimodalTrainer
from data.mintrec_dataset import MIntRecDataset
from data.data_collator import DataCollator  


def train():
    set_seed(42) #设种子

    parser = argparse.ArgumentParser()
    # --- 路径相关参数 ---
    parser.add_argument("--llm_path", type=str, default="/root/huggingface/qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--vision_path", type=str, default="/root/huggingface/google/siglip-so400m-patch14-384")
    parser.add_argument("--train_data_path", type=str, default="/root/user/xyh/Datasets/MIntRec/MIntRec_train.json")
    parser.add_argument("--eval_data_path", type=str, default="/root/user/xyh/Datasets/MIntRec/MIntRec_eval.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/qwen_mintrec_base")
    parser.add_argument("--num_frames", type=int, default=4)

    # --- 训练超参数 ---
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    # --- 日志与保存 ---
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    
    # --- 硬件与性能 ---
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader num workers")
    args = parser.parse_args()

    # 1. 初始化模型
    model = QwenWithSiglip(llm_path=args.llm_path, vision_path=args.vision_path, device="cuda")
    
    # 2. 准备数据
    train_dataset = MIntRecDataset(
        data_json_path=args.train_data_path,
        tokenizer=model.tokenizer,
        image_processor=model.vision_encoder.processor,
        num_frames=args.num_frames # 显存不够可以设为 1 或 2，显存大可以设为 8
    )
    eval_dataset = MIntRecDataset(
        data_json_path=args.eval_data_path,
        tokenizer=model.tokenizer,
        image_processor=model.vision_encoder.processor,
        num_frames=args.num_frames # 显存不够可以设为 1 或 2，显存大可以设为 8
    )
    
    # 3. 训练参数
    training_args = TrainingArguments(
        ### Output
        output_dir=args.output_dir,

        ###save
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,

        ###early stop
        eval_strategy="steps",
        eval_steps=args.save_steps,
        eval_accumulation_steps=1,
        per_device_eval_batch_size=1,
        metric_for_best_model="loss", # 根据哪个指标判断"最好" (建议用 accuracy 或 loss)
        greater_is_better=True,

        overwrite_output_dir=True,
        report_to="swanlab",

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
    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        tokenizer=model.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    #     callbacks=[
    #     EarlyStoppingCallback(
    #         early_stopping_patience=3,  # 容忍度：如果连续 3 次评估指标没有提升，就停止
    #         early_stopping_threshold=0.0 # 阈值：提升多少才算提升 (通常 0.0 即可)
    #     )
    # ]
    )
    
    print("Starting training...")
    trainer.train()
    
    # 5. 打印最佳模型的路径
    print("="*30)
    print(f"🏆 Best model checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"📊 Best metric value: {trainer.state.best_metric}")
    print("="*30)

if __name__ == "__main__":
    train()
import argparse
from transformers import TrainingArguments, Trainer, Qwen2_5_VLProcessor
from model.qwen2_5vl_moe import Qwen2_5_VL_MIntRec
from data.mintrec_dataset import MIntRecDataset
from data.data_collator import DataCollatorForQwenMIntRec

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/root/user/xyh/LLaMA-Factory-main/data/MIntRec_train.json")
    parser.add_argument("--video_dir", type=str, default="/root/user/xyh/Datasets/MIntRec/raw_data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/mintrec_moe")
    args = parser.parse_args()

    # 1. 初始化 Processor
    model_path = "/root/huggingface/qwen/Qwen2.5-VL-7B-Instruct"
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

        ###output
        output_dir=args.output_dir,
        logging_steps=10,
        save_strategy="epoch",
        save_steps=500,
        overwrite_output_dir=True,
        report_to="none",

        ###train
        per_device_train_batch_size=1,  # 视频显存占用大，建议小 Batch
        gradient_accumulation_steps=8,  # 等效 Batch Size = 16
        num_train_epochs=5,
        learning_rate=1.0e-4,             # LoRA 学习率
        lr_scheduler_type="cosine",
        bf16=True,                      # 强烈建议开启 BF16
        warmup_ratio=0.1,
        ddp_timeout=180000000,
        
        ### dataset
        dataloader_num_workers=0, # 单进程加载

        # ###other
        remove_unused_columns=False,    # 必须False，防止Processor生成的video特征被删
        # dataloader_pin_memory=False,     # 锁住视频内存，速度快，视频数据加载有时会冲突，可视情况关闭
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
    
    # 保存 MoE 权重 (LoRA会自动保存)
    torch.save(model.vision_moe.state_dict(), f"{args.output_dir}/vision_moe_final.pt")

if __name__ == "__main__":
    train()
import os
import torch
from transformers import (
    Trainer, 
    TrainingArguments
)

def train():
    # 配置
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    vision_name = "google/siglip-so400m-patch14-384"
    data_path = "mintrec_train.json" # 你的数据文件
    video_dir = "mintrec_videos/"      # 你的视频目录
    output_dir = "output_qwen_siglip"
    
    # 1. 初始化模型
    model = QwenWithSiglip(
        llm_path=model_name, 
        vision_path=vision_name,
        lora_rank=16
    )
    
    # 2. 准备数据
    dataset = MIntRecDataset(
        data_json_path=data_path,
        video_dir=video_dir,
        tokenizer=model.tokenizer,
        image_processor=model.vision_processor,
        num_frames=4 # 显存不够可以设为 1 或 2，显存大可以设为 8
    )
    
    # 3. 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2, # 取决于显存，Qwen-7B + Siglip 很大
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
        bf16=True, # 强烈建议使用 BF16
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False, # 必须设为 False，否则 pixel_values 会被过滤掉
        report_to="none"
    )
    
    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=custom_collate_fn
    )
    
    print("Starting training...")
    trainer.train()
    
    # 5. 保存
    model.llm.save_pretrained(output_dir)
    torch.save(model.projector.state_dict(), os.path.join(output_dir, "projector.pt"))
    print(f"Model and projector saved to {output_dir}")

if __name__ == "__main__":
    train()
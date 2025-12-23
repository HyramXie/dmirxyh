# --- 1. 环境变量设置 ---
export CUDA_VISIBLE_DEVICES=2  # 指定使用哪张显卡，多卡用 "0,1"

# --- 2. 路径配置 ---
model_path="/root/huggingface/qwen/Qwen2.5-VL-7B-Instruct"
data_path="/root/user/xyh/LLaMA-Factory-main/data/MIntRec2_train.json"
output_dir="/root/user/xyh/train/checkpoints/mintrec2_moe"  # 每次实验可以改个名字

# --- 3. 训练超参数配置 ---
batch_size=1          # 单卡 Batch Size (显存占用大建议为1)
grad_accum_steps=8          # 梯度累积步数 (1 * 8 = 8 等效 Batch Size)
epochs=3              # 训练轮数

# --- 4. 日志与保存 ---
logging_steps=10
save_steps=1000

num_workers=0             # Dataloader 进程数 (显存吃紧或报错时设为0)

# --- 4. 执行命令 ---
echo "Starting training..."
echo "Model: $model_path"
echo "Output: $output_dir"

python ../train.py \
    --model_path "$model_path" \
    --data_path "$data_path" \
    --output_dir "$output_dir" \
    --batch_size $batch_size \
    --grad_accum_steps $grad_accum_steps \
    --epochs $epochs \
    --num_workers $num_workers \
    --logging_steps $logging_steps \

echo "Training finished. Results saved to $output_dir"
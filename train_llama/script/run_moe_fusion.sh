# --- 1. 环境变量设置 ---
export CUDA_VISIBLE_DEVICES=0  # 指定使用哪张显卡，多卡用 "0,1"

# --- 2. 路径配置 ---
llm_path="/root/huggingface/llama/Meta-Llama-3-8B-Instruct"
vision_path="/root/huggingface/google/siglip-so400m-patch14-384"
train_data_path="/root/user/xyh/Datasets/MIntRec/MIntRec_train.json"
eval_data_path="/root/user/xyh/Datasets/MIntRec/MIntRec_dev.json"
output_dir="/root/user/xyh/train_llama/checkpoints/llama_mintrec_moe_fusion"  # 每次实验可以改个名字
num_frames=4

# --- 3. 训练超参数配置 ---
batch_size=1          # 单卡 Batch Size (显存占用大建议为1)
grad_accum_steps=8          # 梯度累积步数 (1 * 8 = 8 等效 Batch Size)
epochs=3              # 训练轮数

# --- 4. 日志与保存 ---
logging_steps=10
save_steps=600

num_workers=0             # Dataloader 进程数 (显存吃紧或报错时设为0)

# --- 4. 执行命令 ---
echo "Starting training..."
echo "Model: $llm_path"
echo "Output: $output_dir"

python ../train_moe_fusion.py \
    --llm_path "$llm_path" \
    --vision_path "$vision_path" \
    --train_data_path "$train_data_path" \
    --eval_data_path "$eval_data_path" \
    --output_dir "$output_dir" \
    --num_frames $num_frames \
    --batch_size $batch_size \
    --grad_accum_steps $grad_accum_steps \
    --epochs $epochs \
    --logging_steps $logging_steps \
    --save_steps $save_steps \
    --num_workers $num_workers \

echo "Training finished. Results saved to $output_dir"
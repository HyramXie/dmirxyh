# --- 1. 环境变量设置 ---
export CUDA_VISIBLE_DEVICES=2  # 指定使用哪张显卡，多卡用 "0,1"

# --- 2. 路径配置 ---
llm_path="/root/huggingface/llama/Meta-Llama-3-8B-Instruct"
vision_path="/root/huggingface/google/siglip-so400m-patch14-384"
train_data_path="/root/user/xyh/Datasets/MIntRec/MIntRec_train.json"
eval_data_path="/root/user/xyh/Datasets/MIntRec/MIntRec_dev.json"
output_dir="/root/user/xyh/train_llama/checkpoints/llama_mintrec_fusion_moe_eos"  # 每次实验可以改个名字

# --- 3. 模块选择 ---
use_moe=True
use_fusion=True

# --- 4. 日志与保存 ---
logging_steps=10
save_steps=600

# --- 5. 训练超参数配置 ---
batch_size=1          # 单卡 Batch Size (显存占用大建议为1)
grad_accum_steps=8          # 梯度累积步数 (1 * 8 = 8 等效 Batch Size)
epochs=3              # 训练轮数
num_frames=4
num_workers=0             # Dataloader 进程数 (显存吃紧或报错时设为0)

# --- 6. 执行命令 ---
echo "Starting training..."
echo "Model: $llm_path"
echo "Use MoE: $use_moe | Use Fusion: $use_fusion"
echo "Output: $output_dir"

# 动态构建参数
MOE_FLAG=""
if [ "$use_moe" = true ]; then
    MOE_FLAG="--use_moe"
fi

FUSION_FLAG=""
if [ "$use_fusion" = true ]; then
    FUSION_FLAG="--use_fusion"
fi

python ../train.py \
    --llm_path "$llm_path" \
    --vision_path "$vision_path" \
    $MOE_FLAG \
    $FUSION_FLAG \
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
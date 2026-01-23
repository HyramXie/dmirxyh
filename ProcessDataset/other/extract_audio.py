import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ================= 核心配置区域 (已根据你的路径填好) =================
DATASET_CONFIGS = [
    {
        "name": "MIntRec",
        "input_root": "/root/user/xyh/Datasets/MIntRec/video",
        # 自动将 audio 文件夹放在与 video 同级的位置
        "output_root": "/root/user/xyh/Datasets/MIntRec/audio"
    },
    {
        "name": "MELD",
        "input_root": "/root/user/xyh/Datasets/MELD/video",
        "output_root": "/root/user/xyh/Datasets/MELD/audio"
    }
]

# 音频参数标准 (多模态通用标准)
TARGET_SR = 16000  # 16k 采样率
MAX_WORKERS = min(8, os.cpu_count() // 2)

# ===============================================================

def extract_one_video(args):
    """ 处理单个视频的任务函数 """
    video_path, output_path = args

    try:
        if os.path.exists(output_path):
            return True # 已存在则跳过

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # FFmpeg 命令: 16k, 16bit, 单声道
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',                 # 去除视频
            '-acodec', 'pcm_s16le',# 16-bit PCM wav
            '-ar', str(TARGET_SR), # 采样率
            '-ac', '1',            # 单声道
            '-loglevel', 'error',  # 静默模式
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        return True

    except Exception as e:
        print(f"\n[Error] {video_path}: {e}")
        return False

def process_dataset(config):
    """ 处理单个数据集 """
    name = config['name']
    input_root = config['input_root']
    output_root = config['output_root']
    
    if not os.path.exists(input_root):
        print(f"⚠️  警告: 找不到路径 {input_root}，跳过 {name} 数据集")
        return

    print(f"\n🚀 正在处理数据集: {name}")
    print(f"   输入: {input_root}")
    print(f"   输出: {output_root}")

    # 1. 扫描文件
    tasks = []
    video_exts = ('.mp4', '.avi', '.mkv', '.mov')
    
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(video_exts):
                video_abs_path = os.path.join(root, file)
                # 计算相对路径 (例如: S04/E01/137.mp4)
                rel_path = os.path.relpath(video_abs_path, input_root)
                # 生成输出路径 (例如: .../audio/S04/E01/137.wav)
                output_abs_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + '.wav')
                tasks.append((video_abs_path, output_abs_path))

    total = len(tasks)
    if total == 0:
        print(f"   未找到任何视频文件。")
        return

    # 2. 并行执行
    print(f"   找到 {total} 个视频，开始提取...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(extract_one_video, tasks), total=total, unit="file"))

    print(f"✅ {name} 处理完成! 成功: {sum(results)} / 总数: {total}")

def main():
    # 检查 FFmpeg 是否安装
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("❌ 错误: 未安装 FFmpeg。请先安装: sudo apt install ffmpeg")
        return

    for config in DATASET_CONFIGS:
        process_dataset(config)

if __name__ == '__main__':
    main()
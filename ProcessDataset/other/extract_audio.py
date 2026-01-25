import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

TARGET_SR = 16000
MAX_WORKERS = min(8, os.cpu_count() or 4)

DATASET_CONFIGS = [
    {
        "name": "MIntRec",
        "input_root": "/root/user/xyh/Datasets/MIntRec/video",
        # 自动将 audio 文件夹放在与 video 同级的位置
        "output_root": "/root/user/xyh/Datasets/MIntRec/audio"
    }
]

def extract_one_video(video_path, output_path):
    try:
        if os.path.exists(output_path):
            return True

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(TARGET_SR),
            "-ac", "1",
            output_path
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30   # ⭐ 防止坏视频拖死
        )
        return True

    except subprocess.TimeoutExpired:
        print(f"\n⏱️ ffmpeg 超时，跳过: {video_path}")
        return False
    except Exception as e:
        print(f"\n❌ 错误 {video_path}: {e}")
        return False


def process_dataset(config):
    name = config['name']
    input_root = config['input_root']
    output_root = config['output_root']

    print(f"\n🚀 正在处理数据集: {name}")
    print(f"   输入: {input_root}")
    print(f"   输出: {output_root}")

    tasks = []
    video_exts = ('.mp4', '.avi', '.mkv', '.mov')

    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(video_exts):
                video_abs = os.path.join(root, file)
                rel = os.path.relpath(video_abs, input_root)
                out = os.path.join(output_root, os.path.splitext(rel)[0] + ".wav")
                tasks.append((video_abs, out))

    total = len(tasks)
    print(f"   找到 {total} 个视频，开始提取...")

    success = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(extract_one_video, v, o)
            for v, o in tasks
        ]

        for f in tqdm(as_completed(futures), total=total, unit="file"):
            if f.result():
                success += 1

    print(f"✅ {name} 完成: {success}/{total}")


def main():
    for config in DATASET_CONFIGS:
        process_dataset(config)

if __name__ == "__main__":
    main()

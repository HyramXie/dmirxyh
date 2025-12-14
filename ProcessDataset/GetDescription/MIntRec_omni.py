import os
import pandas as pd
import time
import base64
import cv2
import numpy as np
from openai import OpenAI

# 1. 配置
client = OpenAI(
    api_key="sk-599bb017e7f74dc9ba828773541362d4",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

input_csv = "/root/user/xyh/Datasets/MIntRec/test.tsv"
output_csv = "test_desc.csv"

# 设定阈值：小于 1.0 秒的视频将被视为“图片序列”处理
MIN_VIDEO_DURATION = 1.0 
# 提取多少帧？建议 4-6 帧足够描述短动作
NUM_FRAMES = 4 

PROMPT = """
Focus on the main speaker(s) in the video.
Describe their body language, gestures, and facial expressions in detail.
Capture any subtle changes in emotion (e.g., from happy to confused, or angry to neutral).
If there are multiple people, describe the dynamic and social interaction between them.
Strictly limit your response to under 50 words.
"""

# ---------------------------
# 2. 图像处理工具函数
# ---------------------------

def encode_image_base64(image_array):
    """将 OpenCV 图像转换为 Base64 字符串"""
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode('utf-8')

def extract_frames_as_base64(video_path, num_frames=4):
    """从视频中均匀提取 N 帧，返回 Base64 列表"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 均匀采样索引，例如 [0, 10, 20, 30]
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames_base64 = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames_base64.append(encode_image_base64(frame))
            
    cap.release()
    return frames_base64

def encode_video_base64(video_path):
    """读取完整视频文件为 Base64"""
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

# 3. 核心逻辑：生成消息体
def generate_description(video_path):
    # 1. 检查时长
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("Video file not found or corrupted")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frames / fps if fps > 0 else 0
    cap.release()

    content_payload = []
    mode_log = ""

    # 2. 分支处理
    if duration < MIN_VIDEO_DURATION:
        # === 模式 A: 视频太短 -> 转为多图序列 ===
        mode_log = f"[Seq-Images {duration:.2f}s]"
        print(f"   ⚠️ Video too short ({duration:.2f}s), extracting {NUM_FRAMES} frames...")
        
        # 提取关键帧
        frames_b64_list = extract_frames_as_base64(video_path, num_frames=NUM_FRAMES)
        
        # 构造 OpenAI 格式的多图输入
        # 这种格式下，模型会理解为这是一组连续的画面
        for b64_str in frames_b64_list:
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
            })
            
    else:
        # === 模式 B: 正常视频 -> 传视频流 ===
        mode_log = "[Video]"
        video_b64 = encode_video_base64(video_path)
        content_payload.append({
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
        })

    # 3. 最后追加 Prompt 文本
    content_payload.append({"type": "text", "text": PROMPT})

    # 4. 发送请求
    completion = client.chat.completions.create(
        model="qwen3-omni-flash",
        messages=[{"role": "user", "content": content_payload}],
        modalities=["text"], 
        stream=False 
    )
    
    return mode_log, completion.choices[0].message.content

# ---------------------------
# 4. 主程序 (断点续传 + 实时保存)
# ---------------------------

# 读取数据
df = pd.read_csv(input_csv, sep='\t')

# --- 断点检测 ---
start_index = 0
if os.path.exists(output_csv):
    try:
        df_done = pd.read_csv(output_csv, usecols=[0])
        start_index = len(df_done)
        print(f"⚡️ 继续上次进度，从第 {start_index} 条开始...")
    except:
        pass
else:
    # 写表头
    pd.DataFrame(columns=list(df.columns) + ["video_description"]).to_csv(output_csv, index=False, encoding="utf-8")

# --- 循环 ---
for idx, row in df.iloc[start_index:].iterrows():
    season = row["season"]
    episode = row["episode"]
    clip = row["clip"]
    
    video_path = os.path.join("/root/user/xyh/Datasets/MIntRec/raw_data", season, episode, str(clip) + ".mp4")
    print(f"[{idx}/{len(df)}] Processing: {video_path}")
    
    save_row = row.copy()
    
    try:
        if not os.path.exists(video_path):
            save_row["video_description"] = "[Error: File Not Found]"
        else:
            # 调用处理函数
            mode, desc = generate_description(video_path)
            print(f"   -> {mode} Result: {desc[:50]}...")
            save_row["video_description"] = desc

    except Exception as e:
        print(f"   ❌ Error: {e}")
        save_row["video_description"] = f"[Error: {str(e)}]"

    # 实时写入
    pd.DataFrame([save_row]).to_csv(output_csv, mode='a', header=False, index=False, encoding="utf-8")
    
    # 这里的 sleep 根据是否转码图片可以适当调整，Base64 传输较慢建议保留
    time.sleep(1.5)

print("✅ 所有处理完成。")


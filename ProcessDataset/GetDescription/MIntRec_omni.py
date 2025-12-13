import os
import pandas as pd
import time
import base64
from openai import OpenAI


# 1. 配置 OpenAI Client (阿里云兼容版)
client = OpenAI(
    api_key="sk-f3d9737c3c214c3a96d2abf087546a6c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 载入数据集
input_csv = "/root/user/xyh/Datasets/MIntRec/train.tsv"
output_csv = "mintrec_with_desc_omni.csv"

df = pd.read_csv(input_csv, sep='\t')

PROMPT = """
Focus on the main speaker(s) in the video.
Describe their body language, gestures, and facial expressions in detail.
Capture any subtle changes in emotion (e.g., from happy to confused, or angry to neutral).
If there are multiple people, describe the dynamic and social interaction between them.
Strictly limit your response to under 50 words.
"""

# 3. 单条视频处理函数 (OpenAI SDK版)
def describe_video(video_path):

    #  Base64 编码格式
    def encode_video(video_path):
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode("utf-8")
    base64_video = encode_video(video_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:;base64,{base64_video}"} 
                },
                {
                    "type": "text", 
                    "text": PROMPT
                }
            ]
        }
    ]

    # 注意：这里我们不需要 stream=True，因为我们要一次性拿到完整文本存表格
    # 推荐使用 qwen-vl-max 或 qwen-vl-plus 进行视频理解
    completion = client.chat.completions.create(
        model="qwen3-omni-flash",
        messages=messages,
        modalities=["text"], 
        stream=False 
    )

    # 解析返回结果
    return completion.choices[0].message.content

# 4. 批量处理 (带中断保存)
video_descriptions = [] 
error_occurred = False 

for idx, row in df.iterrows():
    season = row["season"]
    episode = row["episode"]
    clip = row["clip"]

    video_path = os.path.join("/root/user/xyh/Datasets/MIntRec/raw_data", season, episode, str(clip) + ".mp4")

    print(f"[{idx+1}/{len(df)}] Processing: {video_path}")

    try:
        # 调用函数
        desc = describe_video(video_path)
        
        # 打印预览
        print(f"   -> Generated: {desc[:50]}...") 
        video_descriptions.append(desc)
        
        
        # 休眠防限流
        time.sleep(1.5)

    except Exception as e:
        print(f"\n❌ Critical Error at index {idx}: {e}")
        print("⚠️  Stopping loop and saving current progress...")
        
        # --- 切片保存逻辑 ---
        count = len(video_descriptions)
        df_partial = df.iloc[:count].copy()
        df_partial["video_description"] = video_descriptions
        df_partial.to_csv(output_csv, index=False, encoding="utf-8")
        
        print(f"💾 已紧急保存前 {count} 条数据到: {output_csv}")
        error_occurred = True
        break 

# 5. 正常结束保存
if not error_occurred:
    df["video_description"] = video_descriptions
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print("-" * 30)
    print(f"✅ 全部完成！完整结果已保存至: {output_csv}")



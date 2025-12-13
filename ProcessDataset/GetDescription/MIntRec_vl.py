import os
import pandas as pd
import time
from dashscope import MultiModalConversation
import dashscope

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


# 1. 载入数据集
input_csv = "dataset.csv"
output_csv = "dataset_with_desc_partial.csv" # 建议文件名加上 partial 区分

df = pd.read_csv(input_csv)

PROMPT = "Describe the video content in detail."


# 2. 单条视频处理函数
def describe_video(video_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"video": video_path, "fps": 1},
                {"text": PROMPT}
            ]
        }
    ]
    # 建议使用 qwen-vl-max
    response = MultiModalConversation.call(model="qwen-vl-max", messages=messages)

    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    else:
        # 这里抛出异常，会被下面的 try...except 捕获
        raise Exception(f"API Error Code: {response.code}, Message: {response.message}")


# 3. 批量处理 (带中断保存功能)
video_descriptions = [] # 用于存储生成的描述
error_occurred = False  # 标记是否发生错误

for idx, row in df.iterrows():
    season = row["season"]
    episode = row["episode"]
    clip = row["clip"]

    local_video = f"/your/video/root/{season}/{episode}/{clip}.mp4"
    video_path = f"file://{local_video}"

    print(f"[{idx+1}/{len(df)}] Processing: {video_path}")

    try:
        # 调用大模型
        desc = describe_video(video_path)
        print(f"   -> Generated: {desc[:50]}...") 
        video_descriptions.append(desc)
        
        # 成功后休眠防止限流
        time.sleep(1.5)

    except Exception as e:
        print(f"\n❌ Critical Error at index {idx}: {e}")
        print("⚠️  Stopping loop and saving current progress...")
        
        # 1. 获取目前已经生成的数量
        count = len(video_descriptions)
        
        # 2. 截取原始 dataframe 的前 count 行
        df_partial = df.iloc[:count].copy()
        
        # 3. 写入新列
        df_partial["video_description"] = video_descriptions
        
        # 4. 保存到文件
        df_partial.to_csv(output_csv, index=False, encoding="utf-8")
        
        print(f"💾 已紧急保存前 {count} 条数据到: {output_csv}")
        
        error_occurred = True
        break 


# 4. 如果没有报错，正常保存完整文件
if not error_occurred:
    df["video_description"] = video_descriptions
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print("-" * 30)
    print(f"✅ 全部完成！完整结果已保存至: {output_csv}")
else:
    print("-" * 30)
    print("❌ 程序因错误提前终止，请检查日志并查看已保存的部分文件。")
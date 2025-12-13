import pandas as pd
import json
import os

#修改部分
csv_file = "/root/user/xyh/Datasets/MIntRec/train_desc.csv"
json_file = "/root/user/xyh/ProcessDataset/GetDatasetsJson/MIntRec/MIntRec_train_desc.json"

df = pd.read_csv(csv_file)
conversations = []

for i in range(len(df)):
    data = df.loc[i]

    #修改部分
    video_path = os.path.join("/root/user/xyh/Datasets/MIntRec/raw_data", data["season"], data["episode"], str(data["clip"]) + ".mp4")
    
    text = data["text"]
    lable = data["label"]
    desc = data["video_description"]

    conversations.append({
        "messages": [
            {
                "role": "user",
                "content": "<video>\ntext:" + text + "\nVideo Description:" + desc
            },
            {
                "role": "assistant",
                "content": lable
            }
        ],
        "videos": [
            video_path
        ]
    })
    if (i + 1) % 50 ==0:
        print(f'processing {i+1}/{len(df)} images')

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)
print("finish")
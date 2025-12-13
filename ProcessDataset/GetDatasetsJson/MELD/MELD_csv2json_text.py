import pandas as pd
import json
import os

csv_file = "/root/user/xyh/Datasets/MELD/train.tsv"
json_file = "/root/user/xyh/GetDatasetsJson/MELD/MELD_train_text.json"

#修改部分
df = pd.read_csv(csv_file, sep='\t')
conversations = []

for i in range(len(df)):
    data = df.loc[i]
    
    text = data["Utterance"]
    lable = data["label"]
    conversations.append({
        "messages": [
            {
                "role": "user",
                "content": text
            },
            {
                "role": "assistant",
                "content": lable
            }
        ]
    })
    if (i + 1) % 50 ==0:
        print(f'processing {i+1}/{len(df)} images')

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)
print("finish")
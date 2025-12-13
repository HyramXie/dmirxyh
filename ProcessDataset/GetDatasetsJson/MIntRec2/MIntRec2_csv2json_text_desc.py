import pandas as pd
import json
import os

#修改部分
csv_file = "/root/user/xyh/Datasets/MIntRec2/test.tsv"
csv_desc_file= "/root/user/xyh/Datasets/MIntRec2/test_desc.tsv"
json_file = "/root/user/xyh/ProcessDataset/GetDatasetsJson/MIntRec2/MIntRec2_test_text_efe.json"

df = pd.read_csv(csv_file, sep='\t')
df_desc = pd.read_csv(csv_desc_file, sep='\t')
conversations = []

for i in range(len(df)):
    data = df.loc[i]
    data_desc = df_desc.loc[i]

    text = data["text"]
    lable = data["label"].strip()
    conversations.append({
        "messages": [
            {
                "role": "user",
                "content": "text:" + text + "\nEmotions and Facial Expressions:" + data_desc["Emotions and Facial Expressions"]
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
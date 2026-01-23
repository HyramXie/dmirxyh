import pandas as pd

# 1. 读取 TSV 文件
# 使用 sep='\t' 指定制表符作为分隔符
df = pd.read_csv('/root/user/xyh/Datasets/IEMOCAP/test.tsv', sep='\t')

# 2. 定义要移除的特定句子
# 注意：这里直接复制了您提供的文本，并关闭了 regex 以便按字面意思匹配
text_to_remove = "The candidate labels for dialogue act are: [greeting, question, answer, statement-opinion, statement-non-opinion, apology, command, agreement, disagreement, acknowledge, backchannel, others]. Respond in the format: 'dialogue act: [label]'. Only one label should be provided."

# 3. 执行移除操作
# replace 将指定文本替换为空字符串，strip 去除可能留下的多余换行符
df['text'] = df['text'].str.replace(text_to_remove, "", regex=False).str.strip()

# 4. (可选) 移除前导的提示语
# 您的文件中似乎还有一句 "Based on the text... what is the dialogue act of this speaker?"
# 如果您也想把这句连同前面的问号一起去掉，可以取消下面这行的注释：
# prompt_prefix = "Based on the text, video, and audio content, what is the dialogue act of this speaker?"
# df['text'] = df['text'].str.replace(prompt_prefix, "", regex=False).str.strip()

# 5. 保存处理后的文件
df.to_csv('/root/user/xyh/Datasets/IEMOCAP/test_cleaned.tsv', sep='\t', index=False)

print("处理完成！已保存为 test_cleaned.tsv")
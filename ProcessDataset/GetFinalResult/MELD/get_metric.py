import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import defaultdict, Counter
import sys

logger_file = "/root/user/xyh/ProcessDataset/GetFinalResult/MELD/result_report.txt"
json_file = "/root/user/xyh/LLaMA-Factory-main/eval/MELD/generated_predictions.json"
wrong_case_file = "/root/user/xyh/ProcessDataset/GetFinalResult/MELD/case_study.json"

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)   # 同时输出到终端
        self.log.write(message)        # 同时写入文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(logger_file)

# 读取 jsonl 文件


y_true = []
y_pred = []
wrong_cases = []

with open(json_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        # 去除标签中的换行符与多余空格
        true_label = data["label"].strip()
        pred_label = data["predict"].strip()
        y_true.append(true_label)
        y_pred.append(pred_label)

        #获取错误样本
        if true_label != pred_label:
            wrong_cases.append({
                "prompt": data["prompt"],
                "predict": data["predict"].strip(),
                "label": data["label"].strip()
            })

# 计算各项指标
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')
prec = precision_score(y_true, y_pred, average='macro')
rec = recall_score(y_true, y_pred, average='macro')

weighted_f1 = f1_score(y_true, y_pred, average='weighted')
weighted_prec = precision_score(y_true, y_pred, average='weighted')
weighted_rec = recall_score(y_true, y_pred, average='weighted')

# 打印结果
print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {f1:.4f}")
print(f"Macro Precision: {prec:.4f}")
print(f"Macro Recall: {rec:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Weighted Precision: {weighted_prec:.4f}")
print(f"Weighted Recall: {weighted_rec:.4f}")

# with open(wrong_case_file, "w", encoding="utf-8") as f:
#     json.dump(wrong_cases, f, ensure_ascii=False, indent=2)
# print("finish")



# -----------------------
# 统计：按真实标签分组分析错误
# -----------------------

total_per_class = Counter(y_true)       # 每个真实类别的总样本数
wrong_per_class = Counter()              # 每个真实类别的错误数
confusions = defaultdict(Counter)       # 每类“错成了什么”

for case in wrong_cases:
    true_label = case["label"]
    pred_label = case["predict"]
    wrong_per_class[true_label] += 1
    confusions[true_label][pred_label] += 1

# 按错误率排序
error_report = []

for label in total_per_class:
    total = total_per_class[label]
    wrong = wrong_per_class[label]
    error_rate = wrong / total
    error_report.append((label, total, wrong, error_rate))

error_report.sort(key=lambda x: x[3], reverse=True)

# -----------------------
# 打印报告
# -----------------------

print("\n📊 各类别错误率排行榜（按真实标签统计）：")
print("-" * 60)
print(f"{'Class':15s} {'Total':>8s} {'Wrong':>8s} {'Error Rate':>12s}")
print("-" * 60)

for label, total, wrong, rate in error_report:
    print(f"{label:15s} {total:8d} {wrong:8d} {rate:11.2%}")

# -----------------------
# 打印：每个类别最常见的错误预测
# -----------------------

print("\n🔍 每个类别最常被错认成的标签（Top Errors）：")
print("-" * 60)

for label, preds in confusions.items():
    most_common = preds.most_common(3)
    if most_common:
        print(f"\n真实类别: {label}, 总共样本数量：{total_per_class[label]}")
        for wrong_label, cnt in most_common:
            print(f"  → 被预测为 {wrong_label}: {cnt} 次")

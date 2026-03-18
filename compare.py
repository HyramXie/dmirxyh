import json

# 配置文件路径
DATASET_FILE = '/root/user/xyh/Datasets/MIntRec/MIntRec_test.json'
MODEL1_FILE = '/root/user/xyh/train_llama/eval/mintrec_predictions_fusion.json'
MODEL2_FILE = '/root/user/xyh/train_llama/eval/mintrec_predictions_fusion_eos.json'
OUTPUT_FILE = 'model1_better_fusion.json' # 分析结果保存的文件

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_model_differences():
    # 1. 加载数据
    try:
        dataset = load_json(DATASET_FILE)
        model1_preds = load_json(MODEL1_FILE)
        model2_preds = load_json(MODEL2_FILE)
    except FileNotFoundError as e:
        print(f"找不到文件，请检查路径: {e}")
        return

    # 确保三个文件的数据量一致
    if not (len(dataset) == len(model1_preds) == len(model2_preds)):
        print("错误: 三个文件的数据长度不一致，请检查数据是否对齐！")
        return

    model1_better_cases = []
    
    # 2. 遍历数据并进行对比
    for i in range(len(dataset)):
        true_label = dataset[i]['label']
        m1_predict = model1_preds[i]['predict']
        m2_predict = model2_preds[i]['predict']
        
        # 核心逻辑：模型一预测正确，且模型二预测错误
        if m1_predict == true_label and m2_predict != true_label:
            case_info = {
                "index": i,
                "text": dataset[i].get('text', ''),
                "video_path": dataset[i].get('video_path', ''),
                "audio_path": dataset[i].get('audio_path', ''),
                "true_label": true_label,
                "model1_predict": m1_predict,
                "model2_predict": m2_predict
            }
            model1_better_cases.append(case_info)

    # 3. 输出统计结果
    print(f"总数据量: {len(dataset)} 条")
    print(f"模型一比模型二多预测正确的数据量: {len(model1_better_cases)} 条")
    
    # 打印前几个例子看看
    print("\n--- 部分示例展示 ---")
    for case in model1_better_cases[:3]: # 只展示前3条
        print(f"索引 {case['index']}:")
        print(f"  文本: {case['text']}")
        print(f"  真实标签: {case['true_label']}")
        print(f"  模型一(对): {case['model1_predict']}")
        print(f"  模型二(错): {case['model2_predict']}\n")

    # 4. 将提取出的差异数据保存到新文件中，方便你后续深入分析
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(model1_better_cases, f, indent=4, ensure_ascii=False)
    
    print(f"详细的差异数据已保存至: {OUTPUT_FILE}")

if __name__ == '__main__':
    analyze_model_differences()
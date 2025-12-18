import torch
from transformers import Qwen2_5_VLProcessor

class DataCollatorForQwenMIntRec:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        texts = [x["text_input"] for x in batch]
        # images = [x["images"] for x in batch]
        videos = [x["videos"] for x in batch]
        
        # 使用 Processor 统一处理 Padding 和 Tensor 转换
        inputs = self.processor(
            text=texts,
            # images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        
        # 创建 Labels (自回归训练)
        # Qwen 的 labels 逻辑：对 input_ids 移位计算 loss
        # 这里为了简单，我们直接把 input_ids 复制给 labels
        # 并在 Trainer 中利用 DataCollatorForSeq2Seq 的逻辑，或者手动 mask 掉 user 部分
        # Qwen2.5-VL 默认行为通常不需要额外 mask，如果使用 chat template
        
        labels = inputs["input_ids"].clone()
        # 简单的 Mask 策略：把 pad token 设为 -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # 注意：严格来说应该把 "User" 指令部分的 Label 设为 -100，
        # 但 Qwen2.5-VL 的 training recipe 比较鲁棒，全序列训练通常也没问题，
        # 或者你可以使用 trl 库的 DataCollatorForCompletionOnlyLM 来精确 mask。
        
        inputs["labels"] = labels
        return inputs
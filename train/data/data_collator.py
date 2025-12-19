import torch
from transformers import Qwen2_5_VLProcessor

class DataCollatorForQwenMIntRec:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

        # 提前把 assistant 起始 token 编码好（非常关键）
        self.assistant_start_ids = self.tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )

    def __call__(self, batch):
        texts = [x["text_input"] for x in batch]
        videos = [x["videos"] for x in batch]

        inputs = self.processor(
            text=texts,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        # 1. 先 mask pad token
        labels[labels == self.tokenizer.pad_token_id] = -100

        # 2. 精确 mask user / system，只保留 assistant 回复
        for i in range(input_ids.size(0)):
            seq = input_ids[i].tolist()

            start_idx = None
            # 在 token 序列中查找 <|im_start|>assistant
            for j in range(len(seq) - len(self.assistant_start_ids)):
                if seq[j : j + len(self.assistant_start_ids)] == self.assistant_start_ids:
                    start_idx = j + len(self.assistant_start_ids)
                    break

            if start_idx is None:
                # 没找到 assistant，保险起见整条不算 loss
                labels[i, :] = -100
            else:
                # assistant 之前全部 mask
                labels[i, :start_idx] = -100

        inputs["labels"] = labels
        return inputs

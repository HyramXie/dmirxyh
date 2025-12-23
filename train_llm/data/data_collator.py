import torch

def collate_fn(self, batch):
    # 需要 pad input_ids 和 labels 到同一长度
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # 使用 padding pad 到最长
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    attention_mask = input_ids_padded.ne(self.tokenizer.pad_token_id).long()
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded,
        "pixel_values": pixel_values
    }
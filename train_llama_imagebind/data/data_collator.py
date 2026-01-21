import torch

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Llama-3 有时默认没有 pad_token，这里做一个兜底
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        video_paths = [item['video_path'] for item in batch]
        audio_paths = [item['audio_path'] for item in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )

        attention_mask = input_ids_padded.ne(self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "video_paths": video_paths,
            "audio_paths": audio_paths
        }

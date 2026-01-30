from transformers import Trainer
import torch
import os

class MultimodalTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # =====================================
        # 1. 拿到真实模型（DDP / FSDP 安全）
        # =====================================
        model = self.model
        if hasattr(model, "module"):
            model = model.module

        # =====================================
        # 2. 明确保存 LLM + LoRA（不走 super）
        # =====================================
        model.llm.save_pretrained(output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # =====================================
        # 3. 保存可选模块（None-safe）
        # =====================================
        projector = getattr(model, "projector", None)
        if projector is not None:
            torch.save(
                projector.state_dict(),
                os.path.join(output_dir, "projector.pt")
            )

        fusion = getattr(model, "fusion_module", None)
        if fusion is not None:
            torch.save(
                fusion.state_dict(),
                os.path.join(output_dir, "fusion_module.pt")
            )

        moe = getattr(model, "vision_moe", None)
        if moe is not None:
            torch.save(
                moe.state_dict(),
                os.path.join(output_dir, "vision_moe.pt")
            )
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

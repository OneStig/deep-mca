"""
Pretraining (causal LM) on unlabeled x86 basic block hex corpus.

Usage:
  uv run deep-mca-pretrain --config configs/pretrain.yaml
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from deep_mca.data import PAD_ID, BOS_ID, EOS_ID, VOCAB_SIZE, hex_to_tokens


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay to 0 (same style as finetune)."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class HFHexStream(IterableDataset):
    """
    Iterable dataset streaming hex strings from a Hugging Face dataset.

    Each yielded item is a 1D LongTensor token sequence with BOS/EOS,
    truncated to max_seq_len.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        field: str,
        max_seq_len: int,
        streaming: bool = True,
        shuffle_buffer: int = 0,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.field = field
        self.max_seq_len = max_seq_len
        self.streaming = streaming
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "Missing dependency: datasets. Install with:\n"
                "  uv add datasets\n"
                "or\n"
                "  pip install datasets\n"
            ) from e

        ds = load_dataset(self.dataset_name, split=self.split, streaming=self.streaming)

        # For streaming datasets, shuffle uses a buffer (approximate shuffle).
        if self.shuffle_buffer and self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer)

        for ex in ds:
            hex_str = ex.get(self.field)
            if not hex_str or not isinstance(hex_str, str):
                continue

            # Guard odd-length hex strings (shouldn't happen often, but safe)
            if len(hex_str) % 2 != 0:
                hex_str = hex_str[:-1]
                if len(hex_str) == 0:
                    continue

            # Tokenize: [BOS] + bytes + [EOS]
            try:
                tokens = hex_to_tokens(hex_str)
            except ValueError:
                # bytes.fromhex can throw if malformed
                continue

            # Truncate to max_seq_len while preserving BOS/EOS
            if len(tokens) > self.max_seq_len:
                # keep BOS, keep first (max_seq_len - 2) bytes, add EOS
                tokens = [BOS_ID] + tokens[1 : self.max_seq_len - 1] + [EOS_ID]

            yield torch.tensor(tokens, dtype=torch.long)


def collate_lm(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    input_ids = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
    attention_mask = (input_ids != PAD_ID).long()

    labels = input_ids.clone()
    labels[input_ids == PAD_ID] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}



def get_backbone_from_lm_model(model: nn.Module) -> nn.Module:
    """
    Hugging Face naming can vary; handle common cases robustly.
    We want the module that is compatible with `MambaModel` state_dict.
    """
    # Common names seen in HF models
    for name in ("backbone", "mamba", "model"):
        if hasattr(model, name):
            cand = getattr(model, name)
            # Some HF classes wrap a .model which is itself the backbone
            # but we want something whose state_dict matches MambaModel.
            if cand is not None:
                return cand
    raise AttributeError(
        "Could not find backbone on LM model. Expected attribute one of: "
        "backbone, mamba, model. Inspect the model structure to update this."
    )


def train(config: dict[str, Any]) -> None:
    cfg_model = config["model"]
    cfg_data = config["data"]
    cfg_train = config["training"]
    cfg_wandb = config.get("wandb", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(int(cfg_train.get("seed", 42)))

    # Optional wandb
    run = None
    try:
        import wandb

        run = wandb.init(
            project=cfg_wandb.get("project", "deep-mca-pretrain"),
            entity=cfg_wandb.get("entity"),
            name=cfg_wandb.get("name"),
            config=config,
        )
    except ImportError:
        print("wandb not installed, skipping logging")

    # Data (streaming HF)
    ds = HFHexStream(
        dataset_name=cfg_data["dataset"],
        split=cfg_data.get("split", "train"),
        field=cfg_data.get("field", "hex"),
        max_seq_len=int(cfg_data["max_seq_len"]),
        streaming=bool(cfg_data.get("streaming", True)),
        shuffle_buffer=int(cfg_data.get("shuffle_buffer", 0)),
    )

    loader = DataLoader(
        ds,
        batch_size=int(cfg_train["batch_size"]),
        collate_fn=collate_lm,
        num_workers=0,  # streaming + HF dataset iterables: keep 0 unless you know what you're doing
        pin_memory=(device.type == "cuda"),
    )

    # Model: MambaForCausalLM
    try:
        from transformers import MambaConfig, MambaForCausalLM
    except ImportError as e:
        raise ImportError(
            "Missing dependency: transformers. Install with:\n"
            "  uv add transformers\n"
            "or\n"
            "  pip install transformers\n"
        ) from e

    mcfg = MambaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=int(cfg_model["hidden_size"]),
        num_hidden_layers=int(cfg_model["num_layers"]),
        state_size=int(cfg_model["state_size"]),
        pad_token_id=PAD_ID,
    )
    model = MambaForCausalLM(mcfg).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"LM parameters: {param_count:,}")
    print(f"VOCAB_SIZE={VOCAB_SIZE} PAD_ID={PAD_ID} BOS_ID={BOS_ID} EOS_ID={EOS_ID}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_train["lr"]),
        weight_decay=float(cfg_train["weight_decay"]),
    )

    max_steps = int(cfg_train["max_steps"])
    warmup_steps = int(max_steps * float(cfg_train.get("warmup_ratio", 0.0)))
    scheduler = build_scheduler(optimizer, warmup_steps, max_steps)

    grad_clip = float(cfg_train.get("grad_clip", 1.0))
    log_interval = int(cfg_train.get("log_interval", 50))
    save_interval = int(cfg_train.get("save_interval", 2000))

    ckpt_dir = Path(cfg_train.get("checkpoint_dir", "checkpoints/pretrain"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda",enabled=use_amp)
    model.train()
    global_step = 0

    # Iterate until max_steps (streaming loader is infinite-ish)
    data_iter = iter(loader)
    while global_step < max_steps:
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda",enabled=use_amp, dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        prev_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= prev_scale:
            scheduler.step()

        global_step += 1

        if global_step % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            msg = f"step {global_step}/{max_steps}: loss={loss.item():.4f} lr={lr:.2e}"
            print(msg)
            if run:
                run.log({"train/loss": loss.item(), "train/lr": lr, "global_step": global_step}, step=global_step)

        if global_step % save_interval == 0 or global_step == max_steps:
            # Save backbone weights for regressor loading
            backbone = get_backbone_from_lm_model(model)
            backbone_path = ckpt_dir / f"backbone_step{global_step}.pt"
            torch.save(backbone.state_dict(), backbone_path)
            print(f"Saved backbone to {backbone_path}")

            # Also save "latest" symlink-like file
            latest_path = ckpt_dir / "backbone_latest.pt"
            torch.save(backbone.state_dict(), latest_path)

    # Final save
    backbone = get_backbone_from_lm_model(model)
    final_path = ckpt_dir / "backbone.pt"
    torch.save(backbone.state_dict(), final_path)
    print(f"Pretraining complete. Saved final backbone to {final_path}")

    if run:
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()

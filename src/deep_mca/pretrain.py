"""
Pretraining (causal LM) on unlabeled x86 basic block hex corpus.

Usage:
  uv run deep-mca-pretrain --config configs/pretrain.yaml
"""

from __future__ import annotations
import math
import random
from transformers import MambaConfig, MambaForCausalLM
import argparse
import os
from pathlib import Path
from typing import Any
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

from deep_mca.data import TextAssemblyLMTokenizer
from deep_mca.utils import build_schedule


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



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
        tokenizer: TextAssemblyLMTokenizer | None = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.field = field
        self.max_seq_len = max_seq_len
        self.streaming = streaming
        self.shuffle_buffer = shuffle_buffer
        self.tokenizer = tokenizer or TextAssemblyLMTokenizer()

    def __iter__(self):
        
        ds = load_dataset(self.dataset_name, split=self.split, streaming=self.streaming)

        # For streaming datasets, shuffle uses a buffer (approximate shuffle).
        if self.shuffle_buffer and self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer)

        for ex in ds:
            block = ex.get(self.field)
            if not block:
                continue

            if isinstance(block, str):
                instr_list = [block]
            elif isinstance(block, list):
                instr_list = [str(item) for item in block if item]
            else:
                continue

            tokens = self.tokenizer.encode_block(instr_list)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[: self.max_seq_len - 1] + [self.tokenizer.eos_id]

            yield torch.tensor(tokens, dtype=torch.long)


def collate_lm(batch: list[torch.Tensor], pad_id: int) -> dict[str, torch.Tensor]:
    input_ids = pad_sequence(batch, batch_first=True, padding_value=pad_id)
    attention_mask = (input_ids != pad_id).long()

    labels = input_ids.clone()
    labels[input_ids == pad_id] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

@torch.no_grad()
def evaluate(
    model: MambaForCausalLM,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0

    for batch in loader:
        if batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss

        active_tokens = (labels != -100).sum().item()
        total_tokens += active_tokens
        total_loss += loss.item() * active_tokens
        batches += 1

    model.train()
    if total_tokens == 0:
        return {"eval/loss": float("nan"), "eval/ppl": float("nan")}
    avg_loss = total_loss / total_tokens
    return {"eval/loss": avg_loss, "eval/ppl": math.exp(avg_loss)}


def train(config: dict[str, Any]) -> None:
    cfg_model = config["model"]
    cfg_data = config["data"]
    cfg_train = config["training"]
    cfg_wandb = config.get("wandb", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(int(cfg_train.get("seed", 42)))

    # -- wandb --
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
    
    # -- data --
    tokenizer = TextAssemblyLMTokenizer()
    
    train_ds = HFHexStream(
        dataset_name=cfg_data["dataset"],
        split=cfg_data.get("split", "train"),
        field=cfg_data.get("field", "hex"),
        max_seq_len=int(cfg_data["max_seq_len"]),
        streaming=bool(cfg_data.get("streaming", True)),
        shuffle_buffer=int(cfg_data.get("shuffle_buffer", 0)),
        tokenizer=tokenizer
    )

    loader = DataLoader(
        train_ds,
        batch_size=int(cfg_train["batch_size"]),
        collate_fn=lambda batch: collate_lm(batch, tokenizer.pad_id),   
        num_workers=0,  # streaming + HF dataset iterables: keep 0 unless you know what you're doing
        pin_memory=(device.type == "cuda"),
    )
    eval_loader = None
    eval_split = cfg_data.get("eval_split")
    if eval_split:
        try:
            eval_ds = HFHexStream(
                dataset_name=cfg_data["dataset"],
                split=eval_split,
                field=cfg_data.get("field", "hex"),
                max_seq_len=int(cfg_data["max_seq_len"]),
                streaming=bool(cfg_data.get("eval_streaming", cfg_data.get("streaming", True))),
                shuffle_buffer=0,
                tokenizer=tokenizer,
            )
            eval_loader = DataLoader(
                eval_ds,
                batch_size=int(cfg_train["batch_size"]),
                collate_fn=lambda batch: collate_lm(batch, tokenizer.pad_id),
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )
        except Exception as exc:
            print(f"Could not create eval loader for split '{eval_split}': {exc}")
            eval_loader = None
    # Model: MambaForCausalLM

    mcfg = MambaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=int(cfg_model["hidden_size"]),
        num_hidden_layers=int(cfg_model["num_layers"]),
        state_size=int(cfg_model["state_size"]),
        pad_token_id=tokenizer.pad_id,
    )
    model = MambaForCausalLM(mcfg).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"LM parameters: {param_count:,}")
    print(f"VOCAB_SIZE={tokenizer.vocab_size} PAD_ID={tokenizer.pad_id}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_train["lr"]),
        weight_decay=float(cfg_train["weight_decay"]),
    )

    max_steps = int(cfg_train["max_steps"])
    warmup_steps = int(max_steps * float(cfg_train.get("warmup_ratio", 0.0)))
    scheduler = build_schedule(optimizer, warmup_steps, max_steps)

    grad_clip = float(cfg_train.get("grad_clip", 1.0))
    log_interval = int(cfg_train.get("log_interval", 50))
    save_interval = int(cfg_train.get("save_interval", 2000))
    eval_interval = int(cfg_train.get("eval_interval", 0))
    eval_batches = int(cfg_train.get("eval_batches", 10))

    ckpt_dir = Path(cfg_train.get("checkpoint_dir", "checkpoints/pretrain"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda",enabled=use_amp)
    model.train()
    global_step = 0

    # Iterate until max_steps (streaming loader is infinite-ish)
    data_iter = iter(loader)
    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            continue
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        max_token_id = int(input_ids.max().item())
        if max_token_id >= model.config.vocab_size:
            new_size = max_token_id + 1
            model.resize_token_embeddings(new_size)
            model.config.vocab_size = new_size
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
                run.log(
                    {"train/loss": loss.item(), "train/lr": lr, "global_step": global_step},
                    step=global_step,
                )

        if eval_loader and eval_interval > 0 and global_step % eval_interval == 0:
            eval_metrics = evaluate(model, eval_loader, device, eval_batches, use_amp)
            print(
                f"eval step {global_step}: loss={eval_metrics['eval/loss']:.4f} "
                f"ppl={eval_metrics['eval/ppl']:.2f}"
            )
            if run:
                eval_metrics["global_step"] = global_step
                run.log(eval_metrics, step=global_step)
        if global_step % save_interval == 0 or global_step == max_steps:
            # Save backbone weights for regressor loading
            backbone_path = ckpt_dir / f"backbone_step{global_step}.pt"
            torch.save(model.backbone.state_dict(), backbone_path)
            print(f"Saved backbone to {backbone_path}")

            # Also save "latest" symlink-like file
            latest_path = ckpt_dir / "backbone_latest.pt"
            torch.save(model.backbone.state_dict(), latest_path)

    # Final save
    final_path = ckpt_dir / "backbone.pt"
    torch.save(model.backbone.state_dict(), final_path)
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

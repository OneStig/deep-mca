"""
Pretraining (causal LM) on unlabeled x86 basic block hex corpus.

Usage:
  uv run deep-mca-pretrain --config configs/pretrain.yaml
"""

import argparse
import math
import random
from pathlib import Path

import torch
import wandb
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import MambaConfig, MambaForCausalLM

from deep_mca.data import EOS_ID, PAD_ID, VOCAB_SIZE, CollateLM, hex_to_tokens
from deep_mca.utils import build_scheduler

# Text Assembly LM Tokenizer is a stand-in, replace with proper tokenization later


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HFHexMap(Dataset):
    """
    Hugging Face dataset.

    Each yielded item is a 1D LongTensor token sequence with BOS/EOS,
    truncated to max_seq_len.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        field: str,
        max_seq_len: int,
    ):
        super().__init__()
        self.ds = load_dataset(dataset_name, split=split, streaming=False)
        self.field = field
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ex = self.ds[idx]
        hex_str = ex.get(self.field)

        if not hex_str or not isinstance(hex_str, str):
            return torch.tensor([], dtype=torch.long)

        tokens = hex_to_tokens(hex_str)

        # Truncate and force EOS
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len - 1] + [EOS_ID]

        return torch.tensor(tokens, dtype=torch.long)


@torch.no_grad()
def evaluate(
    model: MambaForCausalLM,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss  # mean over non -100 labels in this batch

        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    model.train()

    if total_tokens == 0:
        return {"eval/loss": float("nan"), "eval/ppl": float("nan")}

    avg_loss = total_loss / total_tokens
    return {"eval/loss": avg_loss, "eval/ppl": math.exp(avg_loss)}


def train(config: dict) -> None:
    cfg_model = config["model"]
    cfg_data = config["data"]
    cfg_train = config["training"]
    cfg_wandb = config.get("wandb", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(cfg_train["seed"])

    # -- wandb --

    run = wandb.init(
        project=cfg_wandb.get("project", "deep-mca-pretrain"),
        entity=cfg_wandb.get("entity"),
        name=cfg_wandb.get("name"),
        config=config,
    )

    # -- data --

    train_ds = HFHexMap(
        dataset_name=cfg_data["dataset"],
        split=cfg_data["split"],
        field=cfg_data["field"],
        max_seq_len=cfg_data["max_seq_len"],
    )

    loader = DataLoader(
        train_ds,
        batch_size=cfg_train["batch_size"],
        shuffle=True,
        collate_fn=CollateLM(PAD_ID),
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    eval_loader = None
    eval_split = cfg_data.get("eval_split")

    # can toggle in pretrain.yaml
    if eval_split:
        eval_ds = HFHexMap(
            dataset_name=cfg_data["dataset"],
            split=eval_split,
            field=cfg_data["field"],
            max_seq_len=cfg_data["max_seq_len"],
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=cfg_train["batch_size"],
            shuffle=False,
            collate_fn=CollateLM(PAD_ID),
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
    # Model: MambaForCausalLM

    mcfg = MambaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=cfg_model["hidden_size"],
        num_hidden_layers=cfg_model["num_layers"],
        state_size=cfg_model["state_size"],
        pad_token_id=PAD_ID,
    )
    model = MambaForCausalLM(mcfg).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"LM parameters: {param_count:,}")
    print(f"VOCAB_SIZE={VOCAB_SIZE} PAD_ID={PAD_ID}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_train["lr"]),
        weight_decay=float(cfg_train["weight_decay"]),
    )
    epochs = int(cfg_train["epochs"])
    total_steps = len(loader) * epochs
    warmup_steps = int(total_steps * float(cfg_train["warmup_ratio"]))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    grad_clip = float(cfg_train["grad_clip"])
    log_interval = cfg_train["log_interval"]

    ckpt_dir = Path(cfg_train["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda"
    model.train()
    global_step = 0
    best_eval_loss = float("inf")

    # For finite number of epochs instead of steps
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_n = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            n_tokens = (labels != -100).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_n += n_tokens
            global_step += 1

            # Print base metrics for each log interval step
            if global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                msg = f"step {global_step}/{total_steps}: loss={loss.item():.4f} lr={lr:.2e}"
                print(msg)
                if run:
                    run.log(
                        {"train/loss": loss.item(), "train/lr": lr, "global_step": global_step},
                        step=global_step,
                    )

        avg_train_loss = epoch_loss / epoch_n
        print(f"Epoch {epoch + 1}/{epochs} complete. Avg train loss: {avg_train_loss:.4f}")

        # Only run the evaluation if the split is defined in configs
        if eval_loader:
            eval_metrics = evaluate(model, eval_loader, device)
            print(
                f"Epoch {epoch + 1} eval: "
                f"loss={eval_metrics['eval/loss']:.4f} "
                f"ppl={eval_metrics['eval/ppl']:.2f}"
            )
            if run:
                run.log(
                    {"train/epoch_loss": avg_train_loss, "epoch": epoch + 1, **eval_metrics},
                    step=global_step,
                )

            if eval_metrics["eval/loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["eval/loss"]
                ckpt_path = ckpt_dir / "best_backbone.pt"
                torch.save(model.backbone.state_dict(), ckpt_path)
                print(f"Saved best model to {ckpt_path}")
    # Final save
    final_path = ckpt_dir / "final_backbone.pt"

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

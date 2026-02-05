"""
Performs fine tuning of the final model on throughput data.

uv run deep-mca-finetune --config configs/finetune.yaml
"""

import argparse
import math
from pathlib import Path

import torch
import yaml
from scipy.stats import kendalltau
from torch.utils.data import DataLoader

from deep_mca.data import BHiveDataset, collate_fn
from deep_mca.model import MambaRegressor


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay to 0"""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(
    model: MambaRegressor,
    loader: DataLoader,
    device: torch.device,
    log_targets: bool = True,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mape = 0.0
    all_preds: list[float] = []
    all_targets: list[float] = []
    n = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        targets = batch["targets"].to(device)

        preds = model(input_ids, lengths)
        total_loss += torch.nn.functional.huber_loss(preds, targets, reduction="sum").item()

        # Convert back to original scale for interpretable metrics
        if log_targets:
            preds_orig = preds.exp()
            targets_orig = targets.exp()
        else:
            preds_orig = preds
            targets_orig = targets

        total_mae += (preds_orig - targets_orig).abs().sum().item()
        total_mape += ((preds_orig - targets_orig).abs() / targets_orig.abs()).sum().item()
        all_preds.extend(preds_orig.cpu().tolist())
        all_targets.extend(targets_orig.cpu().tolist())
        n += targets.size(0)

    tau, _ = kendalltau(all_preds, all_targets)
    model.train()
    return {
        "eval/loss": total_loss / n,
        "eval/mae": total_mae / n,
        "eval/mape": total_mape / n * 100,
        "eval/kendall_tau": tau,
    }


def train(config: dict) -> None:
    cfg_model = config["model"]
    cfg_data = config["data"]
    cfg_train = config["training"]
    cfg_wandb = config.get("wandb", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -- wandb --
    run = None
    try:
        import wandb

        run = wandb.init(
            project=cfg_wandb.get("project", "deep-mca"),
            entity=cfg_wandb.get("entity"),
            name=cfg_wandb.get("name"),
            config=config,
        )
    except ImportError:
        print("wandb not installed, skipping logging")

    # -- data --
    log_targets = cfg_data.get("log_targets", True)
    train_ds = BHiveDataset(
        cfg_data["bhive_csv"],
        max_seq_len=cfg_data["max_seq_len"],
        split="train",
        train_ratio=cfg_data["train_ratio"],
        log_targets=log_targets,
    )
    eval_ds = BHiveDataset(
        cfg_data["bhive_csv"],
        max_seq_len=cfg_data["max_seq_len"],
        split="eval",
        train_ratio=cfg_data["train_ratio"],
        log_targets=log_targets,
    )
    print(f"Train: {len(train_ds)} samples, Eval: {len(eval_ds)} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_train["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg_train["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # -- model --
    pretrained_path = cfg_model.get("pretrained_path")
    dropout = float(cfg_model.get("dropout", 0.0))
    if pretrained_path:
        print(f"Loading pretrained backbone from {pretrained_path}")
        model = MambaRegressor.from_pretrained_backbone(
            pretrained_path,
            hidden_size=cfg_model["hidden_size"],
            num_layers=cfg_model["num_layers"],
            state_size=cfg_model["state_size"],
            dropout=dropout,
        )
    else:
        model = MambaRegressor(
            hidden_size=cfg_model["hidden_size"],
            num_layers=cfg_model["num_layers"],
            state_size=cfg_model["state_size"],
            dropout=dropout,
        )
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # -- optimizer & scheduler --
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_train["lr"]),
        weight_decay=float(cfg_train["weight_decay"]),
    )
    total_steps = len(train_loader) * cfg_train["epochs"]
    warmup_steps = int(total_steps * float(cfg_train["warmup_ratio"]))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    # -- training loop --
    checkpoint_dir = Path(cfg_train["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_eval_loss = float("inf")
    global_step = 0
    log_interval = cfg_train["log_interval"]

    for epoch in range(cfg_train["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_n = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["targets"].to(device)

            preds = model(input_ids, lengths)
            loss = torch.nn.functional.huber_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * targets.size(0)
            epoch_n += targets.size(0)
            global_step += 1

            if global_step % log_interval == 0:
                metrics = {
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "global_step": global_step,
                }
                print(
                    f"  step {global_step}: loss={loss.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
                if run:
                    run.log(metrics, step=global_step)

        avg_train_loss = epoch_loss / epoch_n
        eval_metrics = evaluate(model, eval_loader, device, log_targets=log_targets)
        print(
            f"Epoch {epoch + 1}/{cfg_train['epochs']}: "
            f"train_loss={avg_train_loss:.4f} "
            f"eval_loss={eval_metrics['eval/loss']:.4f} "
            f"eval_mae={eval_metrics['eval/mae']:.4f} "
            f"eval_mape={eval_metrics['eval/mape']:.2f}% "
            f"eval_tau={eval_metrics['eval/kendall_tau']:.4f}"
        )

        log_data = {"train/epoch_loss": avg_train_loss, "epoch": epoch + 1, **eval_metrics}
        if run:
            run.log(log_data, step=global_step)

        if eval_metrics["eval/loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["eval/loss"]
            ckpt_path = checkpoint_dir / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best model to {ckpt_path}")

    # Save final model (although almost certainly this will be overfit over best_model)
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pt")
    print(f"Training complete. Best eval loss: {best_eval_loss:.4f}")

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

"""
Uploads the weights to hf hub
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import yaml
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file

from deep_mca.tokenizer import Tokenizer

REPO = "stevenhe04/deep-mca"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.safetensors"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/finetune.yaml"),
    )
    parser.add_argument(
        "--arch",
        required=True,
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]

    tokenizer = Tokenizer(cfg["data"]["vocab_path"])

    tmp_dir = tempfile.mkdtemp()
    try:
        state_dict = load_file(args.checkpoint)
        save_file(state_dict, Path(tmp_dir) / "model.safetensors")

        config = {
            "hidden_size": model_cfg["hidden_size"],
            "num_layers": model_cfg["num_layers"],
            "state_size": model_cfg["state_size"],
            "dropout": model_cfg["dropout"],
            "vocab_size": tokenizer.vocab_size,
        }
        with open(Path(tmp_dir) / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        api = HfApi()
        api.create_repo(REPO, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=REPO,
            repo_type="model",
            path_in_repo=args.arch,
        )

        print(f"uploaded weights to https://huggingface.co/{REPO}/tree/main/{args.arch}")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()

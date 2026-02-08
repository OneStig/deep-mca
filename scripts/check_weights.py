import json
import sys
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from deep_mca.data import VOCAB_SIZE
from deep_mca.hub import load_from_hub

REPO = "stevenhe04/deep-mca"
ARCH = "skylake"


def main() -> None:
    config_yaml = Path("configs/finetune.yaml")
    with open(config_yaml) as f:
        local_cfg = yaml.safe_load(f)["model"]

    expected = {
        "hidden_size": local_cfg["hidden_size"],
        "num_layers": local_cfg["num_layers"],
        "state_size": local_cfg["state_size"],
        "dropout": local_cfg["dropout"],
        "vocab_size": VOCAB_SIZE,
    }

    remote_config_path = hf_hub_download(repo_id=REPO, filename=f"{ARCH}/config.json")
    with open(remote_config_path) as f:
        remote = json.load(f)

    mismatches = []
    for key, local_val in expected.items():
        remote_val = remote.get(key)
        if remote_val != local_val:
            mismatches.append(f"  {key}: local={local_val} remote={remote_val}")

    if mismatches:
        print("config.json out of sync with finetune.yaml:")
        print("\n".join(mismatches))
        sys.exit(1)

    print("config.json matches finetune.yaml")

    model = load_from_hub(repo_id=REPO, arch=ARCH)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model loaded successfully ({n_params:,} parameters)")


if __name__ == "__main__":
    main()

"""
Downloads weights from hf and loads into a model
"""

import json

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from deep_mca.model import MambaRegressor


def load_from_hub(
    repo_id: str = "stevenhe04/deep-mca",
    arch: str = "skylake",
    revision: str = "main",
) -> MambaRegressor:
    config_path = hf_hub_download(
        repo_id=repo_id, filename=f"{arch}/config.json", revision=revision
    )
    weights_path = hf_hub_download(
        repo_id=repo_id, filename=f"{arch}/model.safetensors", revision=revision
    )

    with open(config_path) as f:
        config = json.load(f)

    model = MambaRegressor(
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        state_size=config["state_size"],
        dropout=config["dropout"],
    )

    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model

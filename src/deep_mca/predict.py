import math

import torch
import typer

from deep_mca.data import hex_to_tokens
from deep_mca.hub import load_from_hub
from deep_mca.model import MambaRegressor

_model_cache: dict[tuple[str, str], MambaRegressor] = {}


def _get_model(repo_id: str, arch: str) -> MambaRegressor:
    key = (repo_id, arch)
    if key not in _model_cache:
        _model_cache[key] = load_from_hub(repo_id=repo_id, arch=arch)
    return _model_cache[key]


def predict(
    hex_str: str,
    arch: str = "skylake",
    repo_id: str = "stevenhe04/deep-mca",
) -> float:
    model = _get_model(repo_id, arch)
    
    # For now just input the hex, but once we finish tokenization
    # should just input assembly directly @teddy @huy
    tokens = hex_to_tokens(hex_str)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)

    with torch.no_grad():
        log_pred = model(input_ids, lengths)

    return math.exp(log_pred.item())


app = typer.Typer()


@app.command()
def cli(
    hex_str: str = typer.Option(..., "--hex"),
    arch: str = typer.Option("skylake", "--arch"),
) -> None:
    cycles = predict(hex_str, arch=arch)
    print(f"{cycles:.2f}")


def main() -> None:
    app()

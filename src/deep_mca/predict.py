import math

import torch
import typer

from deep_mca.hub import load_from_hub
from deep_mca.model import MambaRegressor
from deep_mca.tokenizer import Tokenizer

_model_cache: dict[tuple[str, str], MambaRegressor] = {}
_tokenizer_cache: dict[str, Tokenizer] = {}


def predict(
    assembly: str,
    vocab_path: str,
    arch: str = "skylake",
    repo_id: str = "stevenhe04/deep-mca",
) -> float:
    if vocab_path not in _tokenizer_cache:
        _tokenizer_cache[vocab_path] = Tokenizer(vocab_path)
    tokenizer = _tokenizer_cache[vocab_path]

    model_key = (repo_id, arch)
    if model_key not in _model_cache:
        _model_cache[model_key] = load_from_hub(repo_id=repo_id, arch=arch)
    model = _model_cache[model_key]

    tokens = tokenizer.parse_block_to_ids(assembly)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)

    with torch.no_grad():
        log_pred = model(input_ids, lengths)

    return math.exp(log_pred.item())


app = typer.Typer()


@app.command()
def cli(
    assembly: str = typer.Option(..., "--asm"),
    vocab_path: str = typer.Option(..., "--vocab"),
    arch: str = typer.Option("skylake", "--arch"),
) -> None:
    cycles = predict(assembly, vocab_path=vocab_path, arch=arch)
    print(f"{cycles:.2f}")


def main() -> None:
    app()

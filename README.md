# deep-mca

ML-based x86-64 CPU throughput prediction using the Mamba architecture, trained on the [BHive](https://github.com/ithemal/bhive) benchmark dataset.

## Setup

Requires `uv` and LLVM tools (`llvm-mc`, `llvm-mca`, `llvm-objdump`).

```bash
uv sync
uv run scripts/check_env.py  # verify environment
```

On CUDA machines, install optimized Mamba kernels for faster training:

```bash
uv sync --group cuda
```

Without the CUDA group, training still works using HuggingFace's pure-PyTorch Mamba fallback.

## Fine-tuning

```bash
uv run deep-mca-finetune --config configs/finetune.yaml
```

Trains a Mamba model with a regression head on BHive Skylake throughput data. Logs training loss, eval MAE, MAPE, and Kendall's tau to [wandb](https://wandb.ai). Run `wandb login` first to enable dashboard logging.

Edit `configs/finetune.yaml` to adjust model size, learning rate, epochs, etc.

## Lint

```bash
./scripts/lint.sh
```

## Data

- BHive (git submodule): `data/bhive/benchmark/throughput/skl.csv`
- Pretraining corpus: [stevenhe04/x86-bb-24m](https://huggingface.co/datasets/stevenhe04/x86-bb-24m)

## References

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://github.com/state-spaces/mamba)
- [Ithemal: Accurate, Portable and Fast Basic Block Throughput Estimation](https://arxiv.org/pdf/1808.07412)
- [BHive: A Benchmark Suite and Measurement Framework](https://dl.acm.org/doi/pdf/10.1145/3640537.3641572)
- [Learning to Optimize Tensor Programs](https://ieeexplore.ieee.org/document/9042166)

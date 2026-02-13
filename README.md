# deep-mca


## Setup

Requires `uv` and LLVM tools.

```bash
uv sync
uv run scripts/check_env.py
```

If available install mamba-ssm CUDA kernels for speeeeeed.

```bash
uv sync --group cuda
```


## Finetuning

```bash
uv run deep-mca-finetune --config configs/finetune.yaml
```

## Lint

```bash
./scripts/lint.sh
```

## Data

- Pretraining corpus: [stevenhe04/x86-bb-24m](https://huggingface.co/datasets/stevenhe04/x86-bb-24m)
- Pretraining assembly dataset: [henryc13/x86-pretrain](https://huggingface.co/datasets/henryc13/x86-pretrain)

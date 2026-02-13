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

## References

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://github.com/state-spaces/mamba)
- [Ithemal: Accurate, Portable and Fast Basic Block Throughput Estimation](https://arxiv.org/pdf/1808.07412)
- [BHive: A Benchmark Suite and Measurement Framework](https://dl.acm.org/doi/pdf/10.1145/3640537.3641572)
- [Learning to Optimize Tensor Programs](https://ieeexplore.ieee.org/document/9042166)

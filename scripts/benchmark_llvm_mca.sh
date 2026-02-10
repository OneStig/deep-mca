#!/usr/bin/env bash

uv run benchmarks/benchmark_llvm_mca.py \
    --dataset-name Arcticbun/skl_x86 \
    --split train \
    --subset eval \
    --mcpu skylake \
    --output-csv results/skl_x86_train_eval_mca_results.csv

uv run benchmarks/benchmark_llvm_mca.py \
    --dataset-name Arcticbun/hsw_x86 \
    --split train \
    --subset eval \
    --mcpu haswell \
    --output-csv results/hsw_x86_train_eval_mca_results.csv

uv run benchmarks/benchmark_llvm_mca.py \
    --dataset-name Arcticbun/ivb_x86 \
    --split train \
    --subset eval \
    --mcpu ivybridge \
    --output-csv results/ivb_x86_train_eval_mca_results.csv
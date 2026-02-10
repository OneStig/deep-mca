import argparse
import csv
import subprocess
from pathlib import Path

from datasets import load_dataset
from scipy.stats import kendalltau


def wrap_asm(lines: list[str]) -> str:
    """Wrap basic block in a label so llvm-mca can parse it."""
    body = "\n  ".join(lines)
    return f""".text
.globl bb
bb:
  {body}
"""


def run_llvm_mca(asm: str, mcpu: str = "skylake", iterations: int = 100) -> float | None:
    """Run llvm-mca and return the block reciprocal throughput."""
    cmd = [
        "llvm-mca",
        "-mtriple=x86_64",
        f"-mcpu={mcpu}",
        f"-iterations={iterations}",
    ]
    proc = subprocess.run(
        cmd,
        input=asm,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None

    for line in proc.stdout.splitlines():
        if "Block RThroughput:" in line:
            return float(line.split(":")[1].strip())
    return None


def benchmark_block(
    instructions: str | list[str],
    mcpu: str = "skylake",
    iterations: int = 100,
) -> float | None:
    """Benchmark a single basic block and return predicted cycles.

    The `instructions` field comes from the Hugging Face dataset and can be either:
    - a single string containing one instruction per line, or
    - a list of instruction strings.
    """
    try:
        if isinstance(instructions, str):
            asm_lines = [line.strip() for line in instructions.splitlines() if line.strip()]
        else:
            asm_lines = [line.strip() for line in instructions if line.strip()]

        if not asm_lines:
            return None

        asm = wrap_asm(asm_lines)
        rthroughput = run_llvm_mca(asm, mcpu=mcpu, iterations=iterations)
        if rthroughput is None:
            return None
        return rthroughput * iterations
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Benchmark llvm-mca on a dataset of x86-64 basic blocks.")
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help=(
            "Hugging Face dataset name or local parquet path. "
            "The dataset must have 'instructions' and 'cycles' columns."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=("Path to output CSV (default: <dataset_name>_<split>_mca_results.csv)"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help=(
            "Base dataset split to load from Hugging Face / parquet "
            "(e.g., 'train', 'validation', 'test', 'train[:1000]', etc.)."
        ),
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["all", "train", "eval"],
        default="all",
        help=(
            "Logical subset to use within the loaded split: "
            "'train' = first 80%, 'eval' = last 20%, 'all' = entire split."
        ),
    )
    parser.add_argument(
        "--mcpu",
        type=str,
        default="skylake",
        help="Target CPU for llvm-mca (default: skylake)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for llvm-mca (default: 100)",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        # Sanitize dataset name for filename purposes
        dataset_stem = args.dataset_name.replace("/", "_")
        output_path = Path(f"{dataset_stem}_{args.split}_mca_results.csv")

    # Load dataset from Hugging Face (or local parquet)
    try:
        if args.dataset_name.endswith(".parquet"):
            # Local parquet file
            dataset = load_dataset("parquet", data_files=args.dataset_name, split=args.split)
        else:
            # Hugging Face Hub dataset
            dataset = load_dataset(args.dataset_name, split=args.split)
    except Exception as e:
        raise SystemExit(f"Failed to load dataset '{args.dataset_name}': {e}")

    print(f"Loaded dataset '{args.dataset_name}' with split spec '{args.split}'.")
    print(f"Total examples before subset: {len(dataset)}")

    # Apply 80/20 train/eval subset if requested
    total_examples = len(dataset)
    if args.subset in {"train", "eval"} and total_examples > 0:
        split_idx = int(total_examples * 0.8)
        if args.subset == "train":
            indices = list(range(0, split_idx))
            print(f"Using TRAIN subset: first {len(indices)} examples (80% of {total_examples})")
        else:  # eval
            indices = list(range(split_idx, total_examples))
            print(f"Using EVAL subset: last {len(indices)} examples (20% of {total_examples})")
        dataset = dataset.select(indices)

    print(f"Processing {len(dataset)} basic blocks...")
    print(f"Target CPU: {args.mcpu}")
    print(f"Output: {output_path}")

    # Benchmark each block and write results
    results: list[tuple[str, float, float]] = []
    errors = 0
    for i, row in enumerate(dataset):
        instructions = row["instructions"]
        ground_truth = float(row["cycles"])

        predicted = benchmark_block(instructions, mcpu=args.mcpu, iterations=args.iterations)
        if predicted is not None:
            # Store the raw instructions text for reproducibility
            if isinstance(instructions, list):
                instr_text = "\n".join(instructions)
            else:
                instr_text = str(instructions)

            results.append((instr_text, ground_truth, predicted))
        else:
            errors += 1

        # Progress update
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} blocks ({errors} errors)")

    # Write output CSV (create parent directory if needed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instructions", "ground_truth_cycles", "mca_predicted_cycles"])
        writer.writerows(results)

    print(f"\nDone! Processed {len(results)} blocks successfully, {errors} errors.")
    print(f"Results saved to: {output_path}")

    # Calculate and print accuracy statistics
    if results:
        ground_truths = [gt for _, gt, _ in results]
        predictions = [pred for _, _, pred in results]

        abs_errors = [abs(gt - pred) for gt, pred in zip(ground_truths, predictions)]
        rel_errors = [
            abs(gt - pred) / gt * 100 for gt, pred in zip(ground_truths, predictions) if gt > 0
        ]

        # Kendall's Tau calculation
        tau, p_value = kendalltau(ground_truths, predictions)

        print("\nStatistics:")
        print(f"  Mean Absolute Error: {sum(abs_errors) / len(abs_errors):.2f} cycles")
        print(f"  Mean Relative Error: {sum(rel_errors) / len(rel_errors):.2f}%")
        print(f"  Kendall's Tau: {tau:.4f} (p-value: {p_value:.2e})")

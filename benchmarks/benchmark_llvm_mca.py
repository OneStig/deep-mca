import argparse
import subprocess
import csv
import sys
from pathlib import Path

from scipy.stats import kendalltau


def disassemble(hex_str: str, output_intel_syntax: bool = False) -> list[str]:
    """Disassemble hex string to assembly lines."""
    args = []
    for i in range(0, len(hex_str), 2):
        byte = hex_str[i : i + 2]
        args.append("0x" + byte)

    syntax_id = 1 if output_intel_syntax else 0
    cmd = "echo {} | llvm-mc -disassemble -triple=x86_64 -output-asm-variant={}".format(
        " ".join(args), syntax_id
    )
    output = subprocess.check_output(cmd, shell=True)

    lines = []
    for line in output.decode("utf8").splitlines():
        line = line.strip()
        if not line or line.startswith("."):
            continue
        lines.append(line)
    return lines


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


def benchmark_block(hex_str: str, mcpu: str = "skylake", iterations: int = 100) -> float | None:
    """Benchmark a single basic block and return predicted cycles."""
    try:
        asm_lines = disassemble(hex_str)
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
    parser = argparse.ArgumentParser(description="Benchmark llvm-mca on BHive dataset")
    parser.add_argument(
        "--throughput-csv",
        type=str,
        required=True,
        help="Path to BHive throughput CSV (e.g., skl.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output CSV (default: <input>_mca_results.csv)",
    )
    parser.add_argument(
        "--mcpu", type=str, default="skylake", help="Target CPU for llvm-mca (default: skylake)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for llvm-mca (default: 100)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of rows to process (for testing)"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "eval", "all"],
        default="all",
        help="Data split to use: 'train' (first 80%%), 'eval' (last 20%%), or 'all' (default)",
    )

    args = parser.parse_args()

    input_path = Path(args.throughput_csv)
    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        output_path = input_path.parent / f"{input_path.stem}_mca_results.csv"

    # Read input CSV
    rows = []
    with open(input_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                hex_str = row[0].strip()
                try:
                    ground_truth = float(row[1])
                except ValueError:
                    continue
                if hex_str:  # Skip empty hex strings
                    rows.append((hex_str, ground_truth))

    # Apply train/eval split
    total_rows = len(rows)
    if args.split == "train":
        split_idx = int(total_rows * 0.8)
        rows = rows[:split_idx]
        print(f"Using TRAIN split: first {len(rows)} rows (80% of {total_rows})")
    elif args.split == "eval":
        split_idx = int(total_rows * 0.8)
        rows = rows[split_idx:]
        print(f"Using EVAL split: last {len(rows)} rows (20% of {total_rows})")

    if args.limit:
        rows = rows[: args.limit]

    print(f"Processing {len(rows)} basic blocks...")
    print(f"Target CPU: {args.mcpu}")
    print(f"Output: {output_path}")

    # Benchmark each block and write results
    results = []
    errors = 0
    for i, (hex_str, ground_truth) in enumerate(rows):
        predicted = benchmark_block(hex_str, mcpu=args.mcpu, iterations=args.iterations)
        if predicted is not None:
            results.append((hex_str, ground_truth, predicted))
        else:
            errors += 1

        # Progress update
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(rows)} blocks ({errors} errors)")

    # Write output CSV (create parent directory if needed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hex", "ground_truth_cycles", "mca_predicted_cycles"])
        writer.writerows(results)

    print(f"\nDone! Processed {len(results)} blocks successfully, {errors} errors.")
    print(f"Results saved to: {output_path}")

    # Calculate and print basic statistics
    if results:
        ground_truths = [gt for _, gt, _ in results]
        predictions = [pred for _, _, pred in results]

        abs_errors = [abs(gt - pred) for gt, pred in zip(ground_truths, predictions)]
        rel_errors = [
            abs(gt - pred) / gt * 100 for gt, pred in zip(ground_truths, predictions) if gt > 0
        ]

        # Kendall's Tau measures rank correlation
        tau, p_value = kendalltau(ground_truths, predictions)

        print("\nStatistics:")
        print(f"  Mean Absolute Error: {sum(abs_errors) / len(abs_errors):.2f} cycles")
        print(f"  Mean Relative Error: {sum(rel_errors) / len(rel_errors):.2f}%")
        print(f"  Kendall's Tau: {tau:.4f} (p-value: {p_value:.2e})")

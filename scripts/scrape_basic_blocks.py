"""
Scrape x86 basic blocks from binaries for pretraining.
"""

import re
import subprocess
from pathlib import Path

import polars as pl
import typer
from tqdm import tqdm

BLOCK_TERMINATORS = frozenset(
    [
        "jmp",
        "jmpq",
        "je",
        "jne",
        "jz",
        "jnz",
        "jg",
        "jge",
        "jl",
        "jle",
        "ja",
        "jae",
        "jb",
        "jbe",
        "js",
        "jns",
        "jo",
        "jno",
        "jp",
        "jnp",
        "jpe",
        "jpo",
        "jcxz",
        "jecxz",
        "jrcxz",
        "loop",
        "loope",
        "loopne",
        "loopz",
        "loopnz",
        "call",
        "callq",
        "ret",
        "retq",
        "retf",
        "iret",
        "iretq",
        "syscall",
        "sysenter",
        "sysexit",
        "int",
        "int3",
        "into",
        "hlt",
        "ud2",
    ]
)

INSTRUCTION_RE = re.compile(
    r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s)+)\s+(\S+)(?:\s+(.*))?$",
    re.IGNORECASE,
)
LABEL_RE = re.compile(r"^[0-9a-f]+\s+<(.+)>:$", re.IGNORECASE)


def extract_hex_blocks(binary_path: Path, min_instructions: int = 2) -> list[str]:
    """Extract basic block hex strings from a binary."""
    try:
        result = subprocess.run(
            ["objdump", "-d", "-M", "att", str(binary_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []

    blocks = []
    current_hex = []
    n_instrs = 0

    def finish_block():
        nonlocal current_hex, n_instrs
        if n_instrs >= min_instructions:
            blocks.append("".join(current_hex))
        current_hex = []
        n_instrs = 0

    for line in result.stdout.splitlines():
        if LABEL_RE.match(line):
            finish_block()
            continue

        match = INSTRUCTION_RE.match(line)
        if not match:
            continue

        _, hex_bytes, mnemonic, _ = match.groups()
        current_hex.append(hex_bytes.replace(" ", "").strip())
        n_instrs += 1

        if mnemonic.lower() in BLOCK_TERMINATORS:
            finish_block()

    finish_block()
    return blocks


def main(
    directory: Path = typer.Argument(Path("/usr/bin"), help="Directory to scan for binaries"),
    output: Path = typer.Option(Path("data/basic_blocks.parquet"), help="Output parquet file"),
    min_instructions: int = typer.Option(2, help="Minimum instructions per block"),
):
    binaries = [p for p in directory.iterdir() if p.is_file()]
    print(f"Found {len(binaries)} files in {directory}")

    all_hex = []
    failed = 0

    for path in tqdm(binaries, desc="Scraping"):
        blocks = extract_hex_blocks(path, min_instructions)
        if blocks:
            all_hex.extend(blocks)
        else:
            failed += 1

    print(f"Extracted {len(all_hex):,} blocks ({failed} files failed/skipped)")

    output.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({"hex": all_hex})
    df.write_parquet(output)
    print(f"Saved to {output}")


if __name__ == "__main__":
    typer.run(main)

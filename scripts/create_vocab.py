import glob
import pickle
import re

import pandas as pd


def generate_vocab_pickle(file_pattern, output_file="vocab.pkl"):
    # 1. Define Special & Structural Tokens (HARDCODED)
    # These effectively "reserve" IDs 0 to ~30
    vocab = {
        # Standard Special
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "<SEP>": 4,
        # Memory Structure (Required by your Parser)
        "MEM_OPEN": 5,
        "MEM_CLOSE": 6,
        "MEM_SEP": 7,
        "SCALE_1": 8,
        "SCALE_2": 9,
        "SCALE_4": 10,
        "SCALE_8": 11,
        # Segment Overrides
        "SEG_FS": 12,
        "SEG_GS": 13,
        # Value Buckets (Required by Parser)
        "IMM_ZERO": 14,
        "IMM_ONE": 15,
        "IMM_S8": 16,
        "IMM_16": 17,
        "IMM_32": 18,
        "IMM_64": 19,
        "DISP_ZERO": 20,
        "DISP_8": 21,
        "DISP_32": 22,
    }

    next_id = 23  # Start auto-assigning after the hardcoded ones
    unique_tokens = set()

    files = glob.glob(file_pattern)
    print(f"Processing {len(files)} files...")

    # Regex: Matches Opcodes and Registers (Same as before)
    token_pattern = re.compile(r"\b(%[a-z0-9]+|[a-z][a-z0-9]*)\b")

    for file_path in files:
        try:
            df = pd.read_parquet(file_path)
            if "instructions" in df.columns:
                for instruction in df["instructions"]:
                    if isinstance(instruction, str):
                        # Lowercase to ensure 'Mov' and 'mov' map to same ID
                        matches = token_pattern.findall(instruction.lower())
                        unique_tokens.update(matches)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Sort and assign IDs
    sorted_tokens = sorted(list(unique_tokens))
    for token in sorted_tokens:
        # Avoid overwriting hardcoded tokens if they somehow appear in text
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1

    # --- SAVE AS PICKLE ---
    with open(output_file, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Success! Vocabulary ({len(vocab)} tokens) saved to '{output_file}'")


if __name__ == "__main__":
    user_file_path = "/Users/teddydong/Documents/Winter 2026/CS 172b/assembly parquet/*.parquet"
    generate_vocab_pickle(user_file_path, "vocab.pkl")

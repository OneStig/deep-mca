import pickle
import re

from datasets import load_dataset


def generate_vocab_pickle(dataset_names, output_file="vocab.pkl"):
    # 1. Define Special & Structural Tokens (Same as before)
    vocab = {
        # Standard Special
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "<SEP>": 4,
        # Memory Structure
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
        # Value Buckets
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

    next_id = 23
    unique_tokens = set()

    # Regex: Matches Opcodes and Registers
    token_pattern = re.compile(r"\b(%[a-z0-9]+|[a-z][a-z0-9]*)\b")

    print(f"Processing {len(dataset_names)} datasets...")

    for ds_name in dataset_names:
        print(f"  Streaming {ds_name}...")
        try:
            ds = load_dataset(ds_name, split="train")

            for row in ds:
                instruction = row.get("instructions")
                if isinstance(instruction, str):
                    # Lowercase to ensure 'Mov' and 'mov' map to same ID
                    matches = token_pattern.findall(instruction.lower())
                    unique_tokens.update(matches)

        except Exception as e:
            print(f"Error reading {ds_name}: {e}")

    # Sort and assign IDs
    print("Sorting and assigning IDs...")
    sorted_tokens = sorted(list(unique_tokens))
    for token in sorted_tokens:
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1

    # Save as pickle
    with open(output_file, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Success! Vocabulary ({len(vocab)} tokens) saved to '{output_file}'")


if __name__ == "__main__":
    DATASETS = [
        "Arcticbun/hsw_x86",
        "Arcticbun/ivb_x86",
        "Arcticbun/skl_x86",
    ]

    generate_vocab_pickle(DATASETS, "data/vocab.pkl")

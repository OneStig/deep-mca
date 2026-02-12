import pandas as pd
import pickle
import re
import glob
import os
from tqdm import tqdm  # pip install tqdm

from datasets import load_dataset
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
VOCAB_FILE = '../../data/vocab.pkl'
INPUT_DATASETS = [
    "Arcticbun/hsw_x86",
    "Arcticbun/ivb_x86",
    "Arcticbun/skl_x86",
]
OUTPUT_SUFFIX = "_tokenized.parquet"

# ==========================================
# 2. LOAD VOCABULARY
# ==========================================
def load_vocab(pkl_path):
    print(f"Loading vocabulary from {pkl_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Loaded {len(vocab)} tokens.")
        
        # Sanity Check: Ensure structural tokens exist
        required = ["MEM_OPEN", "IMM_32", "SEG_FS"]
        missing = [t for t in required if t not in vocab]
        if missing:
            print(f"WARNING: Your vocab is missing structural tokens: {missing}")
            print("The parser will fail or produce <UNK> for these.")
            
        return vocab
    except FileNotFoundError:
        print(f"ERROR: {pkl_path} not found.")
        exit()

# Load Vocab Global
VOCAB = load_vocab(VOCAB_FILE)
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()} # For debugging

# ==========================================
# 3. PARSING LOGIC
# ==========================================
def bucket_immediate(val_str):
    """Maps raw numbers like '$0x10' to buckets like 'IMM_S8'"""
    try:
        clean = val_str.replace('$', '')
        val = int(clean, 0)
        if val == 0: return "IMM_ZERO"
        if val == 1: return "IMM_ONE"
        if -128 <= val <= 127: return "IMM_S8"
        if -32768 <= val <= 32767: return "IMM_16"
        if -2147483648 <= val <= 2147483647: return "IMM_32"
        return "IMM_64"
    except: return "IMM_32"

def bucket_disp(val):
    """Maps memory offsets like -8 to buckets"""
    if val == 0: return "DISP_ZERO"
    if -128 <= val <= 127: return "DISP_8"
    return "DISP_32"

def parse_block_to_ids(block_str):
    """
    Parses a multi-line assembly string into a list of Token IDs.
    Structure: [BOS, token, token, ..., EOS]
    """
    if not isinstance(block_str, str): return []
    
    # 1. Start with BOS (if it exists in vocab)
    token_ids = []
    if "<BOS>" in VOCAB:
        token_ids.append(VOCAB["<BOS>"])
    
    # Process line by line
    for line in block_str.strip().split('\n'):
        line = line.strip().split('#')[0].strip().lower()
        if not line: continue
        
        parts = line.split(maxsplit=1)
        opcode = parts[0]
        operands = parts[1] if len(parts) > 1 else ""
        
        # Add Opcode
        if opcode in VOCAB: token_ids.append(VOCAB[opcode])
        else: token_ids.append(VOCAB.get("<UNK>", 1))
        
        if not operands: continue
        
        # Parse Operands
        pattern = r'(%[a-z0-9]+:)|(\$?-?0x[0-9a-f]+|\$?-?\d+)|(%[a-z0-9]+)|([(),])'
        
        in_memory = False
        for m in re.finditer(pattern, operands):
            t = m.group(0)
            
            # Helper to add token safely
            def add(k):
                if k in VOCAB: token_ids.append(VOCAB[k])
                elif k.upper() in VOCAB: token_ids.append(VOCAB[k.upper()])
                else: token_ids.append(VOCAB.get("<UNK>", 1))

            if t == ',':
                if in_memory: add("MEM_SEP")
            elif t == '(':
                add("MEM_OPEN"); in_memory = True
            elif t == ')':
                add("MEM_CLOSE"); in_memory = False
            elif t.endswith(':'):
                seg = t[:-1]
                if seg == "%fs": add("SEG_FS")
                elif seg == "%gs": add("SEG_GS")
            elif t.startswith('$'):
                add(bucket_immediate(t))
            elif t.startswith('%'):
                add(t[1:]) 
            elif t[0].isdigit() or t[0] == '-':
                val = int(t, 0)
                if in_memory and val in [1, 2, 4, 8]:
                    add(f"SCALE_{val}")
                else:
                    add(bucket_disp(val))
                    
    # 2. End with EOS (if it exists in vocab)
    if "<EOS>" in VOCAB: 
        token_ids.append(VOCAB["<EOS>"])
    
    return token_ids

# ==========================================
# 4. PROCESSING FROM HUGGING FACE
# ==========================================
def process_hf_datasets(
    dataset_names: list[str],
    text_column: str,
    split: str = "train",
    output_dir: str = "../../data/tokenized_out",
    sample_index: int = 5,
):
    """
    Process each HF dataset like the original 'file loop':
      - load dataset
      - verify column
      - tokenize instructions -> token_ids
      - seq_len
      - print sample row
      - write parquet output for that dataset
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for name in dataset_names:
        safe_name = name.replace("/", "__")
        print(f"\n--- Processing HF dataset: {name} (split={split}) ---")

        ds = load_dataset(name, split=split)

        # Column verification
        if hasattr(ds, "features") and text_column not in ds.features:
            print(f"Skipping {name}: Missing '{text_column}' column. Available: {list(ds.features.keys())}")
            continue

        df = ds.to_pandas()  # may be huge
        if text_column not in df.columns:
            print(f"Skipping {name}: Missing '{text_column}' column.")
            continue

        print(f"Rows: {len(df):,}")

        tqdm.pandas(desc=f"Tokenizing {safe_name}")
        df["token_ids"] = df[text_column].progress_apply(
            lambda x: parse_block_to_ids(x) if isinstance(x, str) else []
        )
        df["seq_len"] = df["token_ids"].apply(len)

        # Sample
        if len(df) > sample_index:
            sample = df.iloc[sample_index]
            print(f"\n>>> Verification Sample (Row {sample_index}):")
            print(f"[Raw Text]:\n{str(sample[text_column]).strip()}")
            print(f"[Token IDs]: {sample['token_ids']}")
            decoded = [ID_TO_TOKEN.get(i, "?") for i in sample["token_ids"]]
            print(f"[Decoded]:   {decoded}")
            print(f"[Length]:    {sample['seq_len']}")

        out_path = out_root / f"{safe_name}_{split}{OUTPUT_SUFFIX}.parquet"
        df.to_parquet(out_path, index=False)
        print(f">>> Saved to: {out_path}")
        continue


if __name__ == "__main__":
    process_hf_datasets(
        dataset_names=INPUT_DATASETS,
        text_column="instructions",
        split="train",
    )
 
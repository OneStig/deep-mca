import json
from datasets import load_dataset, concatenate_datasets
from deep_mca.tokenizer import TextAssemblyTokenizer

ds1 = load_dataset("Arcticbun/hsw_x86")
ds2 = load_dataset("Arcticbun/ivb_x86")
ds3 = load_dataset("Arcticbun/skl_x86")
ds = concatenate_datasets([ds1["train"], ds2["train"], ds3["train"]])
tokenizer = TextAssemblyTokenizer()

STATE_FILE = "state.json"

# load last index
try:
    with open(STATE_FILE) as f:
        start_idx = json.load(f)["idx"]
except FileNotFoundError:
    start_idx = 0

for i in range(start_idx, len(ds)):
    asm_block = ds[i]["instructions"]
    asm_instr_list = asm_block.strip().split("\n")
    tokenized = tokenizer.tokenize_block(asm_instr_list)

    if i % 1000 == 0:
        with open(STATE_FILE, "w") as f:
            json.dump({"idx": i}, f)

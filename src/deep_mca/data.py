import csv
import math
import re
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# TODO: Replace this
# Using a very naive tokenization scheme for now so we can train for now.
# PAD is just to make tensor rectangular, always start with BOS and end with EOS.
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
BYTE_OFFSET = 3
VOCAB_SIZE = 256 + BYTE_OFFSET


def hex_to_tokens(hex_str: str) -> list[int]:
    """Convert a hex string to a list of token IDs with BOS/EOS."""
    # remove once we have proper tokenization
    byte_vals = bytes.fromhex(hex_str)
    return [BOS_ID] + [b + BYTE_OFFSET for b in byte_vals] + [EOS_ID]

class TextAssemblyTokenizer:
    def __init__(self):
        # Vocabularies
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.reg_vocab = {"<NONE>": 0, "<UNK>": 1}

        # Regex patterns for AT&T syntax
        self.re_reg = re.compile(r"%(\w+)")  # Matches %eax, %r15d
        self.re_imm = re.compile(r"\$([-0-9xA-Fa-f]+)")  # Matches $1, $0xFF
        self.re_mem = re.compile(r"(-?0x[0-9a-f]+|-?\d+)?\((%?\w+)(?:,\s*(%?\w+)(?:,\s*(\d+))?)?\)")
        # Matches -60(%rbp) or (%rax, %rcx, 4)

    def _get_id(self, key, vocab):
        if key not in vocab:
            vocab[key] = len(vocab)
        return vocab[key]

    def normalize_value(self, val_str):
        """Converts hex/decimal strings to log-scaled floats."""
        try:
            val = int(val_str, 0)  # Handles '0x10' and '16'
        except (ValueError, TypeError):
            return 0.0

        if val == 0:
            return 0.0
        sign = 1 if val > 0 else -1
        return sign * math.log2(abs(val) + 1)

    def tokenize_block(self, instr_list):
        """
        Args:
            instr_list: List of strings e.g. ['movl %eax, -60(%rbp)', ...]
        Returns:
            List of structured dictionaries for the Mamba Dataset
        """
        tokenized_block = []

        for line in instr_list:
            # 1. Clean and split mnemonic
            parts = line.strip().split()
            if not parts:
                continue

            mnemonic = parts[0]
            operands_str = "".join(parts[1:])  # Rejoin rest to handle spaces

            instr_data = {"mne_id": self._get_id(mnemonic, self.vocab), "regs": [], "numerical": []}

            # 2. Extract Registers (e.g., %eax)
            # We find ALL registers in the line (source, dest, index, base)
            regs = self.re_reg.findall(operands_str)
            for r in regs:
                instr_data["regs"].append(self._get_id(r, self.reg_vocab))

            # 3. Extract Immediates (e.g., $1)
            imms = self.re_imm.findall(operands_str)
            for imm in imms:
                instr_data["numerical"].append(self.normalize_value(imm))

            # 4. Extract Memory Displacements (e.g., -60 from -60(%rbp))
            # The regex finds the number before the parenthesis
            mem_refs = self.re_mem.findall(operands_str)
            for mem in mem_refs:
                disp_str = mem[0]  # The first group is the displacement
                if disp_str:
                    instr_data["numerical"].append(self.normalize_value(disp_str))

            tokenized_block.append(instr_data)

        return tokenized_block
    


class TextAssemblyLMTokenizer(TextAssemblyTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.vocab["<BOS>"] = len(self.vocab)
        self.vocab["<EOS>"] = len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.vocab["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<EOS>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _bucket_numeric(self, value: float) -> int:
        bucket = int(round(value))
        return max(-64, min(64, bucket))

    def encode_block(self, instr_list: list[str]) -> list[int]:
        tokenized_block = self.tokenize_block(instr_list)
        tokens: list[int] = [self.bos_id]
        for instr in tokenized_block:
            tokens.append(self._get_id(f"mne:{instr['mne_id']}", self.vocab))
            for reg in instr["regs"]:
                tokens.append(self._get_id(f"reg:{reg}", self.vocab))
            for num in instr["numerical"]:
                bucket = self._bucket_numeric(float(num))
                tokens.append(self._get_id(f"num:{bucket}", self.vocab))
        tokens.append(self.eos_id)
        return tokens



class BHiveDataset(Dataset):
    """Dataset for bhive throughput data with naive tokenization."""

    def __init__(
        self,
        csv_path: str | Path,
        max_seq_len: int = 512,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        log_targets: bool = True,
    ):
        csv_path = Path(csv_path)
        samples: list[tuple[str, float]] = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                hex_str, throughput = row[0], float(row[1])
                if not hex_str:
                    continue
                # +2 for BOS/EOS
                if len(hex_str) // 2 + 2 > max_seq_len:
                    continue
                samples.append((hex_str, throughput))

        # Deterministic shuffle and split
        # TODO: Later we should just use canonical split? @henry
        gen = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(samples), generator=gen).tolist()
        split_idx = int(len(indices) * train_ratio)

        if split == "train":
            selected = indices[:split_idx]
        else:
            selected = indices[split_idx:]

        self.items: list[tuple[list[int], float]] = []
        for i in selected:
            hex_str, throughput = samples[i]
            tokens = hex_to_tokens(hex_str)
            target = math.log(throughput) if log_targets else throughput
            self.items.append((tokens, target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, float]:
        tokens, target = self.items[idx]
        return torch.tensor(tokens, dtype=torch.long), len(tokens), target


def collate_fn(
    batch: list[tuple[torch.Tensor, int, float]],
) -> dict[str, torch.Tensor]:
    """Pad sequences and return input_ids, lengths, and targets."""
    token_seqs, lengths, targets = zip(*batch, strict=True)
    input_ids = pad_sequence(list(token_seqs), batch_first=True, padding_value=PAD_ID)
    return {
        "input_ids": input_ids,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.float32),
    }

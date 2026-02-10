import json
import math
import re
from pathlib import Path


class TextAssemblyTokenizer:
    PATH = "./vocab.json"

    def __init__(self):
        # Vocabularies
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }
        self.reg_vocab = {"<NONE>": 0, "<UNK>": 1}
        if Path(TextAssemblyTokenizer.PATH).exists():
            self._load_vocab()

        self.vocab_length = len(self.vocab)
        self.reg_vocab_length = len(self.vocab)

        # Regex patterns for AT&T syntax
        self.re_reg = re.compile(r"%(\w+)")  # Matches %eax, %r15d
        self.re_imm = re.compile(r"\$([-0-9xA-Fa-f]+)")  # Matches $1, $0xFF
        self.re_mem = re.compile(r"(-?0x[0-9a-f]+|-?\d+)?\((%?\w+)(?:,\s*(%?\w+)(?:,\s*(\d+))?)?\)")
        # Matches -60(%rbp) or (%rax, %rcx, 4)

    def _get_id(self, key, vocab):
        if key not in vocab:
            vocab[key] = len(vocab)
        return vocab[key]

    def _save_vocab(self):
        """Save vocabs to JSON file."""
        if self.vocab_length == len(self.vocab) and self.reg_vocab_length == len(self.reg_vocab):
            return

        with open(TextAssemblyTokenizer.PATH, "w") as f:
            json.dump({"vocab": self.vocab, "reg_vocab": self.reg_vocab}, f)

        self.vocab_length = len(self.vocab)
        self.reg_vocab_length = len(self.reg_vocab)

    def _load_vocab(self):
        """Load vocabs from JSON file."""
        with open(TextAssemblyTokenizer.PATH) as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.reg_vocab = data["reg_vocab"]

    def _get_id(self, key, vocab):
        if key not in vocab:
            if getattr(self, "_frozen", False):
                return vocab["<UNK>"]  # Use unknown token
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

        self._save_vocab()
        return tokenized_block

import csv
import math
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from deep_mca.tokenizer import TextAssemblyTokenizer
from deep_mca.utils import disassemble_hex

# PAD is just to make tensor rectangular, always start with BOS and end with EOS.
tokenizer = TextAssemblyTokenizer()
PAD_ID = tokenizer.vocab["<PAD>"]
BOS_ID = tokenizer.vocab["<BOS>"]
EOS_ID = tokenizer.vocab["<EOS>"]
VOCAB_SIZE = len(tokenizer.vocab) + len(tokenizer.reg_vocab)


def hex_to_tokens(hex_str: str):
    asm_lines = disassemble_hex(hex_str)
    token_dicts = tokenizer.tokenize_block(asm_lines)
    # Flatten: [mne_id_1, reg_id_1a, reg_id_1b, mne_id_2, ...]
    flat = []
    for d in token_dicts:
        flat.append(d["mne_id"])
        flat.extend(d["regs"])
        flat.extend(d["numerical"])
    return [BOS_ID] + flat + [EOS_ID]


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


class CollateLM:
    """
    Pad sequences and return input-ids, attention_mask, and labels (with pads masked as -100).
    Drops sequences that failed disassembly.
    """

    def __init__(self, pad_id: int):
        self.pad_id = int(pad_id)

    def __call__(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = [x for x in batch if x.numel() > 0]
        if len(batch) == 0:
            dummy = torch.tensor([[self.pad_id]], dtype=torch.long)
            return {
                "input_ids": dummy,
                "attention_mask": (dummy != self.pad_id).long(),
                "labels": torch.full_like(dummy, -100),
            }

        input_ids = pad_sequence(batch, batch_first=True, padding_value=self.pad_id)
        attention_mask = (input_ids != self.pad_id).long()

        labels = input_ids.clone()
        labels[input_ids == self.pad_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

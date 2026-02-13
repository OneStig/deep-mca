import math

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from deep_mca.tokenizer import Tokenizer


class BHiveDataset(Dataset):
    """Dataset for bhive throughput data, pre-disassembled."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Tokenizer,
        max_seq_len: int = 512,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        log_targets: bool = True,
        text_column: str = "instructions",
        target_column: str = "cycles",
    ):
        ds = load_dataset(dataset_name, split="train")
        samples: list[tuple[str, float]] = []
        for row in ds:
            text = row.get(text_column)
            target = row.get(target_column)
            if not text or not isinstance(text, str) or target is None:
                continue
            samples.append((text, float(target)))

        gen = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(samples), generator=gen).tolist()
        split_idx = int(len(indices) * train_ratio)

        if split == "train":
            selected = indices[:split_idx]
        else:
            selected = indices[split_idx:]

        self.items: list[tuple[list[int], float]] = []
        for i in selected:
            text, throughput = samples[i]
            tokens = tokenizer.parse_block_to_ids(text)
            if len(tokens) == 0 or len(tokens) > max_seq_len:
                continue
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
    input_ids = pad_sequence(list(token_seqs), batch_first=True, padding_value=0)
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

"""
Next-token-prediction dataset over a single tokenized text corpus.

Loads the whole encoded corpus into memory as a 1D tensor of token ids
(fine for the character-level baseline) and yields fixed-length
(input, target) windows, where target is input shifted by one position.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset

from tokenizer import CharTokenizer


class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        if data.numel() <= block_size:
            raise ValueError(
                f"Corpus has {data.numel()} tokens but block_size={block_size}; "
                "need at least block_size + 1 tokens."
            )
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return self.data.numel() - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


def load_dataset(
    path: str, block_size: int, train_split: float = 0.9
) -> Tuple[TextDataset, TextDataset, CharTokenizer]:
    """Reads a text file, builds a tokenizer, and splits into train/val."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split_idx = int(len(ids) * train_split)
    train_ids, val_ids = ids[:split_idx], ids[split_idx:]

    train_ds = TextDataset(train_ids, block_size)
    val_ds = TextDataset(val_ids, block_size)
    return train_ds, val_ds, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check the dataset pipeline")
    parser.add_argument("--data", required=True, help="Path to a plain text file")
    parser.add_argument("--block-size", type=int, default=64)
    args = parser.parse_args()

    train_ds, val_ds, tok = load_dataset(args.data, args.block_size)
    x, y = train_ds[0]
    print(f"vocab_size={tok.vocab_size} train_len={len(train_ds)} val_len={len(val_ds)}")
    print(f"x={x.tolist()}")
    print(f"y={y.tolist()}")

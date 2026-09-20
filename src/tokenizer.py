"""
Minimal character-level tokenizer.

This is a deliberately simple baseline so the rest of the pipeline
(dataset -> model -> train -> generate) can be exercised end to end.
Swap this out for a BPE / SentencePiece tokenizer in a later phase
without changing anything downstream, as long as encode/decode and
vocab_size keep the same contract.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List


class CharTokenizer:
    """Encodes/decodes text using a fixed character vocabulary."""

    def __init__(self, chars: Iterable[str]):
        self.chars: List[str] = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(chars=set(text))

    def encode(self, text: str) -> List[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(
                f"Character {exc.args[0]!r} not in tokenizer vocabulary. "
                "Rebuild the tokenizer on text that contains it."
            ) from exc

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(chars=data["chars"])


if __name__ == "__main__":
    sample = "hello distributed gpt"
    tok = CharTokenizer.from_text(sample)
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample
    print(f"vocab_size={tok.vocab_size}")
    print(f"ids={ids}")
    print(f"decoded={tok.decode(ids)!r}")

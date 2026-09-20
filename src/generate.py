"""
Load a trained checkpoint and sample text from a prompt.
"""

from __future__ import annotations

import argparse
import os

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def load_checkpoint(checkpoint_path: str, tokenizer_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    tokenizer = CharTokenizer.load(tokenizer_path)

    model_cfg = GPTConfig(**ckpt["config"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text from a trained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to ckpt.pt")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to tokenizer.json (defaults next to checkpoint)",
    )
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or os.path.join(
        os.path.dirname(args.checkpoint), "tokenizer.json"
    )
    model, tokenizer = load_checkpoint(
        args.checkpoint, tokenizer_path, args.device
    )

    ids = torch.tensor(
        [tokenizer.encode(args.prompt)],
        dtype=torch.long,
        device=args.device,
    )
    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stream a JSONL file and write Megatron-style indexed token binaries.

Input schema: each line is JSON with a `text` field.
Outputs:
  <output_prefix>.bin  — concatenated token ids, uint16
  <output_prefix>.idx  — int64 index array of length N+1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def load_tokenizer(name: str = "gpt2") -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def stream_samples(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                yield text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="JSONL path")
    parser.add_argument("--output-prefix", type=str, required=True, help="output path without suffix")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Truncate long samples")
    args = parser.parse_args()

    input_path = Path(args.input)
    bin_out = Path(f"{args.output_prefix}.bin")
    idx_out = Path(f"{args.output_prefix}.idx")
    if not input_path.exists():
        raise SystemExit(f"Missing input: {input_path}")

    tokenizer = load_tokenizer(args.tokenizer_name)
    texts = list(stream_samples(input_path))
    n_samples = len(texts)
    print(f"Loaded {n_samples} samples from {input_path}")

    offsets = np.empty(n_samples + 1, dtype=np.int64)
    offsets[0] = 0
    chunks = []
    total_tokens = 0
    for text in tqdm(texts, desc="Tokenizing"):
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=args.max_tokens)["input_ids"]
        arr = np.asarray(ids, dtype=np.uint16)
        chunks.append(arr)
        total_tokens += arr.shape[0]
        offsets[len(chunks)] = total_tokens

    if not chunks:
        raise SystemExit("No tokens produced.")

    token_buf = np.concatenate(chunks, axis=0)
    token_buf = np.ascontiguousarray(token_buf)
    print(f"Total tokens: {token_buf.shape[0]}")

    bin_out.parent.mkdir(parents=True, exist_ok=True)
    token_buf.tofile(str(bin_out))
    offsets.tofile(str(idx_out))

    bin_size = bin_out.stat().st_size
    idx_size = idx_out.stat().st_size
    byte_per_token = token_buf.dtype.itemsize
    print(f"Wrote {bin_out}: {bin_size} bytes ({token_buf.shape[0]} tokens * {byte_per_token} bytes)")
    print(f"Wrote {idx_out}: {idx_size} bytes ({offsets.shape[0]} int64 entries)")
    print(f"Row count/samples: {n_samples}")
    print("Verification:")
    for i in [0, n_samples // 2, n_samples - 1]:
        start = int(offsets[i])
        end = int(offsets[i + 1])
        seg = token_buf[start:end]
        decoded = tokenizer.decode(seg.tolist(), skip_special_tokens=False)
        print(f"  sample {i}: bytes [{start}, {end}), tokens={seg.shape[0]}, preview={decoded[:120]!r}")


if __name__ == "__main__":
    main()

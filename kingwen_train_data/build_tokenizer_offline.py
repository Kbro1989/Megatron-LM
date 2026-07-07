#!/usr/bin/env python3
"""Rebuild tokenized .bin/.idx from JSONL without any provider download."""
from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model" / "local-tokenizer"
TOKENIZER_NAME = MODEL_DIR / "tokenizer.json"
TOKENIZER_CONFIG = MODEL_DIR / "tokenizer_config.json" 


def load_local_tokenizer() -> AutoTokenizer:
    if not TOKENIZER_NAME.exists() or not TOKENIZER_CONFIG.exists():
        raise SystemExit(
            f"Missing local tokenizer files: {TOKENIZER_NAME} / {TOKENIZER_CONFIG}"
        )
    return AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        use_fast=True,
    )


def rebuild(jsonl_path: Path, output_prefix: Path, max_tokens: int = 4096) -> None:
    tokenizer = load_local_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") if isinstance(obj, dict) else None
            if isinstance(text, str) and text.strip():
                texts.append(text)

    offsets = [0]
    chunks = []
    total = 0
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
        arr = [int(x) for x in ids]
        chunks.append(arr)
        total += len(arr)
        offsets.append(total)

    flat = [tok for arr in chunks for tok in arr]
    import struct
    bin_path = Path(f"{output_prefix}.bin")
    idx_path = Path(f"{output_prefix}.idx")
    bin_path.write_bytes(struct.pack(f"<{len(flat)}H", *flat))
    idx_path.write_bytes(
        b"".join(int(x).to_bytes(8, "little", signed=True) for x in offsets)
    )
    print(f"rebuild {jsonl_path}: samples={len(texts)}, tokens={total}")
    print(f"  {bin_path}: {bin_path.stat().st_size} bytes")
    print(f"  {idx_path}: {idx_path.stat().st_size} bytes")


def main() -> None:
    train = ROOT / "combined_pretrain_train.jsonl"
    val = ROOT / "combined_pretrain_val.jsonl"
    if not train.exists() or not val.exists():
        raise SystemExit("Missing combined JSONL files.")
    rebuild(train, ROOT / "combined_pretrain_train")
    rebuild(val, ROOT / "combined_pretrain_val")


if __name__ == "__main__":
    main()

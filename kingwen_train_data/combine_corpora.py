#!/usr/bin/env python3
"""Combine King Wen corpus + life memory corpus into one pretrain JSONL."""
from __future__ import annotations
from pathlib import Path

KINGWEN = Path("kingwen_train_data/kingwen_pretrain.jsonl")
LIFE_TRAIN = Path("kingwen_train_data/life_corpus_train.jsonl")
LIFE_VAL = Path("kingwen_train_data/life_corpus_val.jsonl")
OUT_TRAIN = Path("kingwen_train_data/combined_pretrain_train.jsonl")
OUT_VAL = Path("kingwen_train_data/combined_pretrain_val.jsonl")

def read_lines(p: Path):
    if not p.exists():
        return []
    return [line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

train_lines = read_lines(KINGWEN) + read_lines(LIFE_TRAIN)
val_lines = read_lines(LIFE_VAL)

OUT_TRAIN.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
OUT_VAL.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
print(f"train={len(train_lines)} ({OUT_TRAIN.stat().st_size} bytes)")
print(f"val={len(val_lines)} ({OUT_VAL.stat().st_size} bytes)")

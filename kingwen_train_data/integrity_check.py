#!/usr/bin/env python3
"""End-to-end integrity check for Jarvis-native King Wen life corpus."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path("kingwen_train_data")
MODEL = ROOT / "model" / "jarvis-native-kingwen-life"


def check_jsonl(path: Path, expected_domains: set[str]):
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    counts: dict[str, int] = {}
    bad = 0
    for line in lines:
        try:
            obj = json.loads(line)
            text = obj.get("text", "")
            if text.startswith("[") and "]" in text:
                domain = text.split("]")[0].strip("[")
                counts[domain] = counts.get(domain, 0) + 1
        except json.JSONDecodeError:
            bad += 1
    print(f"{path}: samples={len(lines)} bad_json={bad} domains={counts}")
    missing = expected_domains - set(counts.keys())
    if missing:
        print(f"  MISSING domains: {missing}")
    return len(lines), bad, counts


def check_indexed(bin_path: Path, idx_path: Path, expected_samples: int | None = None):
    if not bin_path.exists() or not idx_path.exists():
        print(f"MISSING {bin_path} or {idx_path}")
        return 0, 0, 0
    offsets = np.memmap(idx_path, dtype=np.int64, mode="r")
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    total_tokens = int(tokens.shape[0])
    samples = int(offsets.shape[0]) - 1
    print(f"{bin_path.name}: samples={samples} total_tokens={total_tokens}")
    if expected_samples and samples != expected_samples:
        print(f"  WARNING: expected {expected_samples} samples, got {samples}")
    return samples, total_tokens, offsets.shape[0]


def check_labels(path: Path, expected_samples: int | None = None):
    if not path.exists():
        print(f"MISSING labels {path}")
        return {}
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    counts = {"hexagram_id_set": 0, "voiceWeight_set": 0, "phase_temporal_set": 0, "trajectory_set": 0}
    for line in lines:
        try:
            obj = json.loads(line)
            labels = obj.get("labels", {})
            if labels.get("hexagram_id") is not None:
                counts["hexagram_id_set"] += 1
            if labels.get("voiceWeight") is not None:
                counts["voiceWeight_set"] += 1
            if labels.get("phase_temporal") is not None:
                counts["phase_temporal_set"] += 1
            if labels.get("trajectory") is not None:
                counts["trajectory_set"] += 1
        except json.JSONDecodeError:
            pass
    print(f"{path}: label_counts={counts}")
    return counts


def check_tokenizer(model_dir: Path):
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    probe = [
        "<|kingwen_oracle|>",
        "[KING_WEN_ORACLE]",
        "<|voiceWeight|>",
        "<|past|>",
        "<|future|>",
        "<|chaos|>",
        "<|whimsy|>",
        "<|darkTone|>",
        "<|coherence|>",
    ]
    print(f"tokenizer vocab_size={tok.vocab_size}")
    for t in probe:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        print(f"  {t!r} -> {ids}")
    return tok.vocab_size


def main():
    print("=== INTEGRITY CHECK ===")
    train_samples, bad_train, domains = check_jsonl(
        ROOT / "combined_pretrain_train.jsonl",
        {"KING_WEN_ORACLE", "SOVEREIGN_PIPELINE_SCENE", "HERMES", "OPENJARVIS", "POG2", "GEMINI", "CLAUDE"},
    )
    val_samples, bad_val, val_domains = check_jsonl(
        ROOT / "combined_pretrain_val.jsonl",
        {"HERMES", "KING_WEN_ORACLE", "SOVEREIGN_PIPELINE_SCENE", "OPENJARVIS", "POG2", "GEMINI", "CLAUDE"},
    )

    tr_s, tr_tok, tr_idx = check_indexed(ROOT / "combined_pretrain_train.bin", ROOT / "combined_pretrain_train.idx", expected_samples=train_samples)
    va_s, va_tok, va_idx = check_indexed(ROOT / "combined_pretrain_val.bin", ROOT / "combined_pretrain_val.idx", expected_samples=val_samples)

    label_counts = check_labels(MODEL / "labels_train.jsonl", expected_samples=train_samples)
    val_label_counts = check_labels(MODEL / "labels_val.jsonl", expected_samples=val_samples)

    vocab_size = check_tokenizer(MODEL)

    print("\n=== SUMMARY ===")
    print(f"train_jsonl={train_samples} val_jsonl={val_samples}")
    print(f"train_indexed={tr_s} val_indexed={va_s}")
    print(f"train_tokens={tr_tok} val_tokens={va_tok}")
    print(f"labels_train={label_counts} labels_val={val_label_counts}")
    print(f"vocab_size={vocab_size}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a merged vocabulary manifest for Jarvis-native training.

Outputs:
  kingwen_train_data/model/jarvis-native-kingwen-life/vocab.json
  kingwen_train_data/model/jarvis-native-kingwen-life/merges.txt
  kingwen_train_data/model/jarvis-native-kingwen-life/tokenizer.json
"""
from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

OUT = Path("kingwen_train_data/model/jarvis-native-kingwen-life")
OUT.mkdir(parents=True, exist_ok=True)

BASE = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

added = [
    "<|kingwen_oracle|>",
    "<|sovereign_scene|>",
    "<|openjarvis|>",
    "<|hermes|>",
    "<|pog2|>",
    "<|endoftext|>",
    "<|pad|>",
    "[KING_WEN_ORACLE]",
    "[SOVEREIGN_PIPELINE_SCENE]",
    "[HERMES]",
    "[OPENJARVIS]",
    "[POG2]",
    "[GEMINI]",
    "[CLAUDE]",
    "<|hexagram|>",
    "<|phase_temporal|>",
    "<|phase_bits|>",
    "<|inject_reason|>",
    "<|yao|>",
    "<|vector|>",
    "<|voiceWeight|>",
    "<|coherence|>",
    "<|chaos|>",
    "<|whimsy|>",
    "<|darkTone|>",
    "<|past|>",
    "<|present|>",
    "<|future|>",
    "<|still|>",
    "<|converging|>",
    "<|diverging|>",
    "<|cycling|>",
    "<|porosity|>",
    "<|primary_pool|>",
    "<|secondary_pool|>",
    "<|session_id|>",
    "<|timestamp|>",
    "<|model|>",
    "<|engine|>",
    "<|agent|>",
    "<|outcome|>",
    "<|status|>",
    "<|trace_id|>",
    "<|entity_id|>",
    "<|reinforcement|>",
    "<|weight|>",
]

tokenizer.add_tokens(added)
tokenizer.save_pretrained(OUT)

# Also emit a human-readable manifest
manifest = {
    "base_tokenizer": BASE,
    "vocab_size": int(tokenizer.vocab_size),
    "added_tokens_count": len(added),
    "added_tokens": added,
}
(OUT / "vocab_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"Wrote merged tokenizer to {OUT}")
print(f"vocab_size={tokenizer.vocab_size}")
print(f"added_tokens={len(added)}")

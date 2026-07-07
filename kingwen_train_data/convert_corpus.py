#!/usr/bin/env python3
"""Convert King Wen 512-state expansion into Megatron-compatible JSONL training corpus."""
from pathlib import Path
import json

INPUT = Path("C:/Users/krist/Desktop/KING-WEN-I-CHING-IMMUTABLE-TABLES/collapse_full_128_output.json")
OUTPUT = Path("C:/Users/krist/Desktop/Megatron-LM-review/kingwen_train_data/corpus.jsonl")

if not INPUT.exists():
    raise SystemExit(f"Missing input: {INPUT}")

data = json.loads(INPUT.read_text(encoding="utf-8"))
resolved = data.get("resolved", [])
expanded = data.get("expanded", [])

# Build expand lookup by hexagram_id
expanded_by_hex = {e["hexagram_id"]: e for e in expanded if "hexagram_id" in e}

lines = []
for state in resolved:
    hex_id = state.get("hexagram_id")
    phase_temporal = state.get("phase_temporal", "")
    phase_bits = state.get("phase_bits")
    inject = state.get("inject_site", {})
    exp_vec = state.get("expanded_vector", {})
    res_vec = state.get("resolved_vector", {})
    checklist = state.get("checklist", [])

    # Build yao summary from checklist directions
    yao_parts = []
    for item in checklist[:6]:
        axis = item.get("axis", "")
        direction = item.get("direction", "")
        expected = item.get("expected", "")
        yao_parts.append(f"{axis} {direction}: {expected}")

    hex_name = ""
    upper = ""
    lower = ""
    category = ""
    if hex_id in expanded_by_hex:
        sym = expanded_by_hex[hex_id].get("hexagram_symbols", {})
        hex_name = sym.get("name", "")
        upper = sym.get("upper_trigram", "")
        lower = sym.get("lower_trigram", "")
        category = sym.get("category", "")

    primary = inject.get("primary_pool", "")
    secondary = inject.get("secondary_pool", "")
    porosity = inject.get("porosity", "")
    reason = inject.get("reason", "")
    porosity_window = inject.get("porosity_window", [])

    # Voice vector prose
    def vec_str(v):
        return ", ".join(f"{k}={v.get(k,0):.4f}" for k in ["voiceWeight","coherence","chaos","whimsy","darkTone"])

    # Build deterministic sample text
    text = (
        f"King Wen Oracle state record — hexagram {hex_id} {hex_name}, phase {phase_bits}, temporal {phase_temporal}.\n"
        f"Trigrams {upper} over {lower}, category {category}.\n"
        f"Inject site: primary {primary}, secondary {secondary}, porosity {porosity}, window {porosity_window}, reason {reason}.\n"
        f"Expanded vector: {vec_str(exp_vec)}.\n"
        f"Resolved vector: {vec_str(res_vec)}.\n"
        f"Line states: " + "; ".join(yao_parts) + ".\n"
        f"End of state record."
    )

    lines.append(json.dumps({"text": text}, ensure_ascii=False))

OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(lines)} samples to {OUTPUT}")
print(f"File size: {OUTPUT.stat().st_size} bytes")

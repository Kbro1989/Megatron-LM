#!/usr/bin/env python3
"""Build kingwen_pretrain.jsonl from King Wen resolved states."""
from __future__ import annotations

import json
from pathlib import Path

INPUT = Path("/mnt/c/Users/krist/Desktop/KING-WEN-I-CHING-IMMUTABLE-TABLES/collapse_full_128_output.json")
OUTPUT = Path("kingwen_train_data/kingwen_pretrain.jsonl")


def build_domain_a(state: dict, expanded_lookup: dict[int, dict]) -> str:
    """Deterministic oracle-state summary used as the base domain text."""
    hs = state["hexagram_symbols"]
    sym = expanded_lookup.get(hs["hexagram_id"], {})
    hex_name = hs.get("name") or sym.get("name", "")
    upper = hs.get("upper_trigram", "")
    lower = hs.get("lower_trigram", "")
    category = hs.get("category", "")
    action = hs.get("action", "")
    phase_temporal = state.get("phase_temporal", "")
    phase_polarity = state.get("phase_polarity", "")
    phase_description = state.get("phase_description", "")
    inject = state.get("inject_site", {})
    primary = inject.get("primary_pool", "")
    secondary = inject.get("secondary_pool", "")
    porosity = inject.get("porosity", "")
    porosity_label = inject.get("porosity_label", "")
    reason = inject.get("reason", "")
    expanded_vec = state.get("expanded_vector", {})
    resolved_vec = state.get("resolved_vector", {})
    checklist = state.get("checklist", [])

    def vec_line(v: dict) -> str:
        return (
            f"voiceWeight={v.get('voiceWeight',0):.4f}, "
            f"coherence={v.get('coherence',0):.4f}, "
            f"chaos={v.get('chaos',0):.4f}, "
            f"whimsy={v.get('whimsy',0):.4f}, "
            f"darkTone={v.get('darkTone',0):.4f}"
        )

    yao_parts = []
    for idx, item in enumerate(checklist[:6], 1):
        axis = item.get("axis", "")
        direction = item.get("direction", "")
        expected = item.get("expected", "")
        value = item.get("value", 0.0)
        yao_parts.append(
            f"yao {idx} {axis} {direction} | {expected} | value={value:.4f}"
        )

    return "\n".join([
        f"[KING_WEN_ORACLE]",
        f"hexagram={hs['hexagram_id']} {hex_name} | category={category} | action={action}",
        f"trigrams={upper} over {lower} | binary={hs.get('binary','')}",
        f"phase={state.get('phase_bits','')} {phase_temporal} | polarity={phase_polarity} | {phase_description}",
        f"inject_site primary={primary} secondary={secondary} porosity={porosity} {porosity_label}",
        f"inject_reason={reason}",
        f"expanded_vector {vec_line(expanded_vec)}",
        f"resolved_vector {vec_line(resolved_vec)}",
        "line_states=" + "; ".join(yao_parts),
        "[END_ORACLE]",
    ])


def build_domain_b(state: dict, expanded_lookup: dict[int, dict]) -> str:
    """Sovereign-pipeline-style scene record for a delivery-sequence narrative."""
    hs = state["hexagram_symbols"]
    sym = expanded_lookup.get(hs["hexagram_id"], {})
    hex_name = hs.get("name") or sym.get("name", "")
    category = hs.get("category", "")
    action = hs.get("action", "")
    phase_temporal = state.get("phase_temporal", "")
    phase_description = state.get("phase_description", "")
    inject = state.get("inject_site", {})
    primary = inject.get("primary_pool", "")
    secondary = inject.get("secondary_pool", "")
    porosity = inject.get("porosity", "")
    reason = inject.get("reason", "")
    expanded_vec = state.get("expanded_vector", {})
    resolved_vec = state.get("resolved_vector", {})
    checklist = state.get("checklist", [])
    sample_paths = state.get("sample_paths", [])

    primary_vec = sample_paths[0]["vector"] if len(sample_paths) > 0 else {}
    secondary_vec = sample_paths[1]["vector"] if len(sample_paths) > 1 else {}
    porous_mix_vec = sample_paths[2]["vector"] if len(sample_paths) > 2 else {}

    def vec_line(v: dict) -> str:
        return (
            f"voiceWeight={v.get('voiceWeight',0):.4f}, "
            f"coherence={v.get('coherence',0):.4f}, "
            f"chaos={v.get('chaos',0):.4f}, "
            f"whimsy={v.get('whimsy',0):.4f}, "
            f"darkTone={v.get('darkTone',0):.4f}"
        )

    def status_line(item: dict) -> str:
        return (
            f"{item.get('axis','')} {item.get('direction','')} -> "
            f"{item.get('expected','')} | status={item.get('status','')} | "
            f"value={item.get('value',0):.4f}"
        )

    oracle_lines = [status_line(item) for item in checklist[:6]]

    return "\n".join([
        "[SOVEREIGN_PIPELINE_SCENE]",
        f"Scene hexagram={hs['hexagram_id']} {hex_name} | category={category} | action={action}",
        f"Temporal phase={state.get('phase_bits','')} {phase_temporal} | status={phase_description}",
        f"Delivery primary={primary} | secondary={secondary} | porosity={porosity}",
        f"Inject reason={reason}",
        f"Primary vector {vec_line(primary_vec)}",
        f"Secondary vector {vec_line(secondary_vec)}",
        f"Porous mix vector {vec_line(porous_mix_vec)}",
        f"Base expanded {vec_line(expanded_vec)}",
        f"Resolved expanded {vec_line(resolved_vec)}",
        "Oracle checks=" + "; ".join(oracle_lines),
        f"Scene boundary={hs.get('binary','')} | upper={hs.get('upper_trigram','')} | lower={hs.get('lower_trigram','')}",
        "[END_SCENE]",
    ])


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Missing input: {INPUT}")

    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)

    states = data["resolved"]
    expanded = data["expanded"]
    expanded_by_hex = {int(item["hexagram_id"]): item for item in expanded}
    lines: list[str] = []

    for state in states:
        text_a = build_domain_a(state, expanded_by_hex)
        text_b = build_domain_b(state, expanded_by_hex)
        lines.append(
            json.dumps(
                {
                    "text": text_a,
                    "label_payload": {
                        "hexagram_id": state.get("hexagram_id"),
                        "phase_bits": state.get("phase_bits"),
                        "phase_temporal": state.get("phase_temporal"),
                        "phase_polarity": state.get("phase_polarity"),
                        "inject_site": state.get("inject_site", {}),
                        "expanded_vector": state.get("expanded_vector", {}),
                        "resolved_vector": state.get("resolved_vector", {}),
                        "checklist": state.get("checklist", [])[:6],
                        "source": "collapse_full_128",
                    },
                },
                ensure_ascii=False,
            )
        )
        lines.append(
            json.dumps(
                {
                    "text": text_b,
                    "label_payload": {
                        "hexagram_id": state.get("hexagram_id"),
                        "phase_bits": state.get("phase_bits"),
                        "phase_temporal": state.get("phase_temporal"),
                        "phase_polarity": state.get("phase_polarity"),
                        "inject_site": state.get("inject_site", {}),
                        "expanded_vector": state.get("expanded_vector", {}),
                        "resolved_vector": state.get("resolved_vector", {}),
                        "checklist": state.get("checklist", [])[:6],
                        "sample_paths": state.get("sample_paths", []),
                        "source": "collapse_full_128",
                        "domain": "SOVEREIGN_PIPELINE_SCENE",
                    },
                },
                ensure_ascii=False,
            )
        )

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} samples to {OUTPUT}")
    print(f"File size: {OUTPUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()

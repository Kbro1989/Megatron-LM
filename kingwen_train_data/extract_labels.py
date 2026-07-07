#!/usr/bin/env python3
"""Extract structured labels from combined corpus for native Jarvis training."""
from __future__ import annotations

import json
import re
from pathlib import Path

TRAIN = Path("kingwen_train_data/combined_pretrain_train.jsonl")
VAL = Path("kingwen_train_data/combined_pretrain_val.jsonl")
OUT_TRAIN = Path("kingwen_train_data/model/jarvis-native-kingwen-life/labels_train.jsonl")
OUT_VAL = Path("kingwen_train_data/model/jarvis-native-kingwen-life/labels_val.jsonl")

VOICE_RE = re.compile(r"voiceWeight=([0-9.]+), coherence=([0-9.]+), chaos=([0-9.]+), whimsy=([0-9.]+), darkTone=([0-9.]+)")
HEX_RE = re.compile(r"hexagram=(\d+)\s+([^|]+)")
PHASE_RE = re.compile(r"phase=(\d+)\s+(past|present|future)")
INJECT_RE = re.compile(r"inject_site primary=(\S+) secondary=(\S+) porosity=(\S+)")
REASON_RE = re.compile(r"inject_reason=(.+)")


def extract_labels(text: str) -> dict:
    labels = {
        "hexagram_id": None,
        "hexagram_name": None,
        "phase_bits": None,
        "phase_temporal": None,
        "primary_pool": None,
        "secondary_pool": None,
        "porosity": None,
        "reason": None,
        "voiceWeight": None,
        "coherence": None,
        "chaos": None,
        "whimsy": None,
        "darkTone": None,
        "domain": None,
        "source": None,
        "trajectory": None,
    }

    # Domain/source
    m = re.match(r"\[(\w+)\]\s+source=(.+?)(?:\n|$)", text)
    if m:
        labels["domain"] = m.group(1)
        labels["source"] = m.group(2)

    voice = VOICE_RE.search(text)
    if voice:
        labels["voiceWeight"] = float(voice.group(1))
        labels["coherence"] = float(voice.group(2))
        labels["chaos"] = float(voice.group(3))
        labels["whimsy"] = float(voice.group(4))
        labels["darkTone"] = float(voice.group(5))

    hex_m = HEX_RE.search(text)
    if hex_m:
        labels["hexagram_id"] = int(hex_m.group(1))
        labels["hexagram_name"] = hex_m.group(2).strip()

    phase = PHASE_RE.search(text)
    if phase:
        labels["phase_bits"] = int(phase.group(1))
        labels["phase_temporal"] = phase.group(2)

    inject = INJECT_RE.search(text)
    if inject:
        labels["primary_pool"] = inject.group(1)
        labels["secondary_pool"] = inject.group(2)
        try:
            labels["porosity"] = int(inject.group(3))
        except ValueError:
            labels["porosity"] = None

    reason = REASON_RE.search(text)
    if reason:
        labels["reason"] = reason.group(1).strip()

    # Derive trajectory from phase_bits/porosity heuristics
    if labels["phase_bits"] is not None and labels["porosity"] is not None:
        pb = labels["phase_bits"]
        por = labels["porosity"]
        if pb == 0 or pb == 7:
            labels["trajectory"] = "still"
        elif pb in (1, 2, 3):
            labels["trajectory"] = "converging"
        elif pb in (4, 5, 6):
            labels["trajectory"] = "diverging"
        else:
            labels["trajectory"] = "cycling"
    return labels


def process(input_path: Path, output_path: Path):
    count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = obj.get("label_payload") if isinstance(obj, dict) else None
            if isinstance(payload, dict) and payload.get("source") == "collapse_full_128":
                labels = {
                    "hexagram_id": payload.get("hexagram_id"),
                    "phase_bits": payload.get("phase_bits"),
                    "phase_temporal": payload.get("phase_temporal"),
                    "phase_polarity": payload.get("phase_polarity"),
                    "primary_pool": (payload.get("inject_site", {}) or {}).get("primary_pool"),
                    "secondary_pool": (payload.get("inject_site", {}) or {}).get("secondary_pool"),
                    "porosity": (payload.get("inject_site", {}) or {}).get("porosity"),
                    "reason": (payload.get("inject_site", {}) or {}).get("reason"),
                    "voiceWeight": (payload.get("resolved_vector", {}) or {}).get("voiceWeight"),
                    "coherence": (payload.get("resolved_vector", {}) or {}).get("coherence"),
                    "chaos": (payload.get("resolved_vector", {}) or {}).get("chaos"),
                    "whimsy": (payload.get("resolved_vector", {}) or {}).get("whimsy"),
                    "darkTone": (payload.get("resolved_vector", {}) or {}).get("darkTone"),
                    "domain": obj.get("domain") or payload.get("domain"),
                    "source": payload.get("source"),
                    "trajectory": None,
                }
                checklist = payload.get("checklist", []) or []
                if labels["phase_bits"] is not None and labels["porosity"] is not None:
                    pb = int(labels["phase_bits"])
                    por = int(labels["porosity"])
                    if pb in (0, 7):
                        labels["trajectory"] = "still"
                    elif pb in (1, 2, 3):
                        labels["trajectory"] = "converging"
                    elif pb in (4, 5, 6):
                        labels["trajectory"] = "diverging"
                    else:
                        labels["trajectory"] = "cycling"
            else:
                labels = extract_labels(obj.get("text", "") if isinstance(obj, dict) else "")
            fout.write(json.dumps({"labels": labels}, ensure_ascii=False) + "\n")
            count += 1
    print(f"{input_path.name} -> {output_path.name}: {count} labels ({output_path.stat().st_size} bytes)")


def main():
    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    process(TRAIN, OUT_TRAIN)
    process(VAL, OUT_VAL)


if __name__ == "__main__":
    main()

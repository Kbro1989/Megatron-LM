#!/usr/bin/env python3
"""Build additional domain corpora for Megatron from OpenJarvas and slider captures."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(r"C:/Users/krist")
OPENJARVIS = ROOT / ".openjarvis"
HERMES = ROOT / "AppData" / "Local" / "hermes"
OUT_DIR = Path("kingwen_train_data")
TRAIN_OUT = OUT_DIR / "combined_pretrain_train.jsonl"
VAL_OUT = OUT_DIR / "combined_pretrain_val.jsonl"
LABELS_TRAIN = OUT_DIR / "model/jarvis-native-kingwen-life/labels_train.jsonl"
LABELS_VAL = OUT_DIR / "model/jarvis-native-kingwen-life/labels_val.jsonl"


def _append_jsonl(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")


def _iter_openjarvis_traces() -> Iterable[dict]:
    db = OPENJARVIS / "traces.db"
    if not db.exists():
        return []
    con = sqlite3.connect(str(db), check_same_thread=False)
    cur = con.cursor()
    cur.execute("SELECT trace_id, query, agent, model, engine, outcome, feedback, started_at, metadata FROM traces LIMIT 4000")
    cols = [c[0] for c in cur.description]
    rows = []
    for row in cur.fetchall():
        obj = {cols[i]: row[i] for i in range(len(cols))}
        cur.execute("SELECT step_type, input, output, timestamp FROM trace_steps WHERE trace_id=? LIMIT 20", (obj.get("trace_id"),))
        obj["steps"] = [{"step_type": r[0], "input": r[1], "output": r[2], "timestamp": r[3]} for r in cur.fetchall()]
        rows.append(obj)
    con.close()
    return rows


def _iter_openjarvis_agent_messages() -> Iterable[dict]:
    db = OPENJARVIS / "agents.db"
    if not db.exists():
        return []
    con = sqlite3.connect(str(db), check_same_thread=False)
    cur = con.cursor()
    cur.execute("SELECT id, agent_id, direction, content, mode, status, created_at FROM agent_messages LIMIT 4000")
    cols = [c[0] for c in cur.description]
    rows = [{cols[i]: row[i] for i in range(len(cols))} for row in cur.fetchall()]
    con.close()
    return rows


def _iter_openjarvis_learning() -> Iterable[dict]:
    db = OPENJARVIS / "agents.db"
    if not db.exists():
        return []
    con = sqlite3.connect(str(db), check_same_thread=False)
    cur = con.cursor()
    cur.execute("SELECT id, agent_id, event_type, description, data, created_at FROM agent_learning_log LIMIT 4000")
    cols = [c[0] for c in cur.description]
    rows = [{cols[i]: row[i] for i in range(len(cols))} for row in cur.fetchall()]
    con.close()
    return rows


def _make_text_from_trace(obj: dict) -> str:
    steps = []
    for s in obj.get("steps", []):
        step = []
        if s.get("step_type"):
            step.append(f"step_type={s['step_type']}")
        for k in ["input", "output"]:
            v = s.get(k)
            if isinstance(v, str) and v.strip():
                step.append(f"{k}=" + v.strip()[:2000])
        if step:
            steps.append("\n".join(step))
    parts = [
        "[OPENJARVIS] source=traces.db",
        f"trace_id={obj.get('trace_id','')}",
        f"query={obj.get('query','')}",
        f"agent={obj.get('agent','')}",
        f"model={obj.get('model','')}",
        f"engine={obj.get('engine','')}",
        f"outcome={obj.get('outcome','')}",
        f"feedback={obj.get('feedback')}",
        f"started_at={obj.get('started_at')}",
    ]
    if obj.get("metadata"):
        parts.append("metadata=" + str(obj["metadata"])[:1000])
    if steps:
        parts.append("\n".join(steps))
    return "\n".join(parts)


def _make_text_from_agent(obj: dict) -> str:
    return "\n".join(
        [
            "[OPENJARVIS] source=agent_messages",
            f"id={obj.get('id','')}",
            f"agent_id={obj.get('agent_id','')}",
            f"direction={obj.get('direction','')}",
            f"mode={obj.get('mode','')}",
            f"status={obj.get('status','')}",
            f"created_at={obj.get('created_at')}",
            f"content={str(obj.get('content',''))[:4000]}",
        ]
    )


def _make_text_from_learning(obj: dict) -> str:
    return "\n".join(
        [
            "[OPENJARVIS] source=agent_learning_log",
            f"agent_id={obj.get('agent_id','')}",
            f"event_type={obj.get('event_type','')}",
            f"description={obj.get('description','')}",
            f"created_at={obj.get('created_at')}",
            f"data={str(obj.get('data',''))[:4000]}",
        ]
    )


def _label_from_trace(obj: dict) -> dict:
    return {
        "domain": "OPENJARVIS_INTERACTION",
        "source": "traces.db",
        "trace_id": obj.get("trace_id"),
        "agent": obj.get("agent"),
        "model": obj.get("model"),
        "engine": obj.get("engine"),
        "outcome": obj.get("outcome"),
        "feedback": obj.get("feedback"),
        "started_at": obj.get("started_at"),
        "step_count": len(obj.get("steps", [])),
        "trajectory": None,
        "hexagram_id": None,
        "phase_bits": None,
        "phase_temporal": None,
        "phase_polarity": None,
        "porosity": None,
        "pre_slider": None,
        "post_slider": None,
        "primary_pool": None,
        "secondary_pool": None,
        "reason": None,
        "voiceWeight": None,
        "coherence": None,
        "chaos": None,
        "whimsy": None,
        "darkTone": None,
    }


def _main() -> None:
    train_rows = []
    val_rows = []
    label_rows_train = []
    label_rows_val = []

    traces = list(_iter_openjarvis_traces())
    print("openjarvis traces:", len(traces))
    for obj in traces:
        train_rows.append(json.dumps({"text": _make_text_from_trace(obj)}, ensure_ascii=False))
        label_rows_train.append(json.dumps({"labels": _label_from_trace(obj)}, ensure_ascii=False))

    msgs = list(_iter_openjarvis_agent_messages())
    print("openjarvis agent_messages:", len(msgs))
    for obj in msgs:
        train_rows.append(json.dumps({"text": _make_text_from_agent(obj)}, ensure_ascii=False))
        label_rows_train.append(json.dumps({"labels": {"domain": "OPENJARVIS_INTERACTION", "source": "agent_messages"}}, ensure_ascii=False))

    learning = list(_iter_openjarvis_learning())
    print("openjarvis learning_log:", len(learning))
    for obj in learning:
        train_rows.append(json.dumps({"text": _make_text_from_learning(obj)}, ensure_ascii=False))
        label_rows_train.append(json.dumps({"labels": {"domain": "OPENJARVIS_INTERACTION", "source": "agent_learning_log"}}, ensure_ascii=False))

    if train_rows:
        _append_jsonl(TRAIN_OUT, train_rows)
        _append_jsonl(LABELS_TRAIN, label_rows_train)
    if val_rows:
        _append_jsonl(VAL_OUT, val_rows)
        _append_jsonl(LABELS_VAL, label_rows_val)

    print("appended train samples:", len(train_rows))
    print("appended val samples:", len(val_rows))


if __name__ == "__main__":
    _main()

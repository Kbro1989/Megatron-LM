#!/usr/bin/env python3
"""Consolidate local AI memory stores into a single Megatron-style JSONL."""
from __future__ import annotations

import json
import sqlite3
import random
from pathlib import Path
from typing import Iterable, Optional

random.seed(42)

ROOT = Path(r"C:/Users/krist")
OUT = Path("kingwen_train_data")
TRAIN_OUT = OUT / "life_corpus_train.jsonl"
VAL_OUT = OUT / "life_corpus_val.jsonl"


def iter_db_rows(db: Path, tables: Optional[Iterable[str]] = None, limit: int = 5000) -> Iterable[tuple[str, dict]]:
    if not db.exists():
        return
    try:
        con = sqlite3.connect(str(db), check_same_thread=False)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        all_tables = [r[0] for r in cur.fetchall()]
        targets = tables or all_tables
        for table in targets:
            if table not in all_tables:
                continue
            try:
                cur.execute(f"PRAGMA table_info({table})")
                cols = [r[1] for r in cur.fetchall()]
                if not cols:
                    continue
                text_col_hints = ["content", "text", "body", "description", "query", "result", "prompt", "system_prompt", "summary_memory", "findings", "metadata", "progress", "conversation_state", "tool_state", "payload", "state_json", "profile_json", "data", "finding", "model_config"]
                text_cols = [c for c in cols if any(h in c.lower() for h in text_col_hints)]
                if not text_cols:
                    text_cols = cols
                sql = f"SELECT {', '.join(cols)} FROM {table} LIMIT {limit}"
                cur.execute(sql)
                for row in cur.fetchall():
                    obj = {c: (row[i] if row[i] is not None else "") for i, c in enumerate(cols)}
                    if any(isinstance(obj.get(c), bytes) for c in cols):
                        continue
                    text = "\n".join(str(obj.get(c, "")) for c in text_cols if obj.get(c))
                    if not text.strip():
                        continue
                    yield f"{db.name}:{table}", obj
            except Exception:
                continue
        con.close()
    except Exception:
        return


def make_text(domain: str, source: str, obj: dict, max_preview: int = 5000) -> str:
    parts = [f"[{domain.upper()}] source={source}"]
    for k in ["id", "trace_id", "session_id", "agent_id", "entity_id", "tick_id", "doc_id", "doc_type", "step_type", "event_type", "recorded_at", "timestamp", "created_at", "started_at", "ended_at", "model", "engine", "agent", "outcome", "status", "type", "name", "title", "model_used", "voice_used", "source", "channel"]:
        v = obj.get(k)
        if v not in (None, ""):
            parts.append(f"{k}={v}")
    content_keys = ["content", "text", "body", "description", "query", "result", "prompt", "system_prompt", "summary_memory", "findings", "metadata", "progress", "conversation_state", "tool_state", "payload", "state_json", "profile_json", "data", "finding"]
    for k in content_keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}=" + v.strip()[:max_preview])
            break
    return "\n".join(parts)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train_rows = []

    # OpenJarvis
    oj = ROOT / ".openjarvis"
    table_map = {
        "agents.db": ["managed_agents", "agent_tasks", "agent_checkpoints", "agent_messages", "agent_learning_log"],
        "traces.db": ["traces", "trace_steps"],
        "knowledge.db": ["knowledge_chunks"],
        "memory.db": ["documents"],
        "digest.db": ["digests"],
        "telemetry.db": ["telemetry", "mining_stats"],
        "sync_state.db": ["sync_state"],
        "audit.db": ["security_events"],
    }
    for db_name, tables in table_map.items():
        db = oj / db_name
        if not db.exists():
            continue
        count = 0
        for source, obj in iter_db_rows(db, tables, limit=4000):
            train_rows.append(json.dumps({"text": make_text("openjarvis", source, obj)}, ensure_ascii=False))
            count += 1
        print(f"openjarvis {db_name}: {count}")

    # Hermes state
    hermes_db = ROOT / "AppData" / "Local" / "hermes" / "state.db"
    if hermes_db.exists():
        tables = ["sessions", "messages", "state_meta", "compression_locks"]
        count = 0
        for source, obj in iter_db_rows(hermes_db, tables, limit=4000):
            train_rows.append(json.dumps({"text": make_text("hermes", source, obj)}, ensure_ascii=False))
            count += 1
        print(f"hermes state.db: {count}")

    # POG2 sovereign memory
    sov = ROOT / ".pog2-sovereign"
    for db_name, tables in {
        "memory/memory.db": ["entities", "pedagogy", "memories", "vibe_state", "player_identity", "cognitive_anomalies"],
        "memory/rsmv_cache.db": ["grounded_entities"],
    }.items():
        db = sov / db_name
        if not db.exists():
            continue
        count = 0
        for source, obj in iter_db_rows(db, tables, limit=4000):
            train_rows.append(json.dumps({"text": make_text("pog2", source, obj)}, ensure_ascii=False))
            count += 1
        print(f"pog2 {db_name}: {count}")

    txt_roots = {
        "GEMINI_KNOWLEDGE": ROOT / ".gemini" / "antigravity" / "knowledge",
        "GEMINI_CONVERSATIONS": ROOT / ".gemini" / "antigravity" / "conversations",
        "CLAUDE_PROJECT": ROOT / ".claude" / "projects" / "D--POG2" / "memory",
    }
    exts = {".md", ".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml", ".toml", ".ts", ".js"}
    for domain, folder in txt_roots.items():
        if not folder.exists():
            continue
        count = 0
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if not txt.strip():
                continue
            train_rows.append(json.dumps({"text": f"[{domain}] source={path}\n" + txt[:5000]}, ensure_ascii=False))
            count += 1
            if count >= 1200:
                break
        print(f"{domain}: {count}")

    random.shuffle(train_rows)
    split = max(1, int(len(train_rows) * 0.9))
    train_set = train_rows[:split]
    val_set = train_rows[split:]
    TRAIN_OUT.write_text("\n".join(train_set) + "\n", encoding="utf-8")
    VAL_OUT.write_text("\n".join(val_set) + "\n", encoding="utf-8")
    print(f"memory_rows_total={len(train_rows)}")
    print(f"train={len(train_set)} val={len(val_set)}")
    print(f"train_file={TRAIN_OUT} ({TRAIN_OUT.stat().st_size} bytes)")
    print(f"val_file={VAL_OUT} ({VAL_OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

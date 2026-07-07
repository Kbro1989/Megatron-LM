#!/usr/bin/env python3
"""Never-ending loop task kanban for provider capability mirroring."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
KANBAN_DIR = ROOT / "kanban"
PROVIDER_GATE = KANBAN_DIR / "provider_gate.json"
TASK_STATE = KANBAN_DIR / "task_state.json"
AUDIT_LOG = KANBAN_DIR / "audit.log"


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_providers() -> List[dict]:
    if not PROVIDER_GATE.exists():
        raise SystemExit(f"Missing {PROVIDER_GATE}")
    data = json.loads(PROVIDER_GATE.read_text(encoding="utf-8"))
    return data.get("providers", [])


def current_tasks() -> Dict[str, dict]:
    return _load_json(TASK_STATE)


def enqueue(provider_id: str, capability: str, status: str = "queued") -> dict:
    tasks = current_tasks()
    key = f"{provider_id}::{capability}"
    task = {
        "provider_id": provider_id,
        "capability": capability,
        "status": status,
        "attempts": tasks.get(key, {}).get("attempts", 0) + 1,
        "created_at": time.time(),
        "updated_at": time.time(),
        "result": None,
    }
    tasks[key] = task
    _save_json(TASK_STATE, tasks)
    return task


def complete(key: str, result: Optional[dict] = None, status: str = "completed") -> dict:
    tasks = current_tasks()
    if key not in tasks:
        raise KeyError(key)
    tasks[key]["status"] = status
    tasks[key]["result"] = result
    tasks[key]["updated_at"] = time.time()
    _save_json(TASK_STATE, tasks)
    return tasks[key]


def audit(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def mirror_providers(providers: List[dict]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"mirrored": [], "blocked": [], "skipped": []}
    for provider in providers:
        pid = provider.get("id")
        paywalled = bool(provider.get("paywalled"))
        mirror = provider.get("mirror") or {}
        capabilities = provider.get("capabilities", [])
        if not capabilities:
            summary["skipped"].append({"provider_id": pid, "reason": "no capabilities declared"})
            continue
        if paywalled:
            for capability in capabilities:
                task = enqueue(pid, capability, status="mirrored")
                summary["mirrored"].append(
                    {
                        "provider_id": pid,
                        "capability": capability,
                        "task_key": f"{pid}::{capability}",
                        "attempts": task["attempts"],
                        "mirror": mirror,
                    }
                )
                audit(f"mirrored provider={pid} capability={capability} attempts={task['attempts']}")
            summary["blocked"].append({"provider_id": pid, "capabilities": capabilities, "mirror": mirror})
        else:
            for capability in capabilities:
                task = enqueue(pid, capability, status="available")
                summary["skipped"].append(
                    {
                        "provider_id": pid,
                        "capability": capability,
                        "task_key": f"{pid}::{capability}",
                        "reason": "not paywalled",
                    }
                )
                audit(f"available provider={pid} capability={capability}")
    return summary


def loop() -> None:
    providers = load_providers()
    audit("kanban loop start")
    summary = mirror_providers(providers)
    audit(f"kanban loop end mirrored={len(summary['mirrored'])} blocked={len(summary['blocked'])} skipped={len(summary['skipped'])}")
    print(json.dumps(summary, indent=2))


def _usage_surface() -> Dict[str, Any]:
    root = Path(".").resolve()
    patterns = [
        "**/*transformer_engine*",
        "**/gpt_builders.py",
        "**/pretrain_*.py",
        "**/train_*.py",
        "**/*.sh",
        "**/Dockerfile*",
        "**/*.yaml",
        "**/*.yml",
    ]
    hits: List[dict] = []
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file():
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                count = txt.count("transformer_engine")
                if count:
                    hits.append(
                        {
                            "path": str(p.relative_to(root)),
                            "transformer_engine_hits": count,
                            "provider_id": "nvidia_transformer_engine",
                        }
                    )
    return {"surface_hits": hits, "total_hits": len(hits)}


def loop_forever(interval: int = 60) -> None:
    providers = load_providers()
    seen: set = set()
    pass_no = 0
    audit("kanban forever start")
    while True:
        pass_no += 1
        summary = mirror_providers(providers)
        surface = _usage_surface()
        new_keys = {t["task_key"] for t in summary["mirrored"] + summary["skipped"]}
        report = {
            "pass": pass_no,
            "summary": summary,
            "surface_new": sorted(list(new_keys - seen))[:50],
            "surface_total": surface["total_hits"],
        }
        seen |= new_keys
        audit(f"kanban pass={pass_no} mirrored={len(summary['mirrored'])} blocked={len(summary['blocked'])} skipped={len(summary['skipped'])} surface_total={surface['total_hits']}")
        print(json.dumps(report, indent=2))
        time.sleep(interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run forever")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between scans")
    args = parser.parse_args()
    if args.loop:
        loop_forever(args.interval)
    else:
        loop()

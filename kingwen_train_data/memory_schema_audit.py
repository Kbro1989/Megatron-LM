#!/usr/bin/env python3
import sqlite3
from pathlib import Path

DB_ROOTS = [
    Path("/c/Users/krist/.openjarvis"),
    Path("/c/Users/krist/.pog2-sovereign/memory"),
    Path("/c/Users/krist/AppData/Local/hermes"),
]

for root in DB_ROOTS:
    if not root.exists():
        print(f"\n=== {root} MISSING ===")
        continue
    for db in root.glob("*.db"):
        print(f"\n=== {db} ===")
        try:
            con = sqlite3.connect(str(db))
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            print(f"tables: {tables}")
            for t in tables[:10]:
                cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})").fetchall()]
                print(f"  {t}: {cols}")
                row = cur.execute(f"SELECT * FROM {t} LIMIT 1").fetchone()
                if row:
                    print(f"    sample: {list(row)[:8]}")
            con.close()
        except Exception as e:
            print(f"  ERROR: {e}")

#!/usr/bin/env python3
"""Jarvis-native dataset loader for JSONL + indexed corpora + labels.

Training-only dependency: torch.utils.data.Dataset is imported lazily via
_get_torch_dataset_class() so Windows-native preprocessing can inspect this
module without the WSL venv. Keep this file out of the preprocessing path
unless torch is available; otherwise it remains loadable-only until runtime.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _get_torch_dataset_class():  # pragma: no cover - dependency guard
    try:
        from torch.utils.data import Dataset  # type: ignore[import]
        return Dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Runtime dependency missing: torch. Use WSL venv for dataset mode.") from exc


class JarvisJsonlDataset:
    def __init__(self, jsonl_path: Path, labels_path: Path | None = None, max_seq: int = 4096):
        self.jsonl_path = jsonl_path
        self.labels_path = labels_path
        self.max_seq = max_seq
        self.samples = []
        self.labels = []
        self._load(jsonl_path, labels_path)

    def _load(self, jsonl_path: Path, labels_path: Path | None):
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("text"):
                    self.samples.append(obj["text"])
        if labels_path and labels_path.exists():
            with labels_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self.labels.append(obj.get("labels", {}))
        if not self.labels:
            self.labels = [{} for _ in range(len(self.samples))]
        print(f"JarvisJsonlDataset loaded {len(self.samples)} samples, {len(self.labels)} labels from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "id": idx,
            "text": self.samples[idx],
            "labels": self.labels[idx],
        }


class JarvisIndexedDataset:
    def __init__(self, bin_path: Path, idx_path: Path, labels_path: Path | None = None):
        self.bin_path = bin_path
        self.idx_path = idx_path
        self.labels_path = labels_path
        self.offsets = np.memmap(idx_path, dtype=np.int64, mode="r")
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.labels = []
        self._load_labels()
        print(f"JarvisIndexedDataset loaded {len(self.offsets)-1} samples from {bin_path}")

    def _load_labels(self):
        if self.labels_path and self.labels_path.exists():
            with self.labels_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self.labels.append(obj.get("labels", {}))
        if not self.labels:
            self.labels = [{} for _ in range(len(self.offsets) - 1)]

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, idx):
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        item = {
            "id": idx,
            "tokens": self.tokens[start:end].tolist(),
            "labels": self.labels[idx],
        }
        return item


def main():
    print("Jarvis dataset loaders ready.")
    ds = JarvisJsonlDataset(
        Path("kingwen_train_data/combined_pretrain_train.jsonl"),
        Path("kingwen_train_data/model/jarvis-native-kingwen-life/labels_train.jsonl"),
    )
    print("sample 0:", ds[0])


if __name__ == "__main__":
    main()

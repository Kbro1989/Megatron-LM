"""Domain-aware King Wen / Jarvis-life / slider / ternary dataset for Megatron."""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
import torch


@dataclass(frozen=True)
class SampleMeta:
    """Hold domain, labels, and training weights outside token tensors."""
    domain: str
    usage: str
    sample_id: int
    text: str
    hexagram_id: Optional[int]
    phase_bits: Optional[int]
    phase_temporal: Optional[str]
    phase_polarity: Optional[str]
    porosity: Optional[int]
    trajectory: Optional[str]
    pre_slider: Optional[int]
    post_slider: Optional[int]
    porosity_weight: float
    domain_weight: float
    voice_weight: float


class DomainSample:
    __slots__ = ("text", "ids", "meta", "sample_weight")

    def __init__(self, text: str, ids: List[int], meta: SampleMeta, sample_weight: float):
        self.text = text
        self.ids = ids
        self.meta = meta
        self.sample_weight = float(sample_weight)


class KingwenDatasetConfig(GPTDatasetConfig):
    def __init__(
        self,
        path: pathlib.Path,
        label_path: Optional[pathlib.Path] = None,
        tokenizer_type: str = "GPT2BPETokenizer",
        weight_schema: str = "porosity",
        domain_weights: Optional[Dict[str, float]] = None,
        sample_weight_key: str = "sample_weight",
        **kwargs,
    ):
        self.path = pathlib.Path(path)
        self.label_path = pathlib.Path(label_path) if label_path else None
        self.tokenizer_type = tokenizer_type
        self.weight_schema = weight_schema
        self.domain_weights = domain_weights or {
            "KING_WEN_ORACLE": 1.0,
            "SOVEREIGN_PIPELINE_SCENE": 1.0,
            "OPENJARVIS_INTERACTION": 1.0,
            "HERMES": 1.0,
            "POG2": 1.0,
            "SLIDER_CAPTURE": 1.0,
            "TERNARY_INTERACTION": 1.0,
        }
        self.sample_weight_key = sample_weight_key
        super().__init__(**kwargs)


class KingwenDataset(MegatronDataset):
    """Causal LM dataset with domain metadata and sample weights.

    Domain model:
      KING_WEN_ORACLE / SOVEREIGN_PIPELINE_SCENE  -> base oracle expansion text
      OPENJARVIS_INTERACTION                        -> turn-level Jarvis traces with ternary
      HERMES / POG2                                 -> langauge/runtime memory
      SLIDER_CAPTURE                                -> pre|post slider delta records
      TERNARY_INTERACTION                           -> yao/state-change interactions
    """

    _DOWNSAMPLE_RATE = 1

    def __init__(self, dataset_path: str, config: KingwenDatasetConfig, tokenizer, num_samples: Optional[int] = None):
        super().__init__(dataset_path, config, tokenizer)
        self._cfg = config
        self._tokenizer = tokenizer
        self._samples = self._load(config.path, config.label_path, num_samples)
        print(f"KingwenDataset loaded {len(self._samples)} samples from {config.path}")
        counts: Dict[str, int] = {}
        for s in self._samples:
            counts[s.meta.domain] = counts.get(s.meta.domain, 0) + 1
        print("Domain counts:", counts)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load(path: pathlib.Path, label_path: Optional[pathlib.Path], num_samples: Optional[int]) -> List[DomainSample]:
        samples: List[DomainSample] = []
        labels_map = KingwenDataset._load_labels(label_path) if label_path else {}
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if num_samples is not None and idx >= num_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text") if isinstance(rec, dict) else None
                if not isinstance(text, str) or not text.strip():
                    continue
                lbl = labels_map.get(idx, {})
                meta = KingwenDataset._infer_meta(idx, text, lbl)
                samples.append(DomainSample(text=text, ids=[0], meta=meta, sample_weight=0.0))
        return samples

    @staticmethod
    def _load_labels(path: Optional[pathlib.Path]) -> Dict[int, dict]:
        out: Dict[int, dict] = {}
        if not path or not path.exists():
            return out
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and "labels" in obj:
                    out[idx] = obj["labels"] or {}
        return out

    @staticmethod
    def _infer_meta(idx: int, text: str, lbl: dict) -> SampleMeta:
        domain = KingwenDataset._parse_domain(text)
        usage = KingwenDataset._domain_usage(domain)
        porosity_level = lbl.get("porosity")
        try:
            porosity_int = int(porosity_level) if porosity_level is not None else None
        except (TypeError, ValueError):
            porosity_int = None
        try:
            pre_slider = int(lbl["pre_slider"]) if "pre_slider" in lbl else None
        except (TypeError, ValueError):
            pre_slider = None
        try:
            post_slider = int(lbl["post_slider"]) if "post_slider" in lbl else None
        except (TypeError, ValueError):
            post_slider = None
        return SampleMeta(
            domain=domain,
            usage=usage,
            sample_id=idx,
            text=text,
            hexagram_id=int(lbl["hexagram_id"]) if lbl.get("hexagram_id") is not None else None,
            phase_bits=int(lbl["phase_bits"]) if lbl.get("phase_bits") is not None else None,
            phase_temporal=lbl.get("phase_temporal"),
            phase_polarity=lbl.get("phase_polarity"),
            porosity=porosity_int,
            trajectory=lbl.get("trajectory"),
            pre_slider=pre_slider,
            post_slider=post_slider,
            porosity_weight=KingwenDataset._porosity_weight(porosity_int),
            domain_weight=1.0,
            voice_weight=KingwenDataset._voice_weight(lbl),
        )

    @staticmethod
    def _parse_domain(text: str) -> str:
        if text.startswith("[HERMES]"):
            return "HERMES"
        if text.startswith("[POG2]"):
            return "POG2"
        if text.startswith("[OPENJARVIS]"):
            return "OPENJARVIS_INTERACTION"
        if text.startswith("[SLIDER_CAPTURE]"):
            return "SLIDER_CAPTURE"
        if text.startswith("[TERNARY_INTERACTION]"):
            return "TERNARY_INTERACTION"
        if text.startswith("[SOVEREIGN_PIPELINE_SCENE]"):
            return "SOVEREIGN_PIPELINE_SCENE"
        if text.startswith("[KING_WEN_ORACLE]"):
            return "KING_WEN_ORACLE"
        return "UNKNOWN"

    @staticmethod
    def _domain_usage(domain: str) -> str:
        return {
            "KING_WEN_ORACLE": "train",
            "SOVEREIGN_PIPELINE_SCENE": "train",
            "OPENJARVIS_INTERACTION": "train",
            "HERMES": "corpus",
            "POG2": "corpus",
            "SLIDER_CAPTURE": "train",
            "TERNARY_INTERACTION": "train",
        }.get(domain, "corpus")

    @staticmethod
    def _porosity_weight(porosity: Optional[int]) -> float:
        if porosity is None:
            return 1.0
        mapping = {0: 1.0, 1: 1.05, 2: 1.15, 3: 1.35, 4: 1.5}
        return float(mapping.get(int(porosity), 1.0))

    @staticmethod
    def _voice_weight(lbl: dict) -> float:
        return float(lbl.get("voiceWeight") or 0.0)

    # ------------------------------------------------------------------
    # Tokenization / weight computation
    # ------------------------------------------------------------------
    def _load_sample(self, idx: int):
        sample = self._samples[idx]
        sample.ids = self._tokenizer.tokenize(sample.text)
        sample.sample_weight = self._sample_weight(sample.meta, len(sample.ids))

    def _sample_weight(self, meta: SampleMeta, token_count: int) -> float:
        base = 1.0 if token_count > 0 else 0.0
        w = base * float(self._cfg.domain_weights.get(meta.domain, 1.0)) * meta.porosity_weight
        if meta.usage == "corpus":
            w *= 0.6
        return max(w, 0.0)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        sample = self._samples[idx]
        if not sample.ids:
            self._load_sample(idx)
        ids = list(sample.ids)
        max_samples = getattr(self._config, "max_samples", None)
        if max_samples is not None:
            ids = ids[:max_samples]
        return {
            "text": torch.tensor(ids, dtype=torch.long),
            "id": idx,
            "sample_weight": torch.tensor(sample.sample_weight, dtype=torch.float),
            "domain": sample.meta.domain,
            "usage": sample.meta.usage,
            "hexagram_id": torch.tensor(sample.meta.hexagram_id if sample.meta.hexagram_id is not None else -1, dtype=torch.long),
            "phase_bits": torch.tensor(sample.meta.phase_bits if sample.meta.phase_bits is not None else -1, dtype=torch.long),
            "phase_temporal": sample.meta.phase_temporal or "",
            "phase_polarity": sample.meta.phase_polarity or "",
            "porosity": torch.tensor(sample.meta.porosity if sample.meta.porosity is not None else -1, dtype=torch.long),
            "trajectory": sample.meta.trajectory or "",
            "pre_slider": torch.tensor(sample.meta.pre_slider if sample.meta.pre_slider is not None else -1, dtype=torch.long),
            "post_slider": torch.tensor(sample.meta.post_slider if sample.meta.post_slider is not None else -1, dtype=torch.long),
            "porosity_weight": torch.tensor(sample.meta.porosity_weight, dtype=torch.float),
            "voice_weight": torch.tensor(sample.meta.voice_weight, dtype=torch.float),
        }

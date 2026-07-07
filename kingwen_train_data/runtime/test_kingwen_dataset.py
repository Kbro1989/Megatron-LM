"""Smoke test real KingwenDataset through Megatron's dataset builder."""
import os
import pathlib

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from kingwen_train_data.runtime.kingwen_dataset import KingwenDatasetConfig, KingwenDataset

JSONL = pathlib.Path("kingwen_train_data/combined_pretrain_train.jsonl")
LABELS = pathlib.Path("kingwen_train_data/model/jarvis-native-kingwen-life/labels_train.jsonl")
if not JSONL.exists():
    raise SystemExit(f"Missing {JSONL}")
if not LABELS.exists():
    raise SystemExit(f"Missing labels {LABELS}")

config = KingwenDatasetConfig(
    path=JSONL,
    label_path=LABELS,
    tokenizer_type="GPT2BPETokenizer",
    random_seed=42,
    sequence_length=128,
    reset_position_ids=False,
    reset_attention_mask=False,
    eod_mask_loss=False,
    mid_level_dataset_surplus=0.0,
    split="train",
    weighted_splits=None,
)
datasets = BlendedMegatronDatasetBuilder(
    KingwenDataset, [1.0, None, None], lambda: True, config
).build()
print("datasets:", len(datasets))
ds = datasets[0]
print("dataset len:", len(ds))
item = ds[0]
print("item keys:", sorted(item.keys()))
print("text shape:", item["text"].shape, "dtype:", item["text"].dtype)
print("sample_weight:", item.get("sample_weight"))
print("domain:", item.get("domain"))
print("usage:", item.get("usage"))
print("hexagram_id:", item.get("hexagram_id"))
print("phase_bits:", item.get("phase_bits"))
print("phase_temporal:", item.get("phase_temporal"))
print("phase_polarity:", item.get("phase_polarity"))
print("porosity:", item.get("porosity"))
print("trajectory:", item.get("trajectory"))
print("pre_slider:", item.get("pre_slider"))
print("post_slider:", item.get("post_slider"))
print("porosity_weight:", item.get("porosity_weight"))
print("voice_weight:", item.get("voice_weight"))
print("preview:", item["text"][:16].tolist())

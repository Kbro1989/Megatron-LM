import os
import numpy as np
from pathlib import Path

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.indexed_dataset import IndexedDataset

BIN_PATH = "kingwen_train_data/kingwen_pretrain"

print("IndexedDataset samples:", len(IndexedDataset(BIN_PATH)))
print("First last tokens:", len(IndexedDataset(BIN_PATH)[0]), len(IndexedDataset(BIN_PATH)[-1]))

config = GPTDatasetConfig(
    random_seed=42,
    sequence_length=1024,
    reset_position_ids=False,
    reset_attention_mask=False,
    eod_mask_loss=False,
    mid_level_dataset_surplus=0.005,
)
datasets = BlendedMegatronDatasetBuilder(
    MockGPTDataset, [1.0, None, None], lambda: True, config
).build()
print("Blended datasets:", len(datasets))
print("Train dataset samples:", len(datasets[0]))
batch = datasets[0][0]
print("Sample keys:", sorted(batch.keys()))
print("Sample tokens shape:", batch["tokens"].shape)
print("Sample attention_mask shape:", batch["attention_mask"].shape)
print("Sample labels shape:", batch["labels"].shape)

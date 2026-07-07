import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from kingwen_train_data.test_jsonl_dataset import JsonlDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
ds = JsonlDataset(Path("kingwen_train_data/kingwen_pretrain.jsonl"), tokenizer, max_seq=128)
loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
batch = next(iter(loader))
print("batch keys:", sorted(batch.keys()))
print("text shape:", batch["text"].shape, "dtype:", batch["text"].dtype)
print("sample 0 ids[:16]:", batch["text"][0][:16].tolist())
print("sample 0 decoded:", tokenizer.decode(batch["text"][0].tolist())[:180])

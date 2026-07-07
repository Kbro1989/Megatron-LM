"""Minimal torch Dataset from King Wen JSONL, tokenized with Hugging Face GPT-2 tokenizer."""
import json
import pathlib
import torch
from transformers import AutoTokenizer

SEQ = 128
JSONL = pathlib.Path("kingwen_train_data/kingwen_pretrain.jsonl")


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path: pathlib.Path, tokenizer, max_seq: int = SEQ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.samples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.samples.append(rec.get("text", ""))
        print(f"JsonlDataset loaded {len(self.samples)} samples from {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        ids = self.tokenizer(text, truncation=True, max_length=self.max_seq, return_tensors="pt")["input_ids"][0].tolist()
        return {
            "text": torch.tensor(ids, dtype=torch.long),
            "id": idx,
        }


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = JsonlDataset(JSONL, tokenizer)
    item = ds[0]
    print("item keys:", sorted(item.keys()))
    print("text shape:", item["text"].shape, "dtype:", item["text"].dtype)
    print("preview:", item["text"][:16].tolist())
    print("decoded preview:", tokenizer.decode(item["text"].tolist()))


if __name__ == "__main__":
    main()

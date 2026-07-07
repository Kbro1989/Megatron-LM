from transformers import AutoTokenizer
from pathlib import Path
tok = AutoTokenizer.from_pretrained("kingwen_train_data/model/jarvis-native-kingwen-life", use_fast=True)
print("vocab_size", tok.vocab_size)
for t in ["<|kingwen_oracle|>", "[KING_WEN_ORACLE]", "<|voiceWeight|>"]:
    ids = tok(t, add_special_tokens=False)["input_ids"]
    print(t, "->", ids)

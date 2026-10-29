"""LoRA fine-tune Qwen3-1.7B on the Doom spine corpus (behavior cloning).

Run on the desktop: D:\\doom-venv\\Scripts\\python.exe train_lora.py
Prompt tokens are masked from the loss; the model learns only the [1/2/3] pick.
"""
from __future__ import annotations
import json
import random

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

# memory plan: run with CUDA_VISIBLE_DEVICES=2 (RTX 3060 12GB)
CORPUS = r"D:\doomsrv\corpus2.jsonl"
BASE = r"D:\doomsrv\Qwen3-1.7B-hf"
OUT = r"D:\doomsrv\spine-lora"
MAX_LEN = 384
SUBSAMPLE = 40000
SEED = 13


class SpineDS(Dataset):
    def __init__(self, tok, rows):
        self.tok = tok
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        full = self.tok(r["prompt"] + r["completion"], add_special_tokens=False)
        prompt_only = self.tok(r["prompt"], add_special_tokens=False)
        ids = full["input_ids"][:MAX_LEN]
        labels = list(ids)
        plen = min(len(prompt_only["input_ids"]), len(ids))
        labels[:plen] = [-100] * plen          # loss only on the pick
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def main():
    random.seed(SEED)
    rows = []
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({"prompt": d["prompt"], "completion": d["completion"],
                         "w": d.get("meta", {}).get("weight", 1.0)})
    # weight-aware subsample: winning-episode samples are 2x more likely kept
    # balanced: ~40% deviation labels (2/3), 60% top picks
    dev = [r for r in rows if r["completion"] != "[1]"]
    top = [r for r in rows if r["completion"] == "[1]"]
    n_dev = min(len(dev), int(SUBSAMPLE * 0.4))
    n_top = SUBSAMPLE - n_dev * (len(dev) // max(n_dev, 1) or 1)
    dev = dev * (n_dev // max(len(dev), 1) + 1)
    random.shuffle(dev)
    random.shuffle(top)
    rows = top[:SUBSAMPLE - n_dev] + dev[:n_dev]
    print(f"balanced: dev={n_dev} top={len(rows)-n_dev}")
    print(f"train samples: {len(rows)}")

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa")
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = SpineDS(tok, rows)
    coll = DataCollatorForSeq2Seq(tok, padding=True, max_length=MAX_LEN, pad_to_multiple_of=8)

    args = TrainingArguments(
        output_dir=OUT,
        num_train_epochs=1.5,
        per_device_train_batch_size=6,
        gradient_accumulation_steps=6,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=True,
        logging_steps=50,
        save_steps=2000,
        save_total_limit=2,
        report_to=[],
        seed=SEED,
    )
    tr = Trainer(model=model, args=args, train_dataset=ds, data_collator=coll)
    tr.train()
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print("saved adapter ->", OUT)


if __name__ == "__main__":
    main()

"""Merge N LoRA adapters by weight averaging (LoRA soup).

All shards trained from the same base + same init, different data slices ->
averaging A/B matrices is a valid merge (model-soup literature).
"""
from __future__ import annotations
import torch
from peft import PeftModel
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"D:\doomsrv\Qwen3-1.7B-hf"
SHARDS = [rf"D:\doomsrv\spine-lora-s{i}" for i in range(3)]
OUT = r"D:\doomsrv\spine-lora-merged"

def main():
    sds = []
    for d in SHARDS:
        m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
        pm = PeftModel.from_pretrained(m, d)
        sds.append(get_peft_model_state_dict(pm))
        del m, pm
    keys = sds[0].keys()
    assert all(set(s.keys()) == set(keys) for s in sds), "adapter key mismatch"
    avg = {k: sum(s[k].float() for s in sds) / len(sds) for k in keys}

    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    pm = PeftModel.from_pretrained(base, SHARDS[0])
    ok = set_peft_model_state_dict(pm, avg)
    print("set_peft_model_state_dict ->", ok)
    pm.save_pretrained(OUT)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
    print("merged adapter ->", OUT)

if __name__ == "__main__":
    main()

"""Bake merged LoRA into full model -> HF dir ready for GGUF conversion."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import os as _os
_D = _os.environ.get("JEVDOOM_DATA",
    r"D:\doomsrv" if _os.name == "nt" else _os.path.expanduser("~/doomsrv"))
BASE = _D + "/Qwen3-1.7B-hf"
ADAPTER = _D + "/spine-lora-merged"
OUT = _D + "/spine-lora-baked"

m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
pm = PeftModel.from_pretrained(m, ADAPTER)
merged = pm.merge_and_unload()
merged.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
print("BAKED ->", OUT)

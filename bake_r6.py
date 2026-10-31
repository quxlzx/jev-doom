"""Bake merged LoRA into full model -> HF dir ready for GGUF conversion."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"D:\doomsrv\Qwen3-1.7B-hf"
ADAPTER = r"D:\doomsrv\spine-lora-merged"
OUT = r"D:\doomsrv\Qwen3-1.7B-r6"

m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
pm = PeftModel.from_pretrained(m, ADAPTER)
merged = pm.merge_and_unload()
merged.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
print("BAKED ->", OUT)

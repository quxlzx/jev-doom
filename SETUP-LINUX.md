# Linux setup runbook (Athena migration)

Everything in this repo is OS-agnostic Python. The Windows-only parts were the
`schtasks` launchers and `D:\` paths — on Linux, follow this.

## 1. System deps

```bash
sudo apt install -y build-essential cmake git python3-venv libgl1-mesa-dri \
    libgl1 libglib2.0-0 openjdk-17-jre   # java only for ViZDoom's game loop? no — skip java; vizdoom bundles its own
```

ViZDoom 1.3.x on Linux needs: `libgl1`, and the bundled DOOM.SW runtime
downloads on first import (needs network once).

## 2. Project venv

```bash
git clone https://github.com/quxlzx/jev-doom && cd jev-doom
python3 -m venv .venv
.venv/bin/pip install vizdoom numpy matplotlib rapidocr-onnxruntime
# training extras (only on the GPU box):
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install transformers peft datasets accelerate
```

## 3. Models (spine + brain)

Download to `~/doomsrv/` (or set paths):

- `Qwen3-1.7B-Q8_0.gguf` (base spine) — HuggingFace: Qwen/Qwen3-1.7B-GGUF
- `Qwen3-8B-Q4_K_M.gguf` (brain) — Qwen/Qwen3-8B-GGUF
- llama.cpp: `git clone https://github.com/ggml-org/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j`

Linux launchers (replace of spine.cmd / brain.cmd):

```bash
# spine (R8 baked model; rebuild via bake below if retraining)
./llama.cpp/build/bin/llama-server -m ~/doomsrv/spine-r8-q8.gguf \
    -ngl 99 -c 2048 -t 8 --host 127.0.0.1 --port 8181 \
    --grammar-file src/reflex_pick.gbnf
# brain
./llama.cpp/build/bin/llama-server -m ~/doomsrv/Qwen3-8B-Q4_K_M.gguf \
    -ngl 99 -c 4096 -t 8 --host 127.0.0.1 --port 8182
```

## 4. Training pipeline on Linux

- corpora live in `corpora/` (backed up; corpus9 = R7, corpus8 = R6, corpus10 = R9-partial)
- `train_lora.py` shards: `CUDA_VISIBLE_DEVICES=$i .venv/bin/python train_lora.py --shard $i --nshards 3 --out ~/doomsrv/spine-lora-s$i`
- merge: `python merge_loras.py` (edit BASE/SHARDS/OUT paths to Linux)
- bake: `python bake_r6.py` (edit paths) → `convert_hf_to_gguf.py <dir> --outfile spine-rN-f16.gguf --outtype f16` → `llama-quantize ... Q8_0`
- NOTE: this llama.cpp build has no PEFT-adapter GGUF support — always BAKE.

## 5. Run

```bash
cd src
JEVDOOM_SPINE_URL=http://127.0.0.1:8181 \
JEVDOOM_BRAIN_URL=http://127.0.0.1:8182/v1 JEVDOOM_BRAIN_MODEL=qwen3-8b \
../.venv/bin/python agent.py --episodes 10 --scenario deadly_corridor --skill 1 --spine model
# diagnostics:
../.venv/bin/python trace_viz.py --episodes 3 --skill 1   # writes traces/map_epN.png
```

## State at migration time (2026-10-31)

- deadly_corridor skill 1: 0 wins / 18-28 kills / ~2 deaths per 8 eps
- agent reaches the south corridor (barons guard the last ~500u to the exit)
- R9 corpus partial (corpus10, ~21MB) — collection was paused for the move;
  resume: `collect_data.py --episodes 300 --epsilon 0.15 --scenario deadly_corridor --skill 1 --out corpora/corpus10.jsonl`
- next round: train on corpus10 (baron-era demos) → bake → shootout

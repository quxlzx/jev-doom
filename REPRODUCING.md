# Reproducing jev-doom from a fresh clone

Everything below was run on a clean Linux x86_64 box (Ubuntu, Python 3.14.7,
no display) on 2026-11-04 at commit `55a1020`+fixes. Steps marked
**verified** were executed end-to-end; steps marked **unverified** are
documented from the repo's own scripts but were not re-run in this audit.

## 0. Prerequisites

- Linux x86_64 (the ViZDoom pip wheel ships its own engine binary; no system
  Doom install needed). Python 3.10–3.14 — verified on 3.14.
- No display required: `env.py` runs `window=False` + `Mode.PLAYER`, i.e.
  fully headless. No Xvfb, no GPU.
- Network access once, for `pip install` (the wheel bundles the scenarios).

## 1. Install (verified)

```bash
git clone https://github.com/quxlzx/jev-doom && cd jev-doom
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # vizdoom>=1.3,<2, numpy
# optional extras:
.venv/bin/pip install -r requirements-viz.txt      # matplotlib, imageio-ffmpeg (trace_viz / render_replay)
.venv/bin/pip install -r requirements-train.txt    # torch, transformers, peft, ... (retraining only)
```

Verified install: `vizdoom==1.3.2`, `numpy==2.5.4`, `matplotlib==3.11.2`,
`imageio-ffmpeg==0.8.1` on Python 3.14.7.

## 2. Scenarios / config files — nothing to download (verified)

Both scenarios the agents use ship **inside the ViZDoom pip wheel**. `src/env.py`
resolves them via `vizdoom.scenarios_path`:

- `take_cover` -> `<site-packages>/vizdoom/scenarios/take_cover.cfg` (+ `.wad`)
- `deadly_corridor` -> `<site-packages>/vizdoom/scenarios/deadly_corridor.cfg` (+ `.wad`)

Confirm with:

```bash
.venv/bin/python -c "import vizdoom, os; print(vizdoom.scenarios_path); \
print([f for f in os.listdir(vizdoom.scenarios_path) if 'take_cover' in f or 'deadly' in f])"
```

Expected: the four files `take_cover.cfg/.wad`, `deadly_corridor.cfg/.wad`.
Note: the wheel's bundled cfgs are trimmed stubs — the code deliberately sets
buttons, game variables, timeouts and resolution through the API instead of
trusting the cfg, so no extra config files are needed.

## 3. Run an agent episode (verified, headless)

```bash
cd src
# fast smoke test (~0.5 s/episode, 350-tic episodes):
../.venv/bin/python agent.py --episodes 5 --scenario take_cover --skill 4 --seed 7
# the flagship arena (~60 s/episode, 9000-tic budget):
../.venv/bin/python agent.py --episodes 10 --scenario deadly_corridor --skill 1 --seed 42
```

Expected output shape (real lines from this machine):

```
brain: RuleBrain | spine: ReflexSpine | skill=4
episode 0: WON=0 reward=0.0 kills=0 deaths=0 health=100 tics=14 decisions=14 escalations=0 stances={'navigate': 14} wall=0.5s
```

```
brain: RuleBrain | spine: ReflexSpine | skill=1
episode 0: WON=0 reward=0.0 kills=2 deaths=0 health=50 tics=603 decisions=603 escalations=0 stances={'navigate': 480, 'aggressive': 123} wall=63.0s
```

Caveat: the `tics=` field is ViZDoom's *state number* (one per decision), not
engine tics — `decisions` and `tics` are equal by construction.

Diagnostics (verified): `../.venv/bin/python trace_viz.py --episodes 1 --skill 1`
prints a per-episode line (`ep 0: 400 decisions, kills=2, ...`) and writes
`traces/map_epN.png` + `traces/trace_epN.json`. Demos: add `--record` to
`agent.py` to write `.lmp` files to `replays/`, then
`../.venv/bin/python render_replay.py <file.lmp>` renders an mp4
(**unverified** in this audit — deps installed, not executed).

## 4. Random seeds

Default behavior (no `--seed`): the spine RNG and the ViZDoom engine RNG are
**unseeded** — historical runs are not bit-reproducible.

`agent.py --seed N` (added 2026-11-04) fixes both:
- `DoomGame.set_seed(N)` — the ViZDoom 1.3.x engine RNG (1.2.x tutorials call
  it `set_doom_seed`; that name does not exist in the 1.3.2 wheel).
- `ReflexSpine(rng=random.Random(N))` — the spine's decision RNG.

**Verified determinism:** two runs of
`agent.py --episodes 1 --scenario deadly_corridor --skill 1 --seed 42`
produced byte-identical episode lines (`kills=2 deaths=0 health=50
tics=603 decisions=603 stances={'navigate': 480, 'aggressive': 123}`), and the
same for take_cover `--seed 7`.

Other scripts already carry their own seeds (not re-verified here):
`collect_data.py --seed 7` (default), `collect_dagger.py --seed 21`,
`train_lora.py --seed` (fixed `SEED` + shard offset), `eval_pick.py`
(`random.seed(99)`), `trace_viz.py` (fixed `seed=100+ep`).

## 5. Reproducing the README scoreboard

The headline numbers (R9: 41 kills / 2 deaths per 10 episodes, etc.) were
produced by **trained model spines whose GGUF files and training corpora are
not in this repo** (corpora are gitignored; the champion GGUF lives off-repo).
From a fresh clone you can reproduce:

- the **heuristic-spine** rows (`--spine heur`, the default) — same command
  shape as §3, `--episodes 10 --scenario deadly_corridor --skill 1`. Episode
  variance is large (±4–5 kills), so expect the same *range*, not the same
  totals, and fix `--seed` when comparing policies;
- the corpus-collection step that regenerates training data:
  `../.venv/bin/python collect_data.py --episodes 300 --epsilon 0.15 \
   --scenario deadly_corridor --skill 1 --seed 7 --out ../corpora/corpus.jsonl`
  (**unverified** in this audit — long-running; command taken from
  SETUP-LINUX.md).

The full retrain chain (`train_lora.py` -> `merge_loras.py` -> `bake_r6.py` ->
`convert_hf_to_gguf.py` -> `llama-quantize`) and the llama.cpp spine/brain
servers (`JEVDOOM_SPINE_URL` / `JEVDOOM_BRAIN_URL`, `--spine model`) require
GPUs + multi-GB GGUF downloads and were **not re-verified** in this audit;
see `SETUP-LINUX.md` for the runbook.

## 6. Known drift

- README results predate the `--seed` flag and the `take_cover` crash fix
  (`env._update_nav` returned a 2-tuple for non-corridor scenarios, crashing
  `agent.py --scenario take_cover` on `55a1020`; fixed 2026-11-04). Numbers
  from before that fix are not bit-reproducible; the seed knob is for
  comparisons from here on.

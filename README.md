# jev-doom — a nervous system + a brain

Dual-process Doom agent, modeled on the Jev "System One" idea
([TypeSafe AI](https://typesafe.ai/blog/introducing-system-one-models-and-jev)):

- **Spinal cord (System One)** — fast, typed, probabilistic reflexes at ~17 Hz.
  Unstructured world state in → typed action distribution out, with calibrated
  confidence. No text, no hallucination. Day 1: hand-coded reflex policy behind
  a Jev-shaped `decide()` interface; drop in the real Jev API or a
  grammar-constrained local model by replacing one method.
- **Brain (System Two)** — slow, deliberative LLM loop at ~0.2 Hz. Reads a
  compressed situation summary, writes `{stance, intent}` goals to the
  blackboard that the spine conditions on. `RuleBrain` fallback is always
  available; `LLMBrain` speaks to any OpenAI-compatible endpoint.
- **Escalation arc** — when the spine's top probability < 0.45 it interrupts
  the brain for an instant re-think (startle → think).

```
world --17Hz--> SPINE (typed reflex decisions) --> body
  |                                          ^
  +--0.2Hz--> BRAIN (stance/intent) ---------+
  escalation: confidence < 0.45 -> interrupt brain
```

## Layout

| file | role |
|---|---|
| `src/env.py` | ViZDoom 1.3.x wrapper → typed `WorldState` (objects + labels, player frame) |
| `src/spine.py` | System One: `decide(world, goal) -> Decision(action, probs, confidence, hold_tics)` |
| `src/brain.py` | System Two: `think(world) -> {stance, intent}` (rule or LLM) |
| `src/agent.py` | the organism: loops, escalation, episode stats |

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install vizdoom
cd src && ../.venv/bin/python agent.py --episodes 5 --record
```

Point the brain at any OpenAI-compatible endpoint:

```bash
export JEVDOOM_BRAIN_URL=http://127.0.0.1:8080/v1
export JEVDOOM_BRAIN_KEY=***
export JEVDOOM_BRAIN_MODEL=strata
../.venv/bin/python agent.py --episodes 5
```

## ViZDoom 1.3.x API gotchas (learned the hard way)

- Class is `DoomGame`; mode is `Mode.PLAYER` (not `PLAYING`); buttons via
  `get_available_buttons()`.
- `GameState.objects` / `.labels` / `.number` are **properties**, not methods.
- Object categories are capitalized (`'Monster'`, `'Health'`, `'Ammo'`).
- Entity info: `set_objects_info_enabled(True)`; screen labels need a real
  `set_screen_resolution(...)` (default renders nothing → empty labels).
- `set_episode_timeout` is in **tics** (350 = 10 s), not seconds.
- Doom turn buttons have a **7-tic windup** then 3.5°/tic — a 2-tic turn press
  rotates exactly 0°. Hold turns ≥ 8 tics, proportional to bearing error.
- Recording: `new_episode(recording_file_path=...)` after `init()`
  (no `set_recording_path` in 1.3.x).
- Re-init between episodes: `close()` then `init()`.

## Status / roadmap

- [x] take_cover env, typed state, reflex spine, rule brain — 3 kills/5 eps vs 0 for random
- [x] .lmp demo recording
- [ ] wire real LLM brain (fleet endpoint)
- [ ] Jev API spine (early-access key) or llama.cpp grammar-constrained local spine
- [ ] deadly_corridor scenario
- [ ] render .lmp → video for demos

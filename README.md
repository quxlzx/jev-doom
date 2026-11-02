# jev-doom — a nervous system + a brain

Dual-process Doom agent, modeled on the Jev "System One" idea
([TypeSafe AI](https://typesafe.ai/blog/introducing-system-one-models-and-jev)):
a fast typed reflex layer that acts at game speed, and a slow LLM that only
sets goals.

```
world --~17Hz--> SPINE (typed reflex decisions) --> body
  |                                          ^
  +--~0.2Hz-> BRAIN (stance/intent) ---------+
  escalation: low spine confidence -> interrupt brain
```

- **Spinal cord (System One)** — `decide(world_state, goal) ->
  Decision(action, probs, confidence, hold_tics)`. Two backends:
  - `ReflexSpine`: hand-coded reflex policy (target-lock hysteresis,
    wall-hug/USE sweeps, strafe-dodge, shoot-while-retreating).
  - `ModelSpine`: the heuristic proposes the top-3 motor programs; a
    grammar-constrained Qwen3-1.7B (llama.cpp) picks `[1|2|3]`. Confidence is
    real: softmax over grammar-masked top-logprobs; below a floor the spine
    defers to the heuristic. Small models given free-form action choice latch
    onto one letter — candidate-pick is the shape that works.
- **Brain (System Two)** — Qwen3-8B (or rules) reads a compressed situation
  summary every ~25 decisions and writes `{stance, intent}` to the blackboard.
  Stances: `navigate / aggressive / defensive / retreat`.
- **Navigator** — the ViZDoom sector/wall geometry is rasterized to a walkable
  grid (player-radius buffer, climbable-stair detection from floor-height
  diffs), A* routes to a waypoint chain, and the spine/brain get a compass:
  "next waypoint at bearing +30, exit 2400 units away". No pixels anywhere.
- **Training loop** — iterative DAgger: the deployed policy plays episodes,
  the heuristic labels decisions, outcome-filtering keeps engaged episodes;
  LoRA adapters train data-parallel across 3 consumer GPUs, get merged,
  baked into the base model, and quantized (this llama.cpp build dropped PEFT
  adapter GGUFs, so we bake).

## Layout

| file | role |
|---|---|
| `src/env.py` | ViZDoom 1.3.x wrapper -> typed `WorldState` + waypoint routing |
| `src/spine.py` | System One reflex policy + navigation state machine |
| `src/spine_model.py` | grammar-constrained candidate-pick spine (llama.cpp) |
| `src/brain.py` | System Two: `think(world) -> {stance, intent}` (rule or LLM) |
| `src/nav.py` | sector geometry -> walkable grid -> A* |
| `src/agent.py` | the organism: loops, escalation, episode stats |
| `src/collect_data.py` | crash-proof DAgger corpus collection |
| `train_lora.py`, `merge_loras.py`, `bake_r6.py` | training pipeline |

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install vizdoom numpy
cd src && ../.venv/bin/python agent.py --episodes 5 --record
../.venv/bin/python agent.py --episodes 10 --scenario deadly_corridor --skill 1
```

Point the spine/brain at llama.cpp endpoints:

```bash
export JEVDOOM_SPINE_URL=http://127.0.0.1:8181     # grammar-constrained pick
export JEVDOOM_BRAIN_URL=http://127.0.0.1:8182/v1  # OpenAI-compatible
export JEVDOOM_BRAIN_MODEL=qwen3-8b
../.venv/bin/python agent.py --episodes 10 --scenario deadly_corridor --spine model
```

## Results (deadly_corridor, skill 1, 10 episodes)

| policy | wins | kills | deaths |
|---|---|---|---|
| random | 0 | 0 | ~10 |
| round-2 spine (take_cover-trained) | 0 | 4 | 1 |
| reflex spine + map, session start | 0 | 0 | 0 (stuck at door) |
| reflex spine + map, session end | 0 | 17 | 2 |
| R7 spine (corridor-trained, 81k demos) | 0 | 18 | 4 |
| **R9 spine (baron-era demos, Linux, 349k)** | 0 | **41** | 2 |
| R10 spine 1.7B (passive-era demos, 162k) | 0 | 6 | 0 |
| R10 spine **4B** QLoRA (same demos) | 0 | 10 | 0 |

**Scaling lesson (R10):** 4B vs 1.7B on identical demos moved the
needle 6→10 kills — but both trailed the 1.7B R9's 41, which was
trained on *different* (more aggressive) demos. The training data,
not the spine size, is the binding constraint.

**Coupling lesson (R10+nav):** swapping the heuristic's candidate
menu under a trained spine destroys it: R9 with path-follow nav
scored 6/15 kills, and even a deadlock-only override scored 2/15.
The champion is a *pair* — (heuristic menu, trained picker) — and
they must be retrained together. The old policy's "flawed" stuck
sweeps were also its exploration: they wandered the map and found
monsters. Efficient routing found fewer fights.

take_cover (skill 4, 5 episodes): random 0 kills, heuristic+rule 3,
1.7B spine+rule 4, 1.7B spine+8B brain 6.

## Lessons (learned the hard way)

- Doom turn buttons have a **7-tic windup** then ~3.5°/tic: short taps rotate
  0°, and acceleration from standstill looks exactly like a wall unless
  forward holds are >= ~14 tics.
- Doors rise slowly and auto-close: press USE then **ride forward** through.
- A monster behind a window will hijack an aim loop forever; a
  net-displacement stuck detector forces mission-over-mirage.
- Retreating while not shooting = dying cornered.
- Wall inflation on the nav grid pinches narrow walkways shut; buffer by
  player radius against cell centers instead.
- Training on a broken policy's demos makes a better-behaving broken policy:
  a corridor-trained spine that learned the old stuck habits underperformed
  its own teacher (4 kills vs 17). Fix the teacher, then distill.
- Episode variance is huge (±4-5 kills); keep seeds fixed and N >= 10.

## Status / archived

This project is archived on a high note: the agent went from catatonic at
spawn to fighting through E1M1's full gauntlet — door, demon courtyard,
barons — with the champion (R9) at **41 kills / 2 deaths per 10 episodes**,
reaching within ~400 units of the exit teleporter. The last mile (threading
the pit-rim ledges into the exit corridor) resisted six final experiments
and remains unsolved; the honest write-up above documents why.

### Champion highlight reels

- `media/r9_best.mp4` — R9's best run: clears the door, fights the
  corridor gauntlet, works the courtyard.
- `media/r9_fight.mp4` — close-quarters: sweep-fire vs Zombiemen,
  strafe-dodging, the pit-rim standoff.

### Final scoreboard (deadly_corridor skill 1, 10 episodes unless noted)

| Agent | Wins | Kills | Deaths |
|---|---|---|---|
| Random | 0 | 0 | ~10 |
| Heuristic spine + rule brain (early) | 0 | 3 | 1 |
| + Qwen3-1.7B grammar spine (take_cover era) | 0 | 4 | 1 |
| + Qwen3-8B brain (take_cover era) | 0 | 6 | 1 |
| R2 champion (take_cover) | 0 | 30 / 20eps | — |
| R7 (first corridor-trained) | 0 | 18 | 4 |
| **R9 champion (349k baron-era demos)** | 0 | **41** | 2 |
| R10 1.7B / 4B (passive-era demos) | 0 | 6 / 10 | 0 |

Thank you to the little GPUs that could.

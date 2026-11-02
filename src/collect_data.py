"""Collect a training corpus for the Jev-type spine model.

Runs many episodes with the heuristic expert (and optionally the model spine),
logging every decision in the exact candidate-pick prompt format the spine
uses, tagged with episode outcome. Best-of-N trajectory filtering happens at
export time: keep decisions from high-reward episodes -> behavior cloning
dataset (JSONL: {"prompt": ..., "completion": "[1]"}).

Usage: python collect_data.py --episodes 500 --out corpus.jsonl
"""
from __future__ import annotations
import argparse
import json
import random

from env import DoomEnv
from spine import ReflexSpine
from spine_model import ModelSpine
from brain import RuleBrain


def run_one(env, spine, brain, rng, model_spine=None, epsilon=0.0):
    """One episode; returns list of (prompt, label_idx, meta) + stats."""
    env.reset()
    brain = brain()
    goal = brain.think(env.perceive())
    g = env.g
    samples = []
    # room-clear tracking: "room" = 256-unit map block. Track which monster
    # entity ids were SEEN while in a block, and kills credited to that block.
    # If kills >= seen (>=2), the block was CLEARED -> big weight bonus there.
    seen_ids = {}     # block -> set(entity ids)
    block_kills = {}  # block -> kills credited
    prev_kills = 0
    while not g.is_episode_finished():
        ws = env.perceive()
        if g.get_episode_time() > env.timeout:
            break
        # expert proposal
        h = spine.decide(ws, goal)
        ranked = sorted(h.probs, key=lambda a: -h.probs[a])[:3]
        # shuffle candidate order: the model must READ the candidates,
        # not parrot "1" because the expert's pick is always listed first
        shuffled = ranked[:]
        rng.shuffle(shuffled)
        # epsilon-greedy over candidates: explore 2nd/3rd options sometimes
        if epsilon > 0 and rng.random() < epsilon:
            action = rng.choice(shuffled)
        else:
            action = h.action
        # compound actions (["move_forward","attack"]): execute the combo but
        # LABEL with the component the candidate list contains (attack first:
        # it is the intentional part)
        exec_action = action
        if isinstance(action, (list, tuple)):
            for comp in reversed(action):
                if comp in shuffled:
                    action = comp
                    break
            else:
                action = action[0]
        pick = shuffled.index(action)
        # hold must match the ACTUAL action (turns need windup+rate)
        from spine import bearing_deg
        t = ws.nearest_visible_monster() or min(ws.monsters, key=lambda m: m.dist, default=None)
        hold = min(40, 8 + int(abs(bearing_deg(t)) / 3.5)) if (action.startswith("turn") and t) else 2
        # build the SAME prompt the ModelSpine would show
        if model_spine is not None:
            prompt = model_spine.build_prompt(ws, goal, h, order=shuffled)
        blk = (int(ws.px // 256), int(ws.py // 256))
        if prompt is not None:
            samples.append((prompt, pick + 1, {"stance": goal.get("stance"),
                                               "blk": blk}))
        for m in ws.monsters:
            seen_ids.setdefault(blk, set()).add(m.id)
        r, over = env.step(exec_action, tics=hold)
        # credit kills to the block the agent stood in when they happened
        try:
            k_now = g.get_game_variable(GV.KILLCOUNT)
        except Exception:
            k_now = prev_kills
        if k_now > prev_kills:
            block_kills[blk] = block_kills.get(blk, 0) + (k_now - prev_kills)
        prev_kills = k_now
        if over:
            break
    from env import GV
    try:
        kills = g.get_game_variable(GV.KILLCOUNT)
        health = g.get_game_variable(GV.HEALTH)
    except Exception:
        kills, health = 0, 0
    cleared = {b for b, ids in seen_ids.items()
               if len(ids) >= 2 and block_kills.get(b, 0) >= len(ids)}
    return samples, kills, health, cleared


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default="corpus.jsonl")
    p.add_argument("--skill", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--scenario", default="take_cover")
    a = p.parse_args()
    rng = random.Random(a.seed)
    env = DoomEnv(skill=a.skill, scenario=a.scenario)
    spine = ReflexSpine(rng=rng)
    ms = ModelSpine()  # used only for prompt construction (no HTTP)
    n_kept = 0
    with open(a.out, "w") as f:
      for ep in range(a.episodes):
        try:
            samples, kills, health, cleared = run_one(env, spine, RuleBrain, rng, ms, a.epsilon)
            died = 1 if health <= 0 else 0
            # user-shaped credit: reward kills heavily, punish death lightly,
            # bonus heavily for clearing a room of enemies
            base = 1.0 + 0.6 * kills - 0.4 * died
            for prompt, label, meta in samples:
                # outcome filter: keep exploratory picks only from engaging episodes
                if kills == 0 and label != 1:
                    continue
                blk = tuple(meta.get("blk", ()))
                weight = base * (2.5 if blk in cleared else 1.0)
                meta.pop("blk", None)
                f.write(json.dumps({"prompt": prompt,
                                    "completion": f"[{label}]",
                                    "meta": {**meta, "ep": ep, "kills": kills,
                                             "health": health, "weight": weight}}) + "\n")
            f.flush()
            n_kept += len(samples)
            if ep % 25 == 0:
                print(f"ep {ep}: samples={n_kept} last kills={kills}", flush=True)
        except Exception as e:
            print(f"ep {ep} crashed: {e.__class__.__name__}: {e} — rebuilding env", flush=True)
            try:
                env.close()
            except Exception:
                pass
            env = DoomEnv(skill=a.skill, scenario=a.scenario)
            spine = ReflexSpine(rng=rng)
            continue
    print(f"done: {n_kept} samples -> {a.out}")
    env.close()


if __name__ == "__main__":
    main()

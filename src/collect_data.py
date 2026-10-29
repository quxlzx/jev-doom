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


def run_one(env, spine, brain, rng, model_spine=None):
    """One episode; returns list of (prompt, label_idx, meta) + stats."""
    env.reset()
    brain = brain()
    goal = brain.think(env.perceive())
    g = env.g
    samples = []
    while not g.is_episode_finished():
        ws = env.perceive()
        if g.get_episode_time() > 350:
            break
        # expert proposal
        h = spine.decide(ws, goal)
        # build the SAME prompt the ModelSpine would show
        if model_spine is not None:
            prompt = model_spine.build_prompt(ws, goal, h)
        else:
            prompt = None
        ranked = sorted(h.probs, key=lambda a: -h.probs[a])[:3]
        label = ranked.index(h.action) if h.action in ranked else 0
        if prompt is not None:
            samples.append((prompt, label + 1, {"stance": goal.get("stance")}))
        r, over = env.step(h.action, tics=h.hold_tics)
        if over:
            break
    from env import GV
    try:
        kills = g.get_game_variable(GV.KILLCOUNT)
        health = g.get_game_variable(GV.HEALTH)
    except Exception:
        kills, health = 0, 0
    return samples, kills, health


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default="corpus.jsonl")
    p.add_argument("--skill", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    rng = random.Random(a.seed)
    env = DoomEnv(skill=a.skill)
    spine = ReflexSpine(rng=rng)
    ms = ModelSpine()  # used only for prompt construction (no HTTP)
    n_kept = 0
    with open(a.out, "w") as f:
        for ep in range(a.episodes):
            samples, kills, health = run_one(env, spine, RuleBrain, rng, ms)
            # trajectory quality gate: keep episodes that engaged
            weight = 1.0 if kills > 0 else 0.25
            for prompt, label, meta in samples:
                f.write(json.dumps({"prompt": prompt,
                                    "completion": f"[{label}]",
                                    "meta": {**meta, "ep": ep, "kills": kills,
                                             "health": health, "weight": weight}}) + "\n")
            n_kept += len(samples)
            if ep % 50 == 0:
                print(f"ep {ep}: samples={n_kept} last kills={kills}")
    print(f"done: {n_kept} samples -> {a.out}")
    env.close()


if __name__ == "__main__":
    main()

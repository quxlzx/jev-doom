"""Round 3 DAgger corpus: collect from the TRAINED model's own gameplay.

For every decision: record the model's pick and the heuristic's pick.
Outcome credit over the next ~10 decisions:
  - model deviated AND damage followed  -> label = heuristic pick (correction)
  - model deviated AND kill/no-damage   -> label = model pick (reinforcement)
  - agreement                           -> label = that pick
Run on the desktop next to the LoRA-loaded spine server.
"""
from __future__ import annotations
import argparse
import json
import random

from env import DoomEnv, GV
from spine import ReflexSpine
from spine_model import ModelSpine
from brain import RuleBrain

WINDOW = 10  # decisions of outcome credit after each choice


def run_one(env, ms, hs, rng, base_url):
    import urllib.request
    env.reset()
    brain = RuleBrain()
    goal = brain.think(env.perceive())
    g = env.g
    samples = []   # (prompt, model_pos, heur_pos)
    deltas = []    # (hits_delta, kills_delta) per decision
    prev_hits, prev_kills = 0.0, 0.0
    while not g.is_episode_finished():
        ws = env.perceive()
        if g.get_episode_time() > 350:
            break
        h = hs.decide(ws, goal)
        ranked = sorted(h.probs, key=lambda a: -h.probs[a])[:3]
        rng.shuffle(ranked)
        prompt = ms.build_prompt(ws, goal, h, order=ranked)
        # model pick (trained spine over HTTP)
        body = {"prompt": prompt, "temperature": 0.0, "n_predict": 3}
        try:
            req = urllib.request.Request(base_url + "/completion",
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            out = json.loads(urllib.request.urlopen(req, timeout=5).read())
            digit = next((c for c in out.get("content", "") if c in "123"), "1")
        except Exception:
            digit = str(ranked.index(h.action) + 1)
        model_pos = int(digit) - 1
        heur_pos = ranked.index(h.action)
        action = ranked[min(model_pos, 2)]
        t = ws.nearest_visible_monster() or min(ws.monsters, key=lambda m: m.dist, default=None)
        hold = min(40, 8 + int(abs(__import__("spine").bearing_deg(t)) / 3.5)) if (action.startswith("turn") and t) else 2
        env.step(action, tics=hold)
        hits = g.get_game_variable(GV.HITS_TAKEN)
        kills = g.get_game_variable(GV.KILLCOUNT)
        deltas.append((hits - prev_hits, kills - prev_kills))
        prev_hits, prev_kills = hits, kills
        samples.append((prompt, model_pos, heur_pos))
        if g.is_episode_finished():
            break
    return samples, deltas, prev_kills


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--out", default=r"D:\doomsrv\corpus4.jsonl")
    p.add_argument("--spine", default="http://127.0.0.1:8181")
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--expert-anchor", action="store_true",
                   help="label = expert pick on every deviation (classic DAgger, no self-reinforcement)")
    a = p.parse_args()
    rng = random.Random(a.seed)
    env = DoomEnv(skill=4)
    hs = ReflexSpine(rng=rng)
    ms = ModelSpine(base_url=a.spine)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for ep in range(a.episodes):
            samples, deltas, kills = run_one(env, ms, hs, rng, a.spine)
            for i, (prompt, mpos, hpos) in enumerate(samples):
                window = deltas[i + 1:i + 1 + WINDOW]
                dmg = any(d[0] > 0 for d in window)
                kill = any(d[1] > 0 for d in window)
                if a.expert_anchor and mpos != hpos:
                    label = hpos          # classic DAgger: expert anchors every deviation
                elif mpos != hpos and dmg and not kill:
                    label = hpos          # model eroded: correct toward expert
                else:
                    label = mpos          # reinforce the trained policy
                f.write(json.dumps({"prompt": prompt, "completion": f"[{label+1}]",
                                    "meta": {"ep": ep, "kills": kills,
                                             "agree": mpos == hpos,
                                             "corrected": label != mpos}}) + "\n")
                n += 1
            if ep % 100 == 0:
                print(f"ep {ep}: samples={n}")
    print(f"done: {n} samples -> {a.out}")
    env.close()


if __name__ == "__main__":
    main()

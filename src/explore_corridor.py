"""Explore deadly_corridor with a long budget until the exit is found.
Logs the full trajectory of the winning episode for waypoint extraction.
"""
import json
import sys

from env import DoomEnv
from spine import ReflexSpine
from brain import RuleBrain

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 20

env = DoomEnv(skill=3, scenario="deadly_corridor")
env.g.set_episode_timeout(6000)  # 3 minutes of exploration
env.timeout = 6000
brain = RuleBrain()

for ep in range(EPISODES):
    spine = ReflexSpine(rng=__import__("random").Random(1000 + ep * 37))
    env.reset()
    goal = brain.think(env.perceive())
    g = env.g
    path = []
    while not g.is_episode_finished():
        if g.get_episode_time() > env.timeout:
            break
        ws = env.perceive()
        path.append((round(ws.px), round(ws.py), round(ws.health)))
        d = spine.decide(ws, goal)
        env.step(d.action, tics=d.hold_tics)
        if ws.tic % 30 == 0:
            goal = brain.think(ws)
    alive = g.get_game_variable(__import__("env").GV.HEALTH) > 0
    ended_early = g.get_episode_time() < env.timeout - 5
    print(f"ep {ep}: time={g.get_episode_time()} alive={alive} early={ended_early} "
          f"pathlen={len(path)} last={path[-1] if path else None}")
    if alive and ended_early:
        json.dump(path, open("corridor_path.json", "w"))
        print("WIN — path saved, exit at", path[-1])
        break
env.close()

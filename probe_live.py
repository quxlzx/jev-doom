"""Live probe: one partial episode, timing + decision reasons per step."""
import time
from env import DoomEnv
from spine_model import ModelSpine
from brain import RuleBrain

env = DoomEnv(skill=4)
ms = ModelSpine()
b = RuleBrain()
env.reset()
goal = b.think(env.perceive())
g = env.g
for i in range(25):
    if g.is_episode_finished():
        print("EPISODE OVER")
        break
    ws = env.perceive()
    t0 = time.time()
    d = ms.decide(ws, goal)
    dt = (time.time() - t0) * 1000
    print(f"i={i:2d} {d.action:12s} {dt:5.0f}ms hp={ws.health:3.0f} | {d.reason[:70]}")
    env.step(d.action, tics=d.hold_tics)
env.close()

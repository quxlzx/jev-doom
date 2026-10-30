"""Desktop probe: model spine decisions in deadly_corridor."""
import vizdoom
from env import DoomEnv
from spine_model import ModelSpine
from brain import RuleBrain

env = DoomEnv(skill=4, scenario="deadly_corridor")
ms = ModelSpine()
br = RuleBrain()
env.reset()
goal = br.think(env.perceive())
g = env.g
i = 0
while not g.is_episode_finished() and i < 60:
    ws = env.perceive()
    d = ms.decide(ws, goal)
    nb = round(ws.nav_bearing) if ws.nav_bearing is not None else None
    print(f"i={i} pos=({ws.px:.0f},{ws.py:.0f}) nav={nb} act={d.action:12s} {d.reason[:60]}")
    env.step(d.action, tics=d.hold_tics)
    if ws.tic % 25 == 0:
        goal = br.think(ws)
    i += 1
print("END time", g.get_episode_time(), "hp", g.get_game_variable(vizdoom.GameVariable.HEALTH))
env.close()

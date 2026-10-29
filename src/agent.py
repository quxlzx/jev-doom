"""The whole organism: spinal cord at ~17Hz, brain at ~0.2Hz, escalation arc.

  world --17Hz--> SPINE (typed reflex decisions) --> body
    |                                          ^
    +--0.2Hz--> BRAIN (stance/intent) ---------+
    escalation: spine confidence < 0.45 -> interrupt brain for an instant re-think
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field

from env import DoomEnv
from spine import ReflexSpine
from brain import make_brain, summarize

SPINE_TICS = 2          # 35 tics/s / 2 = ~17 decisions/s
BRAIN_EVERY = 175       # every 5 game-seconds
ESCALATE_EVERY = 25     # min tics between brain interrupts


@dataclass
class EpisodeStats:
    reward: float = 0.0
    tics: int = 0
    decisions: int = 0
    escalations: int = 0
    kills: float = 0.0
    deaths: float = 0.0
    health: float = 0.0
    stances: dict = field(default_factory=dict)
    spine_log: list[str] = field(default_factory=list)


def run_episode(env: DoomEnv, spine: ReflexSpine, brain, seed: int = 0,
                verbose: bool = False) -> EpisodeStats:
    rec = None
    if env.record_dir:
        import time as _t
        rec = f"{env.record_dir}/ep_{int(_t.time())}.lmp"
    env.reset(rec)
    st = EpisodeStats()
    goal = brain.think(env.perceive())
    last_brain_tic = 0
    last_escalation_tic = 0

    while not env.g.is_episode_finished() and st.tics < 420:
        ws = env.perceive()
        st.tics = ws.tic

        # --- System Two tick (slow)
        if ws.tic - last_brain_tic >= BRAIN_EVERY:
            goal = brain.think(ws)
            last_brain_tic = ws.tic
            if verbose:
                print(f"  [brain  t={ws.tic:4d}] {goal['stance']:10s} {goal.get('intent','')}")

        # --- System One tick (fast)
        d = spine.decide(ws, goal)
        st.decisions += 1
        st.spine_log.append(d.reason)
        st.spine_log = st.spine_log[-8:]

        # --- escalation arc: low confidence -> wake the brain now
        if d.low_confidence and ws.tic - last_escalation_tic > ESCALATE_EVERY:
            goal = brain.think(ws)
            last_escalation_tic = ws.tic
            st.escalations += 1
            if verbose:
                print(f"  [escalate t={ws.tic:4d}] conf={d.confidence:.2f} -> {goal['stance']}")

        st.stances[goal.get("stance", "?")] = st.stances.get(goal.get("stance", "?"), 0) + 1

        if verbose and st.decisions % 120 == 0:
            print(f"  [spine  t={ws.tic:4d}] {d.action:10s} conf={d.confidence:.2f} {d.reason}")

        tics = d.hold_tics
        r, over = env.step(d.action, tics=tics)
        st.reward += r
        if over:
            break

    from env import GV
    try:
        st.kills = env.g.get_game_variable(GV.KILLCOUNT)
        st.deaths = env.g.get_game_variable(GV.DEATHCOUNT)
        st.health = env.g.get_game_variable(GV.HEALTH)
    except Exception:
        pass
    return st


def main(episodes: int = 1, skill: int = 4, verbose: bool = True, record: bool = False):
    env = DoomEnv(skill=skill, window=False, record_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "replays") if record else None)
    spine = ReflexSpine()
    brain = make_brain(prefer_llm=True)
    print(f"brain: {type(brain).__name__} | spine: {type(spine).__name__} | skill={skill}")
    results = []
    for ep in range(episodes):
        t0 = time.time()
        st = run_episode(env, spine, brain, seed=ep, verbose=verbose)
        dt = time.time() - t0
        results.append((st, dt))
        if verbose:
            print(f"episode {ep}: reward={st.reward:.1f} kills={st.kills:.0f} "
                  f"deaths={st.deaths:.0f} health={st.health:.0f} tics={st.tics} "
                  f"decisions={st.decisions} escalations={st.escalations} "
                  f"stances={st.stances} wall={dt:.1f}s")
    env.close()
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--skill", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--record", action="store_true", help="record .lmp demos to replays/")
    a = p.parse_args()
    main(episodes=a.episodes, skill=a.skill, verbose=not a.quiet, record=a.record)

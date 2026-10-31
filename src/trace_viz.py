"""Trace agent runs (position, gaze, stance, kills) and render map overlays.

Shows WHERE the agent goes, WHERE it looks, and where it gets stuck —
the diagnostic the numbers can't give.

Usage: python trace_viz.py --episodes 3 --skill 1 --out-dir ../traces
"""
from __future__ import annotations
import argparse, json, math, os, random
from env import DoomEnv
from spine import ReflexSpine
from brain import RuleBrain, make_brain
import vizdoom as zd

GV = zd.GameVariable


def run_traced(ep, skill, seed, spine_kind="heur"):
    env = DoomEnv(skill=skill, scenario="deadly_corridor")
    if spine_kind == "model":
        from spine_model import ModelSpine
        spine = ModelSpine()
    else:
        spine = ReflexSpine(rng=random.Random(seed))
    brain = make_brain(prefer_llm=False) or RuleBrain()
    env.reset()
    goal = brain.think(env.perceive())
    trace = []
    prev_k = 0
    i = 0
    last_state = None
    while not env.g.is_episode_finished() and i < 400:
        ws = env.perceive()
        if i % 10 == 0:
            last_state = env.g.get_state()
        d = spine.decide(ws, goal)
        k = env.g.get_game_variable(GV.KILLCOUNT)
        trace.append({
            "i": i, "t": env.g.get_episode_time(),
            "x": ws.px, "y": ws.py, "ang": ws.angle,
            "stance": goal.get("stance", "?"), "act": d.action,
            "reason": d.reason[:60],
            "hp": ws.health, "kill": k > prev_k,
        })
        prev_k = k
        env.step(d.action, tics=d.hold_tics)
        if ws.tic % 25 == 0:
            goal = brain.think(ws)
        i += 1
    walls = []
    if env.nav and last_state is not None:
        # re-extract raw wall segments for the overlay
        for s in last_state.sectors:
            for ln in s.lines:
                if ln.is_blocking:
                    walls.append((ln.x1, ln.y1, ln.x2, ln.y2))
    env.close()
    return trace, walls


def render(trace, walls, out, ep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_facecolor("#0a0a0f")
    fig.patch.set_facecolor("#0a0a0f")
    lc = LineCollection([[(a, b), (c, d)] for a, b, c, d in walls],
                        colors="#3a3a4a", linewidths=0.7)
    ax.add_collection(lc)

    cmap = LinearSegmentedColormap.from_list("t", ["#00e5ff", "#76ff03", "#ffea00", "#ff1744"])
    n = len(trace)
    pts = [(p["x"], p["y"]) for p in trace]
    segs = [(pts[i], pts[i + 1]) for i in range(n - 1)]
    cols = [cmap(i / max(1, n - 1)) for i in range(n - 1)]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=2.2))

    # gaze cones every ~10 decisions
    for i in range(0, n, max(1, n // 25)):
        p = trace[i]
        r = math.radians(p["ang"])
        L = 90
        ax.annotate("", xy=(p["x"] + L * math.cos(r), p["y"] + L * math.sin(r)),
                    xytext=(p["x"], p["y"]),
                    arrowprops=dict(arrowstyle="->", color=cmap(i / max(1, n - 1)),
                                    alpha=0.6, linewidth=1.2))
        # cone edges (±30 deg FOV)
        for s in (-0.52, 0.52):
            ax.plot([p["x"], p["x"] + 60 * math.cos(r + s)],
                    [p["y"], p["y"] + 60 * math.sin(r + s)],
                    color=cmap(i / max(1, n - 1)), alpha=0.25, linewidth=0.8)

    # start / end / kills / oscillation-stuck zones
    ax.plot(*pts[0], "o", color="white", ms=9, label="start")
    ax.plot(*pts[-1], "s", color="white", ms=9, mfc="none", label="end")
    for p in trace:
        if p["kill"]:
            ax.plot(p["x"], p["y"], "X", color="#ff1744", ms=11)
        if "sweep" in p["reason"] or "stuck" in p["reason"]:
            ax.plot(p["x"], p["y"], ".", color="#ff9100", ms=4, alpha=0.7)

    # exit marker (fixed map)
    ax.plot(1720, -1190, "*", color="#00e676", ms=18, label="exit")
    ax.set_title(f"ep {ep}: path (cyan->red = time), arrows = gaze, X = kill, orange dust = stuck-sweeps",
                 color="w", fontsize=11)
    ax.legend(loc="upper left")
    ax.autoscale()
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color("#444")
    ax.tick_params(colors="#888")
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="#0a0a0f")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--skill", type=int, default=1)
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "traces"))
    p.add_argument("--spine", choices=["heur", "model"], default="heur")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for ep in range(a.episodes):
        trace, walls = run_traced(ep, a.skill, seed=100 + ep, spine_kind=a.spine)
        with open(os.path.join(a.out_dir, f"trace_ep{ep}.json"), "w") as f:
            json.dump(trace, f)
        render(trace, walls, os.path.join(a.out_dir, f"map_ep{ep}.png"), ep)
        kills = sum(1 for t in trace if t["kill"])
        stuck = sum(1 for t in trace if "stuck" in t["reason"] or "sweep" in t["reason"])
        print(f"ep {ep}: {len(trace)} decisions, kills={kills}, stuck-sweeps={stuck}, "
              f"end=({trace[-1]['x']:.0f},{trace[-1]['y']:.0f})")


if __name__ == "__main__":
    main()

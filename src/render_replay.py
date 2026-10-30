"""Render a recorded .lmp demo to mp4 by replaying it and capturing frames.

Usage: python render_replay.py <demo.lmp> <out.mp4> [--fps 35]
Replays with the same scenario config the demo was recorded under.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import vizdoom as zd


def render(lmp: str, out: str, fps: int = 35) -> None:
    import imageio_ffmpeg
    scenario = os.environ.get("SCENARIO", "take_cover")
    cfg = {"take_cover": "take_cover.cfg", "deadly_corridor": "deadly_corridor.cfg"}[scenario]
    game = zd.DoomGame()
    game.set_doom_scenario_path(os.path.join(zd.scenarios_path, cfg))
    game.set_doom_map("map01")
    game.set_doom_skill(4)
    game.set_screen_resolution(zd.RES_640X480)
    game.set_screen_format(zd.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.set_mode(zd.Mode.SPECTATOR)
    game.set_ticrate(fps)
    game.init()
    game.replay_episode(lmp)

    writer = imageio_ffmpeg.write_frames(
        out, (640, 480), fps=fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p"])
    writer.send(None)
    frames = 0
    while not game.is_episode_finished():
        st = game.get_state()
        buf = st.screen_buffer  # (H, W, 3) for RGB24
        if buf.ndim == 2:  # channel-first fallback
            buf = np.transpose(buf.reshape(3, 480, 640), (1, 2, 0))
        writer.send(np.ascontiguousarray(buf))
        game.advance_action()
        frames += 1
    writer.close()
    game.close()
    print(f"rendered {frames} frames -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("lmp")
    p.add_argument("out")
    p.add_argument("--fps", type=int, default=35)
    p.add_argument("--scenario", default=None)
    a = p.parse_args()
    if a.scenario:
        os.environ["SCENARIO"] = a.scenario
    render(a.lmp, a.out, a.fps)

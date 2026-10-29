"""ViZDoom wrapper -> typed WorldState. The senses of the nervous system."""
from __future__ import annotations
import math
import os
from dataclasses import dataclass, field

import vizdoom as zd

GV, B = zd.GameVariable, zd.Button

ACTIONS = ["move_left", "move_right", "move_back", "move_forward", "turn_left", "turn_right", "attack"]
_BUTTONS = {
    "move_left": B.MOVE_LEFT,
    "move_right": B.MOVE_RIGHT,
    "move_back": B.MOVE_BACKWARD,
    "move_forward": B.MOVE_FORWARD,
    "turn_left": B.TURN_LEFT,
    "turn_right": B.TURN_RIGHT,
    "attack": B.ATTACK,
}

SCENARIO = os.path.join(zd.scenarios_path, "take_cover.cfg")


@dataclass
class Entity:
    id: int = -1
    name: str = ""
    category: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0

    @property
    def dist(self) -> float:
        return math.hypot(self.x, self.y)  # relative coords: player at origin


@dataclass
class WorldState:
    tic: int
    health: float
    armor: float
    ammo: float
    attack_ready: bool
    angle: float
    kills: float
    deaths: float
    hits_taken: float
    monsters: list[Entity] = field(default_factory=list)      # world-space, player-relative
    visible: list[Entity] = field(default_factory=list)       # on screen (labels), player-relative bearing
    pickups: list[Entity] = field(default_factory=list)

    def nearest_visible_monster(self) -> Entity | None:
        cands = [m for m in self.visible if m.category == "monster"]
        return min(cands, key=lambda m: m.dist) if cands else None


def _to_player_frame(wx: float, wy: float, wz: float, px: float, py: float, angle_deg: float) -> Entity:
    """World coords -> player frame: x=forward, y=left, bearing in degrees."""
    dx, dy = wx - px, wy - py
    a = math.radians(angle_deg)
    fwd = dx * math.cos(a) + dy * math.sin(a)
    left = -dx * math.sin(a) + dy * math.cos(a)
    return Entity(x=fwd, y=left, z=wz)


class DoomEnv:
    def __init__(self, skill: int = 4, window: bool = False, record_dir: str | None = None):
        self.g = zd.DoomGame()
        self.g.set_doom_scenario_path(SCENARIO)
        self.g.set_doom_map("map01")
        self.g.set_doom_skill(skill)
        self.g.clear_available_buttons()
        for name in ACTIONS:
            self.g.add_available_button(_BUTTONS[name])
        self.g.clear_available_game_variables()
        self._gv = [GV.HEALTH, GV.ARMOR, GV.SELECTED_WEAPON_AMMO, GV.KILLCOUNT, GV.DEATHCOUNT,
                    GV.HITS_TAKEN, GV.ANGLE, GV.POSITION_X, GV.POSITION_Y, GV.ATTACK_READY]
        for v in self._gv:
            self.g.add_available_game_variable(v)
        self._gi = {v: i for i, v in enumerate(self._gv)}
        self.g.set_objects_info_enabled(True)
        self.g.set_labels_buffer_enabled(True)
        self.g.set_screen_resolution(zd.RES_160X120)  # cheap render so labels populate
        self.g.set_episode_timeout(350)  # 350 tics = 10s, like the official take_cover
        self.g.set_episode_start_time(1)
        self.g.set_window_visible(window)
        self.g.set_mode(zd.Mode.PLAYER)
        self.g.set_ticrate(35)
        self.record_dir = record_dir
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        self._prev_hits = 0.0

    def reset(self, recording_file: str | None = None) -> None:
        try:
            self.g.close()
        except Exception:
            pass
        self.g.init()
        if recording_file:
            self.g.new_episode(recording_file_path=recording_file)
        self._prev_hits = 0.0

    def perceive(self) -> WorldState:
        g = self.g
        st = g.get_state()
        gv = g.get_game_variable
        px = gv(GV.POSITION_X)
        py = gv(GV.POSITION_Y)
        ang = gv(GV.ANGLE)
        monsters, pickups = [], []
        for o in st.objects:
            if o.category.lower() == "monster":
                e = _to_player_frame(o.position_x, o.position_y, o.position_z, px, py, ang)
                e.id, e.name, e.category, e.vx, e.vy = o.id, o.name, o.category.lower(), o.velocity_x, o.velocity_y
                monsters.append(e)
            elif o.category.lower() in ("health", "armor", "ammo", "weapon", "countable_item", "inventory", "powerup"):
                e = _to_player_frame(o.position_x, o.position_y, o.position_z, px, py, ang)
                e.id, e.name, e.category = o.id, o.name, o.category.lower()
                pickups.append(e)
        visible = []
        for lb in st.labels:
            if lb.object_category.lower() in ("monster", "health", "armor", "ammo", "weapon", "countable_item", "inventory"):
                e = _to_player_frame(lb.object_position_x, lb.object_position_y,
                                     lb.object_position_z, px, py, ang)
                e.name, e.category = lb.object_name, lb.object_category.lower()
                visible.append(e)
        return WorldState(
            tic=st.number,
            health=gv(GV.HEALTH),
            armor=gv(GV.ARMOR),
            ammo=gv(GV.SELECTED_WEAPON_AMMO),
            attack_ready=gv(GV.ATTACK_READY) > 0.5,
            angle=ang,
            kills=gv(GV.KILLCOUNT),
            deaths=gv(GV.DEATHCOUNT),
            hits_taken=gv(GV.HITS_TAKEN),
            monsters=monsters,
            visible=visible,
            pickups=pickups,
        )

    def step(self, action: str, tics: int = 2) -> tuple[float, bool]:
        """Execute one named action for `tics` tics. Returns (reward, episode_over)."""
        vec = [1.0 if a == action else 0.0 for a in ACTIONS]
        r = self.g.make_action(vec, tics)
        return r, self.g.is_episode_finished()

    def damage_spike(self, ws: WorldState) -> bool:
        spiked = ws.hits_taken > self._prev_hits
        self._prev_hits = ws.hits_taken
        return spiked

    def close(self) -> None:
        self.g.close()

"""System One — the spinal cord.

Interface mirrors a Jev call: unstructured world state in, typed probabilistic
decisions out (every action gets a calibrated probability, no text, no
hallucination). Day 1 ships a hand-built reflex policy behind that exact
interface; swapping in the real Jev API or a grammar-constrained local model
means replacing `decide()` only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass

from env import ACTIONS


@dataclass
class Decision:
    action: str
    probs: dict[str, float]   # calibrated-ish distribution over ACTIONS
    confidence: float         # top prob
    reason: str               # for the brain's reading, not executed
    hold_tics: int = 2        # how long to hold the button (turns need windup+rate)

    @property
    def low_confidence(self) -> bool:
        return self.confidence < 0.45


def bearing_deg(e) -> float:
    """Bearing to a player-frame entity (x=forward, y=left), in degrees."""
    return math.degrees(math.atan2(e.y, e.x))


class ReflexSpine:
    """Hand-coded System One. Fast (<1ms), typed, confidence-scored."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self._lock_id = None       # orientation target hysteresis: stop target-flip
        self._lock_age = 999
        self._last_pos = None      # blocked detection for wall-following explore
        self._pos_hist = []        # net-displacement stuck detector (wall sliding)
        self._spawn = None         # first-seen position (spawn room)
        self._blocked_streak = 0
        self._last_nav_b = None    # nav bearing damping
        self._turn_shrink = 12

    def decide(self, ws, goal: dict) -> Decision:
        probs = {a: 0.0 for a in ACTIONS}
        stance = goal.get("stance", "aggressive")
        # global net-displacement stuck detector: a monster behind a window
        # can hijack the aim branch forever while the exit stays behind a door
        if self._spawn is None:
            self._spawn = (ws.px, ws.py)  # spawn area has a teleporter: never USE here
        near_spawn = (ws.px - self._spawn[0])**2 + (ws.py - self._spawn[1])**2 < 90000
        self._pos_hist.append((ws.px, ws.py))
        if len(self._pos_hist) > 9:
            self._pos_hist.pop(0)
        self._stuck = (len(self._pos_hist) >= 9 and
                       (ws.px - self._pos_hist[0][0])**2 +
                       (ws.py - self._pos_hist[0][1])**2 < 6400)

        # glass/window targets: visible but bullets bounce off windows.
        # If firing at one target yields no kill for too long, blacklist it.
        self._glass_lock = getattr(self, "_glass_lock", {})
        for k in list(self._glass_lock):
            self._glass_lock[k] -= 1
            if self._glass_lock[k] <= 0:
                del self._glass_lock[k]
        self._fire_streak = getattr(self, "_fire_streak", [None, 0])
        target = ws.nearest_visible_monster()
        while target is not None and target.id in self._glass_lock:
            others = [m for m in ws.monsters
                      if m.id != target.id and m.id not in self._glass_lock]
            target = min(others, key=lambda m: m.dist, default=None)
        if (self._stuck and ws.nav_bearing is not None and
                (target is None or target.dist > 250)):
            # mission > mirage: a FAR unreachable monster (window/ledge) must
            # not veto the exit mission. But a monster within 250 units may be
            # the physical blocker — monsters are solid; shoot through it.
            stance = "navigate"
            target = None
        hold = 2

        # --- reflex: flinch on damage even if the brain is asleep
        if ws.health < 35 and stance != "retreat":
            stance = "defensive"

        def turn_hold(bearing: float) -> int:
            # measured physics: 7-tic windup THEN ~4.5deg/tic. The old 3.5/tic
            # no-windup model overshot ~2x, causing acquire oscillation.
            return max(9, min(48, 7 + int(abs(bearing) / 4.5) + 2))

        if target is not None:
            b = bearing_deg(target)
            dist = target.dist
            aligned = abs(b) < 6.0

            if aligned and ws.attack_ready and dist < 1600 and ws.ammo > 0:
                # glass detection: same target, many shots, no progress
                if self._fire_streak[0] == target.id:
                    self._fire_streak[1] += 1
                else:
                    self._fire_streak = [target.id, 1]
                if self._fire_streak[1] >= 18 and dist > 150:
                    self._glass_lock[target.id] = 40
                    self._fire_streak = [None, 0]
                    # treat as no target this decision: navigate instead
                    probs["move_forward"] = 0.8
                    reason = f"glass? {target.name} d={dist:.0f}: 18 shots no kill, ignore"
                    target = None
                elif dist < 500 and self.rng.random() < 0.3:
                    # strafe-dodge while trading shots: barons telegraph fireballs
                    side = "move_left" if self.rng.random() < 0.5 else "move_right"
                    probs[side] = 0.8; probs["attack"] = 0.5
                    hold = 8
                    reason = f"strafe vs {target.name} d={dist:.0f}"
                else:
                    probs["attack"] = 0.85 if stance == "aggressive" else 0.6
                    probs["move_left"] = 0.07
                    probs["move_right"] = 0.07
                    reason = f"fire on {target.name} d={dist:.0f}"
            elif ws.ammo <= 0 and dist < 400:
                # dry pistol: disengage; the nav pickup-detour hunts ammo
                probs["move_back"] = 0.85
                hold = 12
                reason = f"DRY pistol vs {target.name} d={dist:.0f}: back off"
            else:
                turn = "turn_left" if b > 0 else "turn_right"
                probs[turn] = 0.9
                hold = turn_hold(b)
                reason = f"acquire {target.name} bearing={b:.0f} d={dist:.0f}"
        else:
            # no monster on screen: use the world map to orient, then close
            # distance. Navigation is stance-INDEPENDENT: the brain's mood
            # must not drop us into legacy wall-hugging (route exists: walk it)
            if ws.nav_bearing is not None:
                b = ws.nav_bearing
                # waypoint behind: REVERSE, don't turn around. A 180-deg
                # pivot overshoots and the bearing flips -> wavering loop
                # (observed at the pit west stairs: 60+ wasted decisions).
                if abs(b) > 135:
                    # reverse, but if reversing makes no progress, commit a
                    # full turn (observed: 600 decisions reversing in place)
                    self._rev_stuck = getattr(self, "_rev_stuck", [None, 0])
                    moved = (self._last_pos is not None and
                             (ws.px - self._last_pos[0])**2 +
                             (ws.py - self._last_pos[1])**2 >= 100)
                    if self._rev_stuck[0] is not None and not moved:
                        self._rev_stuck[1] += 1
                    else:
                        self._rev_stuck = [ws.px, 0]
                    if self._rev_stuck[1] >= 3:
                        turn = "turn_left" if b > 0 else "turn_right"
                        probs[turn] = 0.9
                        hold = max(9, min(48, 7 + int(abs(b) / 4.5) + 2))
                        reason = f"nav: reverse stuck -> commit turn ({b:.0f})"
                    else:
                        probs["move_back"] = 0.9
                        hold = 14
                        reason = f"nav: waypoint behind ({b:.0f}) -> reverse"
                elif abs(b) > 20:
                    # 20-deadband: Doom wall-slide handles small errors; turning
                    # to exact alignment limit-cycles (min hold overshoots ±15deg)
                    turn = "turn_left" if b > 0 else "turn_right"
                    probs[turn] = 0.85
                    # windup varies by scenario; damp hard when bearing oscillates
    # (sign flipped while not moving = we are overshooting the waypoint)
                    flipped = (self._last_nav_b is not None and
                               (b > 0) != (self._last_nav_b > 0))
                    moved = (self._last_pos is not None and
                             (ws.px - self._last_pos[0])**2 + (ws.py - self._last_pos[1])**2 >= 100)
                    # full proportional turn, always. The old shrink damping
                    # (a band-aid for target-flip oscillation, which target-
                    # lock hysteresis already fixes) caused turn limit-cycles:
                    # shrunk holds rotate ~0deg and the bearing never converges.
                    hold = max(9, min(48, 7 + int(abs(b) / 4.5) + 2))
                    self._turn_shrink = hold
                    self._last_nav_b = b
                    if self._stuck:
                        self._blocked_streak += 1
                        s = self._blocked_streak
                        phase = s % 4
                        if phase == 0 and not near_spawn:
                            probs["use"] = 0.9; hold = 5
                            reason = "nav: oscill sweep -> USE"
                        elif phase == 1:
                            # doors rise slowly and auto-close: press USE then
                            # immediately ride forward through while it is open
                            probs["move_forward"] = 0.9; hold = 30
                            reason = "nav: oscill sweep -> ride door open"
                        elif phase == 3:
                            # strafe around convex corners: forward+turn never
                            # clears them; lateral movement does
                            if abs(b) > 50:
                                side = "move_left" if b > 0 else "move_right"
                                probs[side] = 0.9; hold = 14
                                reason = "nav: oscill sweep -> strafe to path"
                            else:
                                probs["move_forward"] = 0.9; hold = 14
                                reason = "nav: oscill sweep -> step"
                        elif phase == 0:
                            probs["turn_right"] = 0.9; hold = 20
                            reason = "nav: oscill near spawn -> hug (no USE: teleport)"
                        else:
                            turn = "turn_left" if b > 0 else "turn_right"
                            probs[turn] = 0.9; hold = 12
                            reason = "nav: oscill sweep -> turn"
                    else:
                        reason = f"nav: bearing={b:.0f} exit d={ws.nav_dist:.0f} hold={hold}"
                else:
                    # Doom accelerates from standstill: judge "blocked" only after a
                    # long enough hold, else acceleration looks like a wall
                    blocked = (self._last_pos is not None and
                               (ws.px - self._last_pos[0])**2 + (ws.py - self._last_pos[1])**2 < 100)
                    self._last_pos = (ws.px, ws.py)
                    # sliding along a wall resets per-step blocked; the global
                    # net-displacement detector catches oscillating-in-place
                    if not blocked and self._stuck:
                        blocked = True
                    if blocked:
                        self._blocked_streak += 1
                        s = self._blocked_streak
                        if s >= 20:
                            # pivot AND step: a sweep that only spins never
                            # escapes; alternate scan / advance
                            if s % 2 == 0:
                                probs["turn_right"] = 0.9; hold = 24
                                reason = "nav: stuck -> pivot scan"
                            else:
                                probs["move_forward"] = 0.9; hold = 16
                                reason = "nav: stuck -> pivot step"
                        elif s >= 6:
                            # sweep: USE / RIDE forward / turn / step
                            phase = s % 4
                            if phase == 0 and not near_spawn:
                                probs["use"] = 0.9; hold = 5
                                reason = "nav: stuck sweep -> USE"
                            elif phase == 1:
                                probs["move_forward"] = 0.9; hold = 30
                                reason = "nav: stuck sweep -> ride door open"
                            elif phase == 3:
                                # strafe around corners: forward-into-wall + turn
                                # never clears a convex corner; lateral does
                                if abs(b) > 50:
                                    side = "move_left" if b > 0 else "move_right"
                                    probs[side] = 0.9; hold = 14
                                    reason = "nav: stuck sweep -> strafe to path"
                                else:
                                    probs["move_forward"] = 0.9; hold = 14
                                    reason = "nav: stuck sweep -> step"
                            elif phase == 0:
                                probs["turn_right"] = 0.9; hold = 20
                                reason = "nav: stuck sweep -> hug (no USE)"
                            else:
                                probs["turn_right"] = 0.9; hold = 12
                                reason = "nav: stuck sweep -> turn"
                        elif s >= 3:
                            # 90-degree right hug: walks around corners, no overshoot
                            probs["turn_right"] = 0.9; hold = 20
                            reason = "nav: blocked x3 -> 90-deg hug"
                        else:
                            probs["turn_right"] = 0.7; probs["turn_left"] = 0.2; hold = 34
                            reason = f"nav: blocked -> turn (pos {ws.px:.0f},{ws.py:.0f})"
                    else:
                        # weapon pickups are free combat power: detour while aligned
                        WEAPONS = {"Shotgun", "Chaingun", "PlasmaGun", "BFG10K"}
                        pk = min(ws.pickups, key=lambda q: q.dist, default=None)
                        want = (pk is not None and pk.dist < 450 and
                                (pk.name in WEAPONS or ws.health < 60 or ws.ammo < 10))
                        if want and abs(bearing_deg(pk)) > 15:
                            pb = bearing_deg(pk)
                            turn = "turn_left" if pb > 0 else "turn_right"
                            probs[turn] = 0.9
                            hold = max(8, min(40, 8 + int(abs(pb) / 3.5)))
                            reason = f"nav: face pickup {pk.name} d={pk.dist:.0f}"
                        elif want:
                            probs["move_forward"] = 0.9
                            hold = 14
                            reason = f"nav: grab {pk.name} d={pk.dist:.0f}"
                        else:
                            probs["move_forward"] = 0.9
                            hold = 20  # long holds keep Doom momentum
                            reason = f"nav: advance, exit d={ws.nav_dist:.0f}"
                probs_navigate = True
            elif stance == "navigate" and ws.monsters:
                far = [m for m in ws.monsters if m.dist < 700]  # don't chase ghosts
                guide = max(far or ws.monsters, key=lambda m: m.dist)
                b = bearing_deg(guide)
                if abs(b) > 12:
                    turn = "turn_left" if b > 0 else "turn_right"
                    probs[turn] = 0.85
                    hold = max(12, min(48, 8 + int(abs(b) / 3.5) + 5))
                    reason = f"guide: follow corridor to {guide.name} bearing={b:.0f} d={guide.dist:.0f}"
                else:
                    pk = min(ws.pickups, key=lambda p: p.dist, default=None)
                    blocked = (self._last_pos is not None and
                               (ws.px - self._last_pos[0])**2 + (ws.py - self._last_pos[1])**2 < 64)
                    self._last_pos = (ws.px, ws.py)
                    if blocked:
                        self._blocked_streak += 1
                        if self._blocked_streak >= 3:
                            probs["turn_right"] = 0.9; self._blocked_streak = 0; hold = 48
                            reason = "guide: blocked x3 -> pivot"
                        else:
                            probs["turn_right"] = 0.8; hold = 34
                            reason = f"guide: blocked -> right (pos {ws.px:.0f},{ws.py:.0f})"
                    elif pk is not None and pk.dist < 300 and ws.health < 80 and abs(bearing_deg(pk)) > 25:
                        turn = "turn_left" if bearing_deg(pk) > 0 else "turn_right"
                        probs[turn] = 0.7
                        hold = max(12, min(48, 8 + int(abs(bearing_deg(pk)) / 3.5) + 5))
                        reason = f"guide: detour to {pk.name}"
                    else:
                        probs["move_forward"] = 0.85
                        reason = f"guide: advance to {guide.name} d={guide.dist:.0f}"
                probs_navigate = True
            else:
                probs_navigate = False
            nearest = None
            if self._lock_id is not None and self._lock_age < 15:
                cur = next((m for m in ws.monsters if m.id == self._lock_id), None)
                if cur is not None:
                    cand = min(ws.monsters, key=lambda m: m.dist, default=None)
                    if cand is not None and cand.dist < cur.dist * 0.6:
                        nearest = cand
                    else:
                        nearest = cur
            else:
                nearest = min(ws.monsters, key=lambda m: m.dist, default=None)
            if not probs_navigate and nearest is not None:
                if nearest.id != self._lock_id:
                    self._lock_id, self._lock_age = nearest.id, 0
                else:
                    self._lock_age += 1
            if not probs_navigate and nearest is not None:
                b = bearing_deg(nearest)
                if abs(b) > 15:
                    turn = "turn_left" if b > 0 else "turn_right"
                    probs[turn] = 0.85
                    hold = turn_hold(b)
                    reason = f"orient to {nearest.name} bearing={b:.0f} d={nearest.dist:.0f}"
                else:
                    # facing it but not visible => geometry blocks LOS: go around/forward
                    self._lock_age += 1
                    probs["move_forward"] = 0.7
                    probs["move_left" if self.rng.random() < 0.5 else "move_right"] = 0.2
                    reason = f"advance to {nearest.name} d={nearest.dist:.0f} (no LOS)"
            elif not probs_navigate:
                # explore: right-hand-rule wall hug toward the exit (guided by brain)
                blocked = False
                if self._last_pos is not None:
                    dx, dy = ws.px - self._last_pos[0], ws.py - self._last_pos[1]
                    blocked = (dx * dx + dy * dy) < 64  # <8 units moved since last step
                self._last_pos = (ws.px, ws.py)
                pk = min(ws.pickups, key=lambda p: p.dist, default=None)
                if False:  # pickup detour removed: spun in circles chasing bonuses
                    pass
                elif blocked:
                    self._blocked_streak += 1
                    # jittered wall-hug: pure right-hand rule cycles on this map
                    if self._blocked_streak >= 3:
                        # maybe it is a DOOR: try USE, then look for the switch
                        probs["use"] = 0.9
                        self._blocked_streak = 0
                        hold = 3
                        reason = f"explore: blocked x3 -> USE door (pos {ws.px:.0f},{ws.py:.0f})"
                    elif self.rng.random() < 0.25:
                        probs["turn_left"] = 0.85
                        hold = 34
                        reason = f"explore: blocked -> hard LEFT (pos {ws.px:.0f},{ws.py:.0f})"
                    else:
                        probs["turn_right"] = 0.85
                        hold = 34
                        reason = f"explore: blocked -> hard right (pos {ws.px:.0f},{ws.py:.0f})"
                else:
                    self._blocked_streak = 0
                    probs["move_forward"] = 0.6
                    probs["turn_right" if self.rng.random() < 0.7 else "turn_left"] = 0.25
                    hold = 2 if probs["move_forward"] >= probs["turn_right"] else 12
                    reason = f"explore: advance/hug right (pos {ws.px:.0f},{ws.py:.0f})"

        # defensive override: break contact
        self._last_pos = (ws.px, ws.py)
        if stance in ("defensive", "retreat") and target is not None:
            # disengage dominates: back up while firing, stop turning in place
            probs["move_back"] = max(probs.get("move_back", 0) + 0.6, 0.55)
            if ws.attack_ready and abs(bearing_deg(target)) < 12:
                # keep shooting while backing up: a retreating agent that
                # stops firing just dies cornered
                probs["attack"] = max(probs.get("attack", 0.0), 0.75)
            else:
                probs["attack"] *= 0.7
            for t in ("turn_left", "turn_right"):
                probs[t] *= 0.25

        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        top = max(probs, key=probs.get)
        # compound action: Doom buttons are concurrent. Advance while firing
        # breaks the pinch deadlock (turn-to-aim in a chokepoint = pinned).
        if (top == "move_forward" and ws.ammo > 0 and ws.attack_ready
                and target is not None and abs(bearing_deg(target)) < 30
                and ws.nav_bearing is not None and abs(ws.nav_bearing) < 25):
            top = ["move_forward", "attack"]
        elif (top == "attack" and target is not None and target.dist > 250
              and ws.nav_bearing is not None and abs(ws.nav_bearing) < 25):
            top = ["move_forward", "attack"]
        return Decision(action=top, probs=probs, confidence=probs[top] if isinstance(top, str) else 1.0,
                        reason=reason, hold_tics=hold)

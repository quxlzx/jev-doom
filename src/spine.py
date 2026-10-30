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

    def decide(self, ws, goal: dict) -> Decision:
        probs = {a: 0.0 for a in ACTIONS}
        stance = goal.get("stance", "aggressive")
        target = ws.nearest_visible_monster()
        hold = 2

        # --- reflex: flinch on damage even if the brain is asleep
        if ws.health < 35 and stance != "retreat":
            stance = "defensive"

        def turn_hold(bearing: float) -> int:
            # 7-tic windup + 3.5deg/tic; proportional, capped
            return min(40, 8 + int(abs(bearing) / 3.5))

        if target is not None:
            b = bearing_deg(target)
            dist = target.dist
            aligned = abs(b) < 6.0

            if aligned and ws.attack_ready and dist < 1600:
                probs["attack"] = 0.85 if stance == "aggressive" else 0.6
                probs["move_left"] = 0.07
                probs["move_right"] = 0.07
                reason = f"fire on {target.name} d={dist:.0f}"
            else:
                turn = "turn_left" if b > 0 else "turn_right"
                probs[turn] = 0.9
                hold = turn_hold(b)
                reason = f"acquire {target.name} bearing={b:.0f} d={dist:.0f}"
        else:
            # no monster on screen: use the world map to orient, then close distance
            # hysteresis: keep the current target unless it dies or something is
            # much closer (prevents left/right oscillation between two monsters)
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
            if nearest is not None:
                if nearest.id != self._lock_id:
                    self._lock_id, self._lock_age = nearest.id, 0
                else:
                    self._lock_age += 1
            if nearest is not None:
                b = bearing_deg(nearest)
                if abs(b) > 15:
                    turn = "turn_left" if b > 0 else "turn_right"
                    probs[turn] = 0.85
                    hold = turn_hold(b)
                    reason = f"orient to {nearest.name} bearing={b:.0f} d={nearest.dist:.0f}"
                else:
                    # facing it but not visible => geometry blocks LOS: go around/forward
                    probs["move_forward"] = 0.7
                    probs["move_left" if self.rng.random() < 0.5 else "move_right"] = 0.2
                    reason = f"advance to {nearest.name} d={nearest.dist:.0f} (no LOS)"
            else:
                # explore: right-hand-rule wall hug toward the exit (guided by brain)
                blocked = False
                if self._last_pos is not None:
                    dx, dy = ws.px - self._last_pos[0], ws.py - self._last_pos[1]
                    blocked = (dx * dx + dy * dy) < 64  # <8 units moved since last step
                self._last_pos = (ws.px, ws.py)
                pk = min(ws.pickups, key=lambda p: p.dist, default=None)
                if pk is not None and pk.dist < 350 and abs(bearing_deg(pk)) > 20 and ws.health < 80:
                    turn = "turn_left" if bearing_deg(pk) > 0 else "turn_right"
                    probs[turn] = 0.7
                    hold = min(40, 8 + int(abs(bearing_deg(pk)) / 3.5))
                    reason = f"explore: route to {pk.name} d={pk.dist:.0f}"
                elif blocked:
                    probs["turn_right"] = 0.85
                    hold = 34  # ~90 deg
                    reason = f"explore: blocked -> hard right (pos {ws.px:.0f},{ws.py:.0f})"
                else:
                    probs["move_forward"] = 0.6
                    probs["turn_right"] = 0.25
                    hold = 2 if probs["move_forward"] >= probs["turn_right"] else 12
                    reason = f"explore: advance/hug right (pos {ws.px:.0f},{ws.py:.0f})"

        # defensive override: break contact
        self._last_pos = (ws.px, ws.py)
        if stance in ("defensive", "retreat") and target is not None:
            # disengage dominates: back up while firing, stop turning in place
            probs["move_back"] = max(probs.get("move_back", 0) + 0.6, 0.55)
            probs["attack"] *= 0.7
            for t in ("turn_left", "turn_right"):
                probs[t] *= 0.25

        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        top = max(probs, key=probs.get)
        return Decision(action=top, probs=probs, confidence=probs[top],
                        reason=reason, hold_tics=hold)

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
            nearest = min(ws.monsters, key=lambda m: m.dist, default=None)
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
                side = self.rng.choice(["move_left", "move_right"])
                probs[side] = 0.5
                reason = "no monsters at all"

        # defensive override: break contact
        if stance in ("defensive", "retreat") and target is not None:
            probs["move_back"] += 0.25
            probs["attack"] *= 0.6

        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        top = max(probs, key=probs.get)
        return Decision(action=top, probs=probs, confidence=probs[top],
                        reason=reason, hold_tics=hold)

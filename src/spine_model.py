"""System One, powered by an open model: grammar-constrained llama.cpp spine.

Architecture: the spinal cord (heuristic geometry) proposes candidate motor
programs; the model (Qwen3-1.7B on gpu-host, llama.cpp) SELECTS among them.
GBNF grammar forces a typed answer ("[1]".."[3]") the model cannot get wrong
structurally. Agreement between model pick and heuristic top choice is the
confidence signal; disagreement escalates to the brain.

Falls back to the pure heuristic on any HTTP failure — the organism never
dies from a model sneeze.
"""
from __future__ import annotations
import json
import urllib.request

from spine import Decision, ReflexSpine, bearing_deg


class ModelSpine:
    def __init__(self, base_url: str = "http://127.0.0.1:8181", timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.fallback = ReflexSpine()

    PROMPT_TMPL = """Pick the best Doom action. Reply with its bracketed number.
State:
Stance: {stance}
Health {health:.0f}, armor {armor:.0f}, ammo {ammo:.0f}, attack_ready {ready}.
{target_line}{nearest_line}{pickup_line}
Candidates:
1) {c1}
2) {c2}
3) {c3}
Answer:"""

    def _state_lines(self, ws):
        t = ws.nearest_visible_monster()
        target_line = "On screen target: none.\n"
        if t:
            b = bearing_deg(t)
            target_line = (f"On screen: {t.name} bearing {b:+.0f} "
                           f"dist {t.dist:.0f} target_aligned {'yes' if abs(b) < 8 else 'no'}.\n")
        nearest = min(ws.monsters, key=lambda m: m.dist, default=None)
        nearest_line = ""
        if nearest:
            nearest_line = (f"Nearest monster in level: {nearest.name} bearing "
                            f"{bearing_deg(nearest):+.0f} dist {nearest.dist:.0f}.\n")
        pk = min(ws.pickups, key=lambda p: p.dist, default=None)
        pickup_line = ""
        if pk and pk.dist < 500 and ws.health < 70:
            pickup_line = f"Nearby pickup: {pk.name} bearing {bearing_deg(pk):+.0f} dist {pk.dist:.0f}.\n"
        return target_line, nearest_line, pickup_line, nearest

    def decide(self, ws, goal: dict) -> Decision:
        # spinal cord proposes candidate motor programs; the model selects.
        h = self.fallback.decide(ws, goal)
        ranked = sorted(h.probs, key=lambda a: -h.probs[a])[:3]
        target_line, nearest_line, pickup_line, nearest = self._state_lines(ws)
        desc = {
            "attack": "fire pistol now",
            "turn_left": "turn left to face the monster",
            "turn_right": "turn right to face the monster",
            "move_forward": "advance toward nearest monster",
            "move_back": "back away from threat",
            "move_left": "strafe left",
            "move_right": "strafe right",
        }
        cands = [f"{desc.get(a, a)} [{a}]" for a in ranked]
        prompt = self.PROMPT_TMPL.format(
            stance=goal.get("stance", "aggressive"),
            health=ws.health, armor=ws.armor, ammo=ws.ammo,
            ready="yes" if ws.attack_ready else "no",
            target_line=target_line, nearest_line=nearest_line, pickup_line=pickup_line,
            c1=cands[0], c2=cands[1] if len(cands) > 1 else "wait [attack]",
            c3=cands[2] if len(cands) > 2 else "wait [attack]",
        )
        body = {"prompt": prompt, "temperature": 0.0, "n_predict": 3}

        def hold_for(action: str) -> int:
            t = ws.nearest_visible_monster() or nearest
            if action.startswith("turn") and t is not None:
                return min(40, 8 + int(abs(bearing_deg(t)) / 3.5))
            return 2

        try:
            req = urllib.request.Request(
                self.base_url + "/completion",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read())
            content = out.get("content", "")
            digit = next((c for c in content if c in "123"), None)
            if digit is None:
                raise ValueError(f"no pick in {content!r}")
            idx = int(digit) - 1
            action = ranked[idx] if idx < len(ranked) else h.action
            agree = idx == 0
            conf = 0.9 if agree else 0.35
            probs = dict(h.probs)
            probs[action] = max(probs.get(action, 0.0), conf)
            total = sum(probs.values()) or 1.0
            probs = {k: v / total for k, v in probs.items()}
            conf = probs[action]
            return Decision(action=action, probs=probs, confidence=conf,
                            reason=f"model#{digit}{'~' if agree else '!='}heur#1 "
                                   f"({h.reason})",
                            hold_tics=hold_for(action))
        except Exception as e:
            h.reason = f"fallback({e.__class__.__name__}): {h.reason}"
            return h

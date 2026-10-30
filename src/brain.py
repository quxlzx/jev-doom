"""System Two — the brain.

Slow, deliberative loop. Reads a compressed situation summary, writes goals
(stance + intent) to the blackboard that the spinal cord conditions on.
Two implementations behind one interface:
  RuleBrain — deterministic fallback, zero deps.
  LLMBrain  — any OpenAI-compatible endpoint, JSON-mode goals.
"""
from __future__ import annotations
import json
import os
import urllib.request


def summarize(ws, spine_log: list[str]) -> str:
    t = ws.nearest_visible_monster()
    lines = [
        f"tic={ws.tic} health={ws.health:.0f} armor={ws.armor:.0f} ammo={ws.ammo:.0f} "
        f"kills={ws.kills:.0f} deaths={ws.deaths:.0f}",
        f"monsters in level: {len(ws.monsters)}; on screen: "
        f"{sum(1 for m in ws.visible if m.category=='monster')}",
    ]
    if t:
        lines.append(f"nearest visible monster: {t.name} dist={t.dist:.0f}")
    if spine_log:
        lines.append("recent reflexes: " + "; ".join(spine_log[-4:]))
    return "\n".join(lines)


class RuleBrain:
    """Reflex arc of the brain: pure thresholds. Always available."""

    def think(self, ws) -> dict:
        if ws.nearest_visible_monster() is None and ws.health >= 30:
            return {"stance": "navigate", "intent": "follow right wall toward exit"}
        if ws.health < 30:
            return {"stance": "retreat", "intent": "break contact, heal if reachable"}
        if ws.health < 55:
            return {"stance": "defensive", "intent": "use cover, pick shots"}
        return {"stance": "aggressive", "intent": "engage nearest monster"}


class LLMBrain:
    """OpenAI-compatible chat endpoint. Must return JSON: {stance, intent}."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float = 20.0):
        self.base_url = (base_url or os.environ.get("JEVDOOM_BRAIN_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("JEVDOOM_BRAIN_KEY", "")
        self.model = model or os.environ.get("JEVDOOM_BRAIN_MODEL", "")
        self.timeout = timeout

    def think(self, ws) -> dict:
        summary = summarize(ws, [])
        req_body = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 120,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content":
                 "You are the deliberative brain of a Doom agent. Your spinal cord handles "
                 "reflexes; you set strategy. Guide: this is a deadly corridor — the exit is at "
                 "the far end; when no monster is on screen use stance=navigate and the "
                 "spine follows the right-hand wall forward. Reply JSON: "
                 '{"stance": "aggressive|defensive|retreat", "intent": "<one short line>"}. '
                 "Consider health, ammo, and monster pressure."},
                {"role": "user", "content": summary},
            ],
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(req_body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read())
            content = out["choices"][0]["message"]["content"]
            goal = json.loads(content)
            if goal.get("stance") not in ("aggressive", "defensive", "retreat", "navigate"):
                raise ValueError("bad stance")
            return {"stance": goal["stance"], "intent": goal.get("intent", "")}
        except Exception as e:
            # brain stroke: fall back to the brainstem, never crash the loop
            return {**RuleBrain().think(ws), "intent": f"(LLM failed: {e})"}


def make_brain(prefer_llm: bool = True):
    if prefer_llm and os.environ.get("JEVDOOM_BRAIN_URL"):
        return LLMBrain()
    return RuleBrain()

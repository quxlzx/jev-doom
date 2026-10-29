"""Holdout eval: pick accuracy of the spine server on corpus prompts.

Usage: python eval_pick.py <corpus.jsonl> <base_url> [n]
Samples n prompts, sends each to the completion endpoint, compares the
bracketed pick to the corpus label.
"""
from __future__ import annotations
import json
import random
import sys
import urllib.request

def main():
    corpus, url = sys.argv[1], sys.argv[2].rstrip("/")
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    random.seed(99)
    rows = [json.loads(l) for l in open(corpus, encoding="utf-8")]
    if len(sys.argv) > 4 and sys.argv[4] == "dev":
        rows = [r for r in rows if r["completion"] != "[1]"]
    sample = random.sample(rows, min(n, len(rows)))
    hits = 0
    dist = {}
    for i, r in enumerate(sample):
        body = {"prompt": r["prompt"], "temperature": 0.0, "n_predict": 3}
        try:
            req = urllib.request.Request(url + "/completion",
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            out = json.loads(urllib.request.urlopen(req, timeout=30).read())
            pick = next((c for c in out.get("content", "") if c in "123"), "?")
        except Exception:
            pick = "!"
        label = r["completion"][1]
        dist[pick] = dist.get(pick, 0) + 1
        if pick == label:
            hits += 1
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(sample)} acc={hits/(i+1):.3f}")
    print(f"FINAL acc={hits/len(sample):.3f} over {len(sample)} | pick dist={dist} | label dist="
          f"{ {l: sum(1 for r in sample if r['completion'][1]==l) for l in '123'} }")

if __name__ == "__main__":
    main()

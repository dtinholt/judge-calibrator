"""Writes the synthetic demo data in this folder. Run from the build root:
    python3 demo/data/make_synthetic.py

Every verdict here comes from a fixed pattern and a seeded shuffle. No model judged anything,
and no figure computed from these files says anything about any judge. They exist so the demo
report and the drift page have something to show while the registered judge-reliability study
(October 2026) is unpublished.
"""
import json
import random
from pathlib import Path

D = Path(__file__).resolve().parent
rng = random.Random("judgecal synthetic demo v1")
CL = ["PASS", "PARTIAL", "FAIL"]
ids = [f"s{n:02d}" for n in range(1, 61)]
base = ["PASS"] * 30 + ["PARTIAL"] * 18 + ["FAIL"] * 12
rng.shuffle(base)


def flip(labels, k, towards=None):
    out = list(labels)
    for i in rng.sample(range(len(out)), k):
        choices = [c for c in CL if c != out[i]]
        out[i] = towards if towards and towards != out[i] else rng.choice(choices)
    return out


runs = [base, flip(base, 5, "PARTIAL"), flip(base, 6, "PARTIAL")]
mid = flip(base, 7)
small = flip(base, 13)


def rows(labels, judge=None, repeat=0, date=None):
    for i, c in zip(ids, labels):
        r = {"item": i, "repeat": repeat, "class": c,
             "confidence": "low" if rng.random() < 0.08 else "high"}
        if judge:
            r["judge"] = judge
        if date:
            r["date"] = date
        yield r


def write(name, it):
    with open(D / name, "w") as f:
        for r in it:
            f.write(json.dumps(r) + "\n")


write("items.jsonl", ({"id": i, "text": f"synthetic case {i}"} for i in ids))
write("frontier.jsonl", (r for k, lab in enumerate(runs) for r in rows(lab, repeat=k)))
write("mid.jsonl", rows(mid))
write("small.jsonl", rows(small))
write("strata.jsonl", ({"id": i, "stratum": "short cases" if n % 2 else "long cases"} for n, i in enumerate(ids)))
write("verdicts-2026-09-28.jsonl", (r for k in (0, 1) for r in rows(runs[k], "frontier", k, "2026-09-28")))
write("verdicts-2026-10-05.jsonl", rows(flip(base, 8), "frontier", 0, "2026-10-05"))
(D / "rubric.json").write_text(json.dumps({
    "name": "synthetic-demo",
    "classes": CL,
    "instruction": "Synthetic demo rubric. Decide whether the case meets the requirement.",
    "descriptions": {"PASS": "meets it", "PARTIAL": "meets part of it", "FAIL": "does not meet it"},
}, indent=1) + "\n")

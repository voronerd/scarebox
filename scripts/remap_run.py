"""Map a new replay's tracker IDs onto the labelled run (same video) by per-frame box overlap,
so timeline_sim.py can score it with the existing labels. Usage: remap_run.py <labelled_dir> <new_dir>"""
import json
import shutil
import sys
from collections import Counter, defaultdict

old_dir, new_dir = sys.argv[1], sys.argv[2]


def load(p):
    return {json.loads(l)["f"]: json.loads(l) for l in open(p)}


def iou(a, b):
    x1, y1 = max(a[1], b[1]), max(a[2], b[2])
    x2, y2 = min(a[3], b[3]), min(a[4], b[4])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - i
    return i / u if u else 0


old, new = load(f"{old_dir}/tracks.jsonl"), load(f"{new_dir}/tracks.jsonl")
votes = defaultdict(Counter)
for f, row in new.items():
    for nb in row["b"]:
        best = max(old.get(f, {"b": []})["b"], key=lambda ob: iou(nb, ob), default=None)
        if best and iou(nb, best) > 0.5:
            votes[nb[0]][best[0]] += 1
m = {n: c.most_common(1)[0][0] for n, c in votes.items()}
out = []
for l in open(f"{new_dir}/events.jsonl"):
    e = json.loads(l)
    if e["kind"] == "trigger":
        e["tracks"] = [m[t] for t in e["tracks"] if t in m]
    if e["kind"] == "result" and e.get("spotlight"):
        e["spotlight"]["track"] = m.get(e["spotlight"]["track"], -1)
    out.append(json.dumps(e))
open(f"{new_dir}/events.jsonl", "w").write("\n".join(out) + "\n")
for f in ("tracks.json", "labels.json"):
    shutil.copy(f"{old_dir}/{f}", f"{new_dir}/{f}")
print(f"mapped {len(m)} tracker ids")

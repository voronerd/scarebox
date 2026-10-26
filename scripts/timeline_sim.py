#!/usr/bin/env python3
"""timeline_sim.py: score who got heckled over a captured video run, and replay the same
timeline under other policies (2026-10-19, Martin: "temporal optimisation needed").

Inputs (from a capture run dir, see scarebox/capture.py and scripts/track_shots.py):
  tracks.jsonl  every frame's tracked boxes   tracks.json  per-track span + best frame
  events.jsonl  triggers / results / played / dropped
  labels.json   {track_id: {"costume": str, "standout": 0|1|2}} -- the stand-in judge's grades
                (2 = must announce, 1 = fine to announce, 0 = no costume / not worth a line)

Everything runs on VIDEO time (frame / fps), so a person's "still in view" window is exact.

  python scripts/timeline_sim.py runs/capture/<run> [--fps 30] [--json out.json]

Reports, for the ACTUAL run and for each simulated POLICY, over the whole video and over
BUSY stretches only (3+ people in view):
  must-announce covered  % of standout-2 people who got a line while still in view
  worth covered          % of standout>=1 people who got a line while still in view
  on-time                % of played lines that started while their person was still in view
  wasted                 % of played lines spent on standout-0 people (no costume / generic)
and ORACLE: the best a single speaker could have done (an upper bound, not a policy).
"""
from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--json")
ap.add_argument("--busy", type=int, default=3, help="people in view for a moment to count as busy")
ap.add_argument("--gap", type=float, default=2.5, help="silence after each line (s)")
ap.add_argument("--line-s", type=float, default=None, help="line length (s); default = measured median")
ap.add_argument("--late", type=float, default=0.5,
                help="a line still counts this long after its person leaves frame (speaker further down the path)")
ap.add_argument("--max-age", type=float, default=5.0, help="stage.max_line_age_s: drop a line queued longer than this")
ap.add_argument("--busy-gap", type=float, default=None, help="gap after a line while crowded (stage.busy_gap_s)")
ap.add_argument("--busy-line-s", type=float, default=None, help="line length while crowded (busy mode picks short lines)")
args = ap.parse_args()
run = Path(args.run)
FPS = args.fps

# ---- load -------------------------------------------------------------------------------
frame_t: list[tuple[float, int]] = []  # (monotonic t, frame) to map pipeline clocks to video time
in_view: dict[int, int] = {}  # frame -> number of people tracked
for line in (run / "tracks.jsonl").open():
    row = json.loads(line)
    frame_t.append((row["t"], row["f"]))
    in_view[row["f"]] = len(row["b"])
ts = [t for t, _ in frame_t]
tracks = {int(k): v for k, v in json.loads((run / "tracks.json").read_text()).items()}
# tracks.json is per PERSON (track_shots.py stitches tracker ID switches); map raw tracker
# IDs in the trigger log onto those people.
ID2P = {int(t): pid for pid, v in tracks.items() for t in v.get("ids", [pid])}


def persons(raw_ids) -> list[int]:
    return sorted({ID2P[int(t)] for t in raw_ids if int(t) in ID2P})
labels = {int(k): v for k, v in json.loads((run / "labels.json").read_text()).items()}
events = [json.loads(l) for l in (run / "events.jsonl").open()]


def video_s_from_mono(t: float) -> float:
    """Pipeline monotonic time -> video seconds, via the per-frame capture timestamps."""
    i = min(max(bisect.bisect_left(ts, t), 0), len(ts) - 1)
    return frame_t[i][1] / FPS


def standout(tid: int) -> int:
    return int(labels.get(tid, {}).get("standout", 0))


def span(tid: int) -> tuple[float, float]:
    t = tracks[tid]
    return t["first_f"] / FPS, t["last_f"] / FPS


people = [tid for tid in tracks if tid in labels and standout(tid) >= 0]  # -1 = unusable tile
busy_frames = {f for f, n in in_view.items() if n >= args.busy}


def busy_person(tid: int) -> bool:
    a, b = tracks[tid]["first_f"], tracks[tid]["last_f"]
    return any(f in busy_frames for f in range(a, b + 1, 5))


# ---- scoring ------------------------------------------------------------------------------
@dataclass
class Line:
    start_s: float  # video seconds when the line started
    tids: list[int]  # the people it was about
    score: int | None = None
    # Who the line actually NAMES. A group line ("an entire royal court!") names nobody: it
    # can be on time, but it doesn't count as announcing anyone's costume.
    named: int | None = None


def score(lines: list[Line], name: str) -> dict:
    covered: set[int] = set()
    on_time = wasted = group = 0
    for ln in lines:
        live = [t for t in ln.tids if t in tracks and span(t)[0] <= ln.start_s <= span(t)[1] + LATE_S]
        if live:
            on_time += 1
            if ln.named is not None and ln.named in live:
                covered.add(ln.named)
        if ln.named is None:
            group += 1
        elif standout(ln.named) == 0:
            wasted += 1

    def pct(sel, base):
        return round(100.0 * len(sel) / len(base), 1) if base else None

    out = {"policy": name, "lines": len(lines)}
    for scope, pool in (("all", people), ("busy", [t for t in people if busy_person(t)])):
        must = [t for t in pool if standout(t) == 2]
        worth = [t for t in pool if standout(t) >= 1]
        out[scope] = {
            "people": len(pool),
            "must": len(must),
            "must_covered_pct": pct([t for t in must if t in covered], must),
            "worth_covered_pct": pct([t for t in worth if t in covered], worth),
        }
    out["on_time_pct"] = round(100.0 * on_time / len(lines), 1) if lines else None
    out["wasted_pct"] = round(100.0 * wasted / len(lines), 1) if lines else None
    out["group_line_pct"] = round(100.0 * group / len(lines), 1) if lines else None
    return out


# ---- the ACTUAL run ------------------------------------------------------------------------
trig = {e["id"]: e for e in events if e["kind"] == "trigger"}
res = {e["id"]: e for e in events if e["kind"] == "result"}
played = [e for e in events if e["kind"] == "played" and e["id"] in trig]
def _named(eid):
    r = res.get(eid) or {}
    if r.get("spotlight"):  # a group, but the line names the one standout member
        ps = persons([r["spotlight"]["track"]])
        return ps[0] if ps else None
    ps = persons(trig[eid]["tracks"])
    return ps[0] if len(ps) == 1 and r.get("category") != "group" else None


actual = [Line(video_s_from_mono(e["t_start"]), persons(trig[e["id"]]["tracks"]), e.get("score"), _named(e["id"]))
          for e in played]
line_lengths = [e["t_end"] - e["t_start"] for e in played] or [3.5]
LINE_S = args.line_s or sorted(line_lengths)[len(line_lengths) // 2]  # median measured line length
judge_ms = sorted(r["ms"]["judge"] for r in res.values() if r.get("ms", {}).get("judge"))
JUDGE_S = (judge_ms[len(judge_ms) // 2] / 1000.0) if judge_ms else 0.8

# ---- simulated policies -------------------------------------------------------------------
GAP_S = args.gap
LATE_S = args.late
BUSY_GAP_S = args.busy_gap
BUSY_LINE_S = args.busy_line_s
CROWD_CROSSINGS = 3  # tuning.crowd_crossings: this many crossings in 10 s = crowded
MAX_AGE_S = args.max_age
crossings = sorted(
    ((trig[e]["f"] / FPS, persons(trig[e]["tracks"])) for e in trig if persons(trig[e]["tracks"])),
    key=lambda x: x[0],
)


def simulate(name: str, candidates_fn, max_judging: int = 2) -> list[Line]:
    """One speaker. `candidates_fn` turns the timeline into (ready_s, tids, priority) offers:
    a line about `tids` that could start from `ready_s` (after judging). The speaker, when
    free, takes the highest-priority offer that is still fresh (not older than MAX_AGE_S and
    its person still in view), newest first on ties."""
    offers = sorted(candidates_fn(max_judging), key=lambda o: o[0])
    lines: list[Line] = []
    free_at = 0.0
    pending: list[tuple[float, list[int], float]] = []
    i = 0
    t = 0.0
    end = max((span(p)[1] for p in tracks), default=0.0)
    while t <= end:
        while i < len(offers) and offers[i][0] <= t:
            pending.append(offers[i])
            i += 1
        pending = [o for o in pending if t - o[0] <= MAX_AGE_S and any(span(p)[1] + LATE_S >= t for p in o[1] if p in tracks)]
        if t >= free_at and pending:
            best = max(pending, key=lambda o: (o[2], o[0]))
            pending.remove(best)
            lines.append(Line(t, best[1], named=best[1][0] if len(best[1]) == 1 else None))
            # stage.crowded(): more than one line waiting, or a burst of recent crossings.
            crowded = len(pending) >= 1 or sum(1 for c, _ in crossings if t - 10.0 <= c <= t) >= CROWD_CROSSINGS
            line_s = BUSY_LINE_S if (crowded and BUSY_LINE_S) else LINE_S
            gap = min(GAP_S, BUSY_GAP_S) if (crowded and BUSY_GAP_S is not None) else GAP_S
            free_at = t + line_s + gap
        t += 0.1
    return lines


def judged(max_judging: int, offers):
    """Apply a judging-concurrency cap: an offer waits for a free judge slot; JUDGE_S each."""
    slots = [0.0] * max_judging
    out = []
    for t0, tids, pri in sorted(offers, key=lambda o: o[0]):
        k = min(range(max_judging), key=lambda j: slots[j])
        if slots[k] > t0 + 1.0:  # judge too backed up: the crossing is skipped (busy-mode cap)
            continue
        start = max(t0, slots[k])
        slots[k] = start + JUDGE_S
        out.append((start + JUDGE_S, tids, pri))
    return out


# stage.SCORE_*: a standout costume usually gets a specific line (3), an ordinary one a
# category line (2), no costume plain goading (1).
LINE_PRI = {2: 3, 1: 2, 0: 1, -1: 0}


def best_of(tids):
    return max((standout(t) for t in tids), default=0)


def current(mj):
    # Today: one event per crossing. A group is one event with a generic group line (score 2,
    # names nobody); a single runner's line priority follows how specific it can be.
    return judged(mj, [(t0, tids, 2 if len(tids) > 1 else LINE_PRI[standout(tids[0])]) for t0, tids in crossings])


def spotlight(mj):
    # A group is split: each member is its own offer; the standout one wins the next slot.
    return judged(mj, [(t0, [t], LINE_PRI[standout(t)]) for t0, tids in crossings for t in tids])


def early(mj):
    # Judge each person as soon as they've been in view ~1 s (well before the line); the line
    # is ready at their crossing (or when judging finishes, if later).
    cross_at = {}
    for t0, tids in crossings:
        for t in tids:
            cross_at.setdefault(t, t0)
    offers = []
    for t in people:
        first, _ = span(t)
        ready = first + 1.0 + JUDGE_S
        offers.append((min(ready, cross_at.get(t, ready)) if t in cross_at else ready, [t], LINE_PRI[standout(t)]))
    return [o for o in offers]  # no judge cap modelled: judging is spread over each approach


def oracle() -> list[Line]:
    """Upper bound: knowing everyone's grade, greedily schedule lines highest standout first,
    each starting inside its person's in-view window (+ LATE_S), one speaker. Uses the busy
    timing throughout when given, so it stays an upper bound for the busy policies too."""
    slot = (BUSY_LINE_S or LINE_S) + (min(GAP_S, BUSY_GAP_S) if BUSY_GAP_S is not None else GAP_S)
    busy: list[tuple[float, float]] = []
    lines = []
    for t in sorted(people, key=lambda p: (-standout(p), span(p)[1])):
        if standout(t) == 0:
            continue
        a, b = span(t)
        b += LATE_S
        s = a + JUDGE_S
        while s <= b:
            if all(s + slot <= x or s >= y for x, y in busy):
                busy.append((s, s + slot))
                lines.append(Line(s, [t], named=t))
                break
            s += 0.2
    return lines


report = [
    score(actual, "ACTUAL run (today's code)"),
    score(simulate("current", current), "sim: current policy"),
    score(simulate("spotlight", spotlight), "sim: spotlight (split groups)"),
    score(simulate("early", early), "sim: early judging + spotlight"),
    score(simulate("current-mj1", lambda mj: current(1)), "sim: current, judge 1 at a time"),
    score(simulate("spotlight-mj3", lambda mj: spotlight(3)), "sim: spotlight, judge 3 at a time"),
    score(oracle(), "ORACLE (upper bound)"),
]
meta = {"line_s_median": round(LINE_S, 2), "judge_s_median": round(JUDGE_S, 2), "gap_s": GAP_S,
        "max_age_s": MAX_AGE_S, "people_graded": len(people), "crossings": len(crossings),
        "busy_threshold": args.busy, "late_s": LATE_S, "busy_gap_s": BUSY_GAP_S, "busy_line_s": BUSY_LINE_S}
print(json.dumps(meta))
hdr = f"{'policy':38} {'lines':>5} {'must%':>6} {'worth%':>6} {'busy must%':>10} {'busy worth%':>11} {'on-time%':>8} {'wasted%':>7} {'group%':>6}"
print(hdr)
for r in report:
    print(f"{r['policy']:38} {r['lines']:>5} {str(r['all']['must_covered_pct']):>6} {str(r['all']['worth_covered_pct']):>6} "
          f"{str(r['busy']['must_covered_pct']):>10} {str(r['busy']['worth_covered_pct']):>11} {str(r['on_time_pct']):>8} {str(r['wasted_pct']):>7} {str(r['group_line_pct']):>6}")
if args.json:
    Path(args.json).write_text(json.dumps({"meta": meta, "report": report}, indent=1))

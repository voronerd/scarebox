"""The public export must not point readers at private planning files.

The working repo keeps its plans, summaries, status board and bake-off
write-ups in a private planning directory that never ships. A shipped file
that says "see <one of those files>" sends a stranger to something they
cannot open, so this test scans every file the export would ship for such
references:

* fixed patterns for the planning file kinds (status board, roadmap, brief,
  rollup, checklist, numbered plan and summary files, evidence write-ups,
  the decision-required note), which also run inside an exported tree;
* when the private planning directory is present (the working repo), the
  name of every Markdown file in it that the export does not itself ship.

A path under ``runs/evidence/`` is exempt: shipped scripts write their own
output there, so it names a file the reader generates, not a private one.

Patterns are built at run time so this file passes its own scan and the gate.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORT = ROOT / "scripts" / "export_public.sh"
PLANNING = ROOT / ("." + "planning")
GENERATED_PREFIX = "runs/" + "evidence/"
BINARY_SUFFIXES = {".wav", ".mp3", ".png", ".jpg", ".jpeg", ".gif", ".onnx", ".pt", ".pth", ".gz", ".zip"}

_MD = r"\." + "md"
FIXED = [
    re.compile(r"(?<![\w.-])\d{2}-\d{2}-(?:" + "SUMMARY|PLAN" + ")" + _MD + r"\b"),
    re.compile(r"(?<![\w.-])(?:" + "STATUS|ROADMAP|BRIEF|ROLLUP|CHECKLIST" + ")" + _MD + r"\b"),
    re.compile(r"(?<![\w.-])" + "DECISION-" + "REQUIRED" + _MD + r"\b"),
    re.compile(r"(?<![\w.-])[\w-]*" + "bake" + r"-?off" + _MD + r"\b"),
    re.compile(r"(?<![\w.-])(?:[\w.-]+/)*" + "evid" + r"ence/[\w.-]+\.(?:md|jsonl?)\b"),
    re.compile(re.escape(PLANNING.name + "/")),
]


def allowlist() -> list[str]:
    text = EXPORT.read_text()
    m = re.search(r"^ALLOWLIST=\(\n(.*?)^\)", text, re.S | re.M)
    assert m, "ALLOWLIST block not found in export_public.sh"
    return [ln.strip() for ln in m.group(1).splitlines() if ln.strip() and not ln.strip().startswith("#")]


def _git_files(entries: list[str]) -> list[Path] | None:
    if shutil.which("git") is None:
        return None
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        top = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=30, env=env)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != ROOT:
            return None
        ls = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--", *entries],
                            capture_output=True, text=True, timeout=60, env=env, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return [ROOT / p for p in ls.stdout.split("\0") if p and (ROOT / p).is_file()]


def shipped_files() -> list[Path]:
    """Files the export ships: tracked files under the allowlist, or (in an
    exported tree, which has no git of its own) every file under it."""
    entries = allowlist()
    files = _git_files(entries)
    if files is not None:
        return files
    out = []
    for e in entries:
        p = ROOT / e
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".pytest_cache")]
                out.extend(Path(dirpath) / f for f in filenames)
    return out


def private_patterns(shipped: list[Path]) -> list[re.Pattern]:
    if not PLANNING.is_dir():
        return []
    shipped_names = {p.name for p in shipped}
    names = {p.name for p in PLANNING.rglob("*" + ".md")} - shipped_names - {"README.md"}
    return [re.compile(r"(?<![\w.-])" + re.escape(n) + r"(?![\w-])") for n in sorted(names)]


def find_refs(text: str, patterns: list[re.Pattern]) -> list[tuple[int, str]]:
    hits = []
    for pat in patterns:
        for m in pat.finditer(text):
            start = m.start()
            # Exempt generated output paths: either the match itself spells the
            # generated directory, or that directory immediately precedes it.
            if m.group(0).startswith(GENERATED_PREFIX) or text[max(0, start - len(GENERATED_PREFIX)):start] == GENERATED_PREFIX:
                continue
            hits.append((text.count("\n", 0, start) + 1, m.group(0)))
    return hits


def test_find_refs_catches_planning_kinds():
    planted = ["see " + "STATUS" + ".md", "per 02-01-" + "SUMMARY" + ".md", "the " + "evid" + "ence/tts-bake" + "off.md",
               "in `01-spikes/" + "evid" + "ence/vlm-bake" + "off.md`", "(" + "DECISION-" + "REQUIRED.md)"]
    for body in planted:
        assert find_refs(body, FIXED), body
    ok = ["writes " + GENERATED_PREFIX + "soak.json", "`" + GENERATED_PREFIX + "DECISION-" + "REQUIRED.md`",
          "see docs/MODELS.md", "docs/DESIGN.md and docs/RUNBOOK.md"]
    for body in ok:
        assert not find_refs(body, FIXED), body


def test_shipped_files_cite_no_private_planning_file():
    shipped = shipped_files()
    assert shipped, "no shipped files found"
    patterns = FIXED + private_patterns(shipped)
    me = Path(__file__).resolve()
    bad = []
    for p in shipped:
        if p.resolve() == me or p.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line, what in find_refs(text, patterns):
            bad.append(f"{p.relative_to(ROOT)}:{line}: {what}")
    assert not bad, "shipped files cite private planning files:\n" + "\n".join(bad)

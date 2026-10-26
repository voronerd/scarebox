#!/usr/bin/env python3
"""Leak gate: refuse to publish a tree that carries private detail.

Usage:
    scripts/leak_gate.py DIR [--env ENVFILE] [--deny-file FILE]

Every file under DIR is scanned, text and binary alike, along with every file
name and symlink target. Three kinds of rule apply:

* built-in rules, shipped here: private (RFC 1918) and carrier-grade NAT /
  tailnet IPv4 addresses, MAC addresses (except the documentation example
  02:00:00:00:00:01), claude dot ai session and artifact links, and email
  addresses other than the two allowed ones;
* the deny file: one literal per line (blank lines and lines starting with
  "#" are ignored), matched case-insensitively as a whole word; this is where
  private words such as hostnames and personal names live, so they are never
  published by the gate itself;
* the env file: every non-empty KEY=VALUE value is matched as a literal.
  Values at least MIN_ENV_LEN characters long match anywhere in a file;
  shorter ones match as a whole token (not touching a letter or digit on
  either side), so a short secret is still caught without every short
  value tripping on the inside of ordinary words. No value is skipped.

Findings go to stdout, one per line, as ``<path relative to DIR>:<line>: <rule>``
(line 0 means the file name, a symlink target or an unreadable file). The
matched text is never printed: rules name the kind of leak, the deny-file line
number or the env key, never the value.

Exit status: 0 clean, 1 findings, 2 usage or input error.

A ``.git`` entry (directory or gitdir file) is not scanned: it is git's
metadata, not published content.
"""

from __future__ import annotations

import argparse
import bisect
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

MIN_ENV_LEN = 6

EXAMPLE_MAC = "02:00:00:00:00:01"
ALLOWED_EMAILS = frozenset({"martin@msawtell.com", "noreply@anthropic.com"})

_IPV4 = re.compile(r"(?<![\d.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?!\.?\d)")
_MAC = re.compile(
    r"(?<![0-9A-Fa-f:-])[0-9A-Fa-f]{2}([:-])[0-9A-Fa-f]{2}(?:\1[0-9A-Fa-f]{2}){4}(?![0-9A-Fa-f:-])"
)
# Cisco-style dotted MAC: three dot-separated groups of four hex digits (hhhh.hhhh.hhhh).
_MAC_DOTTED = re.compile(r"(?<![0-9A-Fa-f.])[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}(?![0-9A-Fa-f.])")
_SESSION_LINK = re.compile(r"claude\.ai/(?:code/)?session", re.IGNORECASE)
_ARTIFACT_LINK = re.compile(r"claude\.ai/(?:code/)?artifact", re.IGNORECASE)
# Domain part of an email, matched at an "@" (the local part is walked back by
# hand so a long run of text without "@" never costs quadratic time).
_EMAIL_DOMAIN = re.compile(r"@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,63})(?![A-Za-z0-9-])")
_EMAIL_LOCAL_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._%+-")
# Underscore is a word BREAK for deny words, so a deny entry also catches identifiers built
# from it (an entry "box" matches BOX_HOST and box_ip, not boxes).
_WORD_CHAR = "A-Za-z0-9"


class GateError(Exception):
    """Bad input: missing directory, unreadable deny or env file."""


@dataclass
class Rules:
    deny: list[tuple[int, str]] = field(default_factory=list)  # (deny-file line, literal)
    env: list[tuple[str, str]] = field(default_factory=list)  # (key, value)
    _deny_re: re.Pattern | None = None
    _deny_index: dict[str, int] = field(default_factory=dict)
    # (key, utf-8 pattern, latin-1 pattern) for values shorter than MIN_ENV_LEN
    _short_env: list[tuple[str, re.Pattern, re.Pattern]] = field(default_factory=list)

    def compile(self) -> None:
        self._short_env = []
        for key, value in self.env:
            if len(value) < MIN_ENV_LEN:
                self._short_env.append(
                    (key, _token_re(value), _token_re(value.encode("utf-8").decode("latin-1")))
                )
        if not self.deny:
            return
        words = sorted({w for _, w in self.deny}, key=len, reverse=True)
        alt = "|".join(re.escape(w) for w in words)
        self._deny_re = re.compile(
            rf"(?<![{_WORD_CHAR}])(?:{alt})(?![{_WORD_CHAR}])", re.IGNORECASE
        )
        for lineno, word in self.deny:
            self._deny_index.setdefault(word.lower(), lineno)

    def deny_line(self, matched: str) -> int:
        hit = self._deny_index.get(matched.lower())
        if hit is not None:
            return hit
        for lineno, word in self.deny:
            if re.fullmatch(re.escape(word), matched, re.IGNORECASE):
                return lineno
        return 0


def load_deny(path: Path) -> list[tuple[int, str]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GateError(f"cannot read deny file {path}: {exc.strerror}") from None
    out = []
    for lineno, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
        word = line.strip()
        if word and not word.startswith("#"):
            out.append((lineno, word))
    return out


def _token_re(value: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])")


def load_env(path: Path) -> list[tuple[str, str]]:
    """Return [(key, value)] for every non-empty value, short ones included."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GateError(f"cannot read env file {path}: {exc.strerror}") from None
    values = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # Unquoted: a " #" starts a comment. Keeping the prefix still
            # catches the secret, since values are matched as substrings.
            cut = value.find(" #")
            if cut != -1:
                value = value[:cut].rstrip()
        if not value:
            continue
        values.append((key, value))
    return values


def _private_ipv4(octets: tuple[int, int, int, int]) -> str | None:
    a, b = octets[0], octets[1]
    if any(o > 255 for o in octets):
        return None
    if a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168):
        return "private-ipv4"
    if a == 100 and 64 <= b <= 127:
        return "cgnat-ipv4"
    return None


def _emails(text: str):
    at = text.find("@")
    while at != -1:
        m = _EMAIL_DOMAIN.match(text, at, min(len(text), at + 256))
        if m:
            start = at
            while start > 0 and at - start < 64 and text[start - 1] in _EMAIL_LOCAL_CHARS:
                start -= 1
            local = text[start:at].lstrip(".")
            if local:
                addr = f"{local}@{m.group(1)}".lower()
                if addr not in ALLOWED_EMAILS:
                    yield at, "email-address"
        at = text.find("@", at + 1)


def scan_text(text: str, rules: Rules, latin1: bool) -> list[tuple[int, str]]:
    """Return (offset, rule) for every hit in text. Never returns matched text."""
    hits: list[tuple[int, str]] = []
    for m in _IPV4.finditer(text):
        rule = _private_ipv4(tuple(int(g) for g in m.groups()))
        if rule:
            hits.append((m.start(), rule))
    for m in _MAC.finditer(text):
        if m.group(0).lower().replace("-", ":") != EXAMPLE_MAC:
            hits.append((m.start(), "mac-address"))
    for m in _MAC_DOTTED.finditer(text):
        hits.append((m.start(), "mac-address"))
    for m in _SESSION_LINK.finditer(text):
        hits.append((m.start(), "claude-session-link"))
    for m in _ARTIFACT_LINK.finditer(text):
        hits.append((m.start(), "claude-artifact-link"))
    hits.extend(_emails(text))
    if rules._deny_re is not None:
        for m in rules._deny_re.finditer(text):
            matched = m.group(0)
            if latin1:
                matched = matched.encode("latin-1").decode("utf-8", "replace")
            hits.append((m.start(), f"deny-word (deny file line {rules.deny_line(matched)})"))
    for key, utf8_re, latin1_re in rules._short_env:
        for m in (latin1_re if latin1 else utf8_re).finditer(text):
            hits.append((m.start(), f"env-value {key}"))
    for key, value in rules.env:
        if len(value) < MIN_ENV_LEN:
            continue  # matched as a token above
        needle = value.encode("utf-8").decode("latin-1") if latin1 else value
        pos = text.find(needle)
        while pos != -1:
            hits.append((pos, f"env-value {key}"))
            pos = text.find(needle, pos + 1)
    return hits


def _views(data: bytes) -> list[tuple[str, bool]]:
    """Text views to scan, each with whether it is a byte-for-byte latin-1 view.

    UTF-8 files are scanned as text. Anything else is scanned byte for byte
    (latin-1), and a file with a UTF-16 byte-order mark is also scanned as
    UTF-16, so neither reading can hide a leak from the other.
    """
    try:
        return [(data.decode("utf-8"), False)]
    except UnicodeDecodeError:
        pass
    views = [(data.decode("latin-1"), True)]
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            views.append((data.decode("utf-16"), False))
        except UnicodeDecodeError:
            pass
    return views


def _line_of(newlines: list[int], offset: int) -> int:
    return bisect.bisect_left(newlines, offset) + 1


def scan_file(path: Path, rules: Rules) -> set[tuple[int, str]]:
    """Return (line, rule) hits for one regular file; raises OSError."""
    out: set[tuple[int, str]] = set()
    for text, latin1 in _views(path.read_bytes()):
        hits = scan_text(text, rules, latin1)
        if hits:
            newlines = [m.start() for m in re.finditer("\n", text)]
            out.update((_line_of(newlines, off), rule) for off, rule in hits)
    return out


def scan_tree(root: Path, rules: Rules) -> tuple[list[tuple[str, int, str]], int]:
    """Return (sorted unique findings as (relpath, line, rule), files scanned)."""
    findings: set[tuple[str, int, str]] = set()
    scanned = 0

    def name_hits(rel: str, value: str, where: str) -> None:
        for _, rule in scan_text(value, rules, latin1=False):
            findings.add((rel, 0, f"{rule} ({where})"))

    def walk_error(exc: OSError) -> None:
        try:
            rel = Path(exc.filename).relative_to(root).as_posix()
        except (TypeError, ValueError):
            rel = "."
        findings.add((rel, 0, "unreadable"))

    for dirpath, dirnames, filenames in os.walk(root, onerror=walk_error):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if d != ".git")
        entries = [(d, True) for d in dirnames] + [(f, False) for f in sorted(filenames) if f != ".git"]
        for name, is_dir in entries:
            full = here / name
            rel = full.relative_to(root).as_posix()
            name_hits(rel, rel, "file name")
            if full.is_symlink():
                try:
                    name_hits(rel, os.readlink(full), "link target")
                except OSError:
                    findings.add((rel, 0, "unreadable"))
                if is_dir:
                    dirnames.remove(name)  # never follow a symlinked directory
                continue
            if is_dir:
                continue
            if not full.is_file():
                continue  # sockets, fifos, devices: nothing to publish
            scanned += 1
            try:
                for line, rule in scan_file(full, rules):
                    findings.add((rel, line, rule))
            except OSError:
                findings.add((rel, 0, "unreadable"))
    return sorted(findings), scanned


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Scan a tree for private detail before it is published."
    )
    ap.add_argument("dir", type=Path, help="directory to scan")
    ap.add_argument("--env", type=Path, help="env file whose values must not appear")
    ap.add_argument("--deny-file", type=Path, help="private words, one per line")
    args = ap.parse_args(argv)

    try:
        if not args.dir.is_dir():
            raise GateError(f"not a directory: {args.dir}")
        rules = Rules()
        if args.deny_file is not None:
            rules.deny = load_deny(args.deny_file)
        if args.env is not None:
            rules.env = load_env(args.env)
        rules.compile()
        findings, scanned = scan_tree(args.dir.resolve(), rules)
    except GateError as exc:
        print(f"leak gate: {exc}", file=sys.stderr)
        return 2

    for rel, line, rule in findings:
        print(f"{rel}:{line}: {rule}")
    if findings:
        files = len({f[0] for f in findings})
        print(f"leak gate: {len(findings)} finding(s) in {files} file(s)", file=sys.stderr)
        return 1
    print(f"leak gate: clean ({scanned} file(s) scanned)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

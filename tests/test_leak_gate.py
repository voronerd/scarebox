"""Tests for scripts/leak_gate.py.

Every pattern the gate must catch is assembled at run time from harmless
pieces, so this file itself passes the gate (see test_gate_sources_pass_gate).
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE = REPO_ROOT / "scripts" / "leak_gate.py"

_spec = importlib.util.spec_from_file_location("leak_gate", GATE)
leak_gate = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("leak_gate", leak_gate)
_spec.loader.exec_module(leak_gate)


def dotted(*parts) -> str:
    return ".".join(str(p) for p in parts)


def mac(*octets: str, sep: str = ":") -> str:
    return sep.join(octets)


def email(local: str, domain: str) -> str:
    return local + "@" + domain


def claude_link(kind: str) -> str:
    return "https://" + "claude" + ".ai/" + kind


# Fake private words and a fake secret: invented, so they never collide with
# anything real.
DENY_WORDS = ["frobnitzbox", "Quillanthe"]
SECRET = "zq" + "-fake-SECRET-" + "4417"
SHORT = "ab" + "c1"


def run_gate(*args, timeout=120):
    return subprocess.run(
        [sys.executable, str(GATE), *map(str, args)],
        capture_output=True, text=True, timeout=timeout,
    )


@pytest.fixture
def deny_file(tmp_path):
    p = tmp_path / "words.deny"
    p.write_text("# comment line\n\n" + "\n".join(DENY_WORDS) + "\n")
    return p


@pytest.fixture
def env_file(tmp_path):
    p = tmp_path / "test.env"
    p.write_text(
        "# comment\n"
        f"AIM_TOKEN={SECRET}\n"
        f"export QUOTED='{SECRET[::-1]}'\n"
        f"TINY={SHORT}\n"
        "EMPTY=\n"
    )
    return p


def plant(tmp_path, body, name="sub/f.txt"):
    d = tmp_path / "tree"
    f = d / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("ok line\n" + body + "\n")
    return d


PLANTS = [
    ("host: " + dotted(192, 168, 7, 9), "private-ipv4"),
    ("url http://" + dotted(172, 20, 1, 1) + ":8090/", "private-ipv4"),
    ("lan " + dotted(10, 1, 2, 3) + ".", "private-ipv4"),
    ("tail " + dotted(100, 99, 1, 2), "cgnat-ipv4"),
    ('mac: "' + mac("a0", "6f", "32", "04", "3e", "db") + '"', "mac-address"),
    ("mac " + mac("A0", "6F", "32", "04", "3E", "DB", sep="-"), "mac-address"),
    ("see " + claude_link("code/session_01XyZ"), "claude-session-link"),
    ("see " + claude_link("artifact/QwErTy12"), "claude-artifact-link"),
    ("mail " + email("someone", "example.org"), "email-address"),
    ("mac " + ".".join(["a06f", "3204", "3edb"]), "mac-address"),
    ("ssh frobnitzbox now", "deny-word (deny file line 3)"),
    ("export " + "frobnitzbox".upper() + "_HOST=1", "deny-word (deny file line 3)"),
    ("thanks QUILLANTHE!", "deny-word (deny file line 4)"),
    ("token = " + SECRET, "env-value AIM_TOKEN"),
    ("q " + SECRET[::-1], "env-value QUOTED"),
]


@pytest.mark.parametrize("body,rule", PLANTS, ids=[r for _, r in PLANTS])
def test_each_rule_is_caught(tmp_path, deny_file, env_file, body, rule):
    d = plant(tmp_path, body)
    r = run_gate(d, "--env", env_file, "--deny-file", deny_file)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert f"sub/f.txt:2: {rule}" in r.stdout.splitlines(), out
    assert SECRET not in out and SECRET[::-1] not in out
    assert "Traceback" not in out


def test_matched_text_is_never_printed(tmp_path, deny_file, env_file):
    bodies = [b for b, _ in PLANTS]
    d = plant(tmp_path, "\n".join(bodies))
    r = run_gate(d, "--env", env_file, "--deny-file", deny_file)
    assert r.returncode == 1
    out = r.stdout + r.stderr
    for needle in [SECRET, SECRET[::-1], *DENY_WORDS, "QUILLANTHE",
                   dotted(192, 168, 7, 9), email("someone", "example.org")]:
        assert needle not in out, needle


def test_allowed_values_are_clean(tmp_path, deny_file):
    d = plant(tmp_path, "\n".join([
        "gbox1 at " + dotted(198, 51, 100, 7) + " and " + dotted(127, 0, 0, 1),
        "bind " + dotted(0, 0, 0, 0) + ", public " + dotted(8, 8, 8, 8),
        "edge " + dotted(172, 32, 0, 1) + " and " + dotted(100, 128, 0, 1),
        "version " + dotted(1, 10, 0, 1, 5),
        "mac " + mac("02", "00", "00", "00", "00", "01") + " time 12:34:56",
        "by " + email("martin", "msawtell.com") + ", " + email("NoReply", "Anthropic.com"),
        "@pytest.mark.skipif and frobnitzboxes and xQuillanthe",
        "Martin wrote this.",
    ]))
    r = run_gate(d, "--deny-file", deny_file)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout == ""


def test_binary_file_is_scanned(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    (d / "x.wav").write_bytes(
        b"RIFF\x00\x01\xff\n\njunk " + dotted(192, 168, 1, 2).encode() + b" more\x00\x00"
    )
    r = run_gate(d)
    assert r.returncode == 1
    assert "x.wav:3: private-ipv4" in r.stdout.splitlines(), r.stdout


def test_non_utf8_deny_and_env_matches(tmp_path, deny_file, env_file):
    d = tmp_path / "bin"
    d.mkdir()
    (d / "blob").write_bytes(b"\xff\xfe\x00" + b"\xc3\x28 " + SECRET.encode() + b" frobnitzbox\x00")
    r = run_gate(d, "--env", env_file, "--deny-file", deny_file)
    lines = r.stdout.splitlines()
    assert "blob:1: env-value AIM_TOKEN" in lines, r.stdout
    assert "blob:1: deny-word (deny file line 3)" in lines, r.stdout


def test_file_names_and_link_targets_are_scanned(tmp_path, deny_file):
    d = tmp_path / "names"
    (d / "frobnitzbox-notes").mkdir(parents=True)
    (d / "frobnitzbox-notes" / "ok.txt").write_text("fine\n")
    (d / "link").symlink_to("/srv/" + dotted(10, 0, 0, 5) + "/data")
    r = run_gate(d, "--deny-file", deny_file)
    lines = r.stdout.splitlines()
    assert r.returncode == 1
    assert "frobnitzbox-notes:0: deny-word (deny file line 3) (file name)" in lines, r.stdout
    assert "link:0: private-ipv4 (link target)" in lines, r.stdout
    assert "frobnitzbox" not in r.stdout.replace("frobnitzbox-notes", "")


def test_git_metadata_is_skipped(tmp_path):
    d = tmp_path / "clone"
    (d / ".git").mkdir(parents=True)
    (d / ".git" / "config").write_text("url = " + email("git", "example.org") + "\n")
    (d / "README.md").write_text("hello\n")
    r = run_gate(d)
    assert r.returncode == 0, r.stdout


def test_short_env_value_is_scanned_not_skipped(tmp_path, env_file):
    # Regression: values under MIN_ENV_LEN used to be skipped with only a
    # stderr note, so a short secret sailed through with exit 0.
    d = plant(tmp_path, "the word " + SHORT + " is here")
    r = run_gate(d, "--env", env_file)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "sub/f.txt:2: env-value TINY" in r.stdout.splitlines()
    assert SHORT not in r.stdout + r.stderr
    assert "not scanned" not in r.stderr


def test_short_secret_with_punctuation_is_caught(tmp_path):
    short = "Qx" + "9!z"
    env = tmp_path / "x.env"
    env.write_text("CAMERA_PASSWORD=" + short + "\n")
    d = plant(tmp_path, "password is " + short)
    r = run_gate(d, "--env", env)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "sub/f.txt:2: env-value CAMERA_PASSWORD" in r.stdout.splitlines()
    assert short not in r.stdout + r.stderr


def test_short_env_value_matches_as_token_only(tmp_path, env_file):
    # Inside a longer word it is not that value; next to punctuation it is.
    d = plant(tmp_path, "x" + SHORT + "y and " + SHORT + "z")
    assert run_gate(d, "--env", env_file).returncode == 0
    d2 = tmp_path / "t2"
    (d2 / "sub").mkdir(parents=True)
    (d2 / "sub" / "f.txt").write_text("user=" + SHORT + ";\n")
    r = run_gate(d2, "--env", env_file)
    assert "sub/f.txt:1: env-value TINY" in r.stdout.splitlines(), r.stdout


def test_short_env_value_in_binary_file(tmp_path, env_file):
    d = tmp_path / "bin"
    d.mkdir()
    (d / "x.wav").write_bytes(b"RIFF\x00\xff " + SHORT.encode() + b" \x00\x00")
    r = run_gate(d, "--env", env_file)
    assert r.returncode == 1
    assert "x.wav:1: env-value TINY" in r.stdout.splitlines(), r.stdout


def test_hostile_and_large_files(tmp_path):
    d = tmp_path / "hostile"
    d.mkdir()
    (d / "bad.txt").write_bytes(b"\xff\xfe\x00bad utf8 \xc3\x28\n" * 10)
    (d / "big.txt").write_text("a" + "@" + "b.co " + "x" * 5_000_000)
    (d / "ats.txt").write_text(("@" + "y" * 300) * 20_000)
    t0 = time.monotonic()
    r = run_gate(d)
    assert time.monotonic() - t0 < 60
    assert r.returncode == 1  # the planted address in big.txt
    assert "big.txt:1: email-address" in r.stdout.splitlines()
    assert "Traceback" not in r.stderr


def test_unreadable_file_is_a_finding(tmp_path):
    d = tmp_path / "locked"
    d.mkdir()
    f = d / "secret.txt"
    f.write_text("x\n")
    f.chmod(0)
    try:
        if _can_read(f):
            pytest.skip("running with privileges that ignore file modes")
        r = run_gate(d)
        assert r.returncode == 1
        assert "secret.txt:0: unreadable" in r.stdout.splitlines()
    finally:
        f.chmod(0o644)


def _can_read(p: Path) -> bool:
    try:
        p.read_bytes()
        return True
    except OSError:
        return False


@pytest.mark.parametrize("extra", [[], ["--deny-file", "missing.deny"], ["--env", "missing.env"]])
def test_bad_input_exits_2_without_traceback(tmp_path, extra):
    d = tmp_path / "ok"
    d.mkdir()
    target = tmp_path / "missing-dir" if not extra else d
    args = [target] + [tmp_path / a if a.startswith("missing") else a for a in extra]
    r = run_gate(*args)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "Traceback" not in r.stderr


def test_load_env_parsing(tmp_path):
    p = tmp_path / "e.env"
    p.write_text(
        'A="quoted value"\n'
        "B=plain-value # trailing comment\n"
        "  export C = spaced-value  \n"
        "not a pair\n"
    )
    values = leak_gate.load_env(p)
    assert values == [("A", "quoted value"), ("B", "plain-value"), ("C", "spaced-value")]


def test_load_env_keeps_short_values(tmp_path, env_file):
    values = leak_gate.load_env(env_file)
    assert ("TINY", SHORT) in values
    assert all(v for _, v in values)  # EMPTY= is not a value


def test_gate_sources_pass_gate(tmp_path):
    """The gate and its tests must themselves be publishable."""
    d = tmp_path / "self"
    d.mkdir()
    shutil.copy(GATE, d / "leak_gate.py")
    shutil.copy(Path(__file__), d / "test_leak_gate.py")
    r = run_gate(d)
    assert r.returncode == 0, r.stdout


def test_utf16_text_is_scanned(tmp_path):
    d = tmp_path / "u16"
    d.mkdir()
    (d / "notes.txt").write_bytes(("a\nhost " + dotted(10, 2, 3, 4) + "\n").encode("utf-16"))
    r = run_gate(d)
    assert r.returncode == 1
    assert "notes.txt:2: private-ipv4" in r.stdout.splitlines(), r.stdout

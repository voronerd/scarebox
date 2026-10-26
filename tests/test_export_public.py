"""Tests for scripts/export_public.sh.

Each test builds a throwaway git repo holding a copy of the export script and
the leak gate, so the tests are hermetic: they never read the real repo's
.env or deny list, and they run the same from the working repo or from an
exported tree (which has no .git of its own).

Planted leaks are assembled at run time so this file itself passes the gate.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "export_public.sh"
GATE = ROOT / "scripts" / "leak_gate.py"

pytestmark = [
    pytest.mark.skipif(shutil.which("git") is None, reason="needs the git binary"),
    pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash"),
    pytest.mark.skipif(not SCRIPT.exists() or not GATE.exists(), reason="export script or gate missing"),
]

# A private address and a deny word, built so neither appears literally here.
PRIVATE_IP = ".".join(["10", "9", "8", "7"])
DENY_WORD = "zq" + "plonk"
ENV_VALUE = "v4lue-" + "Q" * 8 + "-xyz"


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="noreply@anthropic.com",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="noreply@anthropic.com",
    )
    return env


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, env=_env(),
                   capture_output=True, text=True, timeout=60)


def _write(base: Path, rel: str, text: str) -> None:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A small git repo shaped like scarebox, with tracked and untracked files."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    (r / "scripts").mkdir()
    shutil.copy(SCRIPT, r / "scripts" / "export_public.sh")
    shutil.copy(GATE, r / "scripts" / "leak_gate.py")
    tracked = {
        "scarebox/__init__.py": "",
        "scarebox/core.py": "x = 1\n",
        "tests/test_core.py": "def test_x():\n    pass\n",
        "config/scarebox.example.yaml": "a: 1\n",
        "library/manifest.json": "{}\n",
        "library/sfx/SOURCES.md": "made here\n",
        "library/candidates/raw.json": "{}\n",  # tracked but not allowlisted
        ".planning" + "/notes.md": "private\n",  # tracked but not allowlisted (split so the
        # exported copy of this file carries no planning-dir path for the publish check)
        "README.md": "hello\n",
        "LICENSE": "license\n",
        "pyproject.toml": "[project]\nname='x'\n",
        ".gitignore": "config/local.yaml\n.env\nscripts/leak_gate.deny\n__pycache__/\n",
        ".env.example": "AIM_TOKEN=\n",
        "notes-at-root.md": "not allowlisted\n",
    }
    for rel, text in tracked.items():
        _write(r, rel, text)
    _git(r, "add", "--", "scripts", *tracked)
    # Untracked or ignored files under allowlisted paths must not ship.
    _write(r, "scarebox/scratch.py", "untracked\n")
    _write(r, "config/local.yaml", "private: 1\n")
    _write(r, "scarebox/__pycache__/core.cpython-311.pyc", "bytecode\n")
    return r


def _export(repo: Path, dest: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(repo / "scripts" / "export_public.sh"), str(dest)],
                          env=_env(), capture_output=True, text=True, timeout=120)


def test_exports_only_tracked_allowlisted_files(repo: Path, tmp_path: Path) -> None:
    dest = tmp_path / "pub"
    r = _export(repo, dest)
    assert r.returncode == 0, r.stdout + r.stderr
    for rel in ["scarebox/__init__.py", "scarebox/core.py", "tests/test_core.py",
                "config/scarebox.example.yaml", "library/manifest.json", "library/sfx/SOURCES.md",
                "scripts/export_public.sh", "scripts/leak_gate.py", "README.md", "LICENSE",
                "pyproject.toml", ".gitignore", ".env.example"]:
        assert (dest / rel).is_file(), rel
    for rel in ["library/candidates", ".planning", "notes-at-root.md", "scarebox/scratch.py",
                "config/local.yaml", "scarebox/__pycache__", ".git"]:
        assert not (dest / rel).exists(), rel


def test_exports_working_tree_content_and_skips_deleted(repo: Path, tmp_path: Path) -> None:
    (repo / "scarebox" / "core.py").write_text("x = 2\n")  # edited, not re-staged
    (repo / "README.md").unlink()  # tracked but deleted
    dest = tmp_path / "pub"
    r = _export(repo, dest)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (dest / "scarebox" / "core.py").read_text() == "x = 2\n"
    assert not (dest / "README.md").exists()


def test_gate_findings_make_export_fail(repo: Path, tmp_path: Path) -> None:
    _write(repo, "docs/host.md", f"ok\nserver at {PRIVATE_IP}\n")
    _git(repo, "add", "--", "docs/host.md")
    dest = tmp_path / "pub"
    r = _export(repo, dest)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "docs/host.md:2" in r.stdout


def test_uses_repo_deny_file_and_env(repo: Path, tmp_path: Path) -> None:
    _write(repo, "docs/a.md", f"line\nhello {DENY_WORD.upper()}\n")
    _write(repo, "docs/b.md", f"token {ENV_VALUE}\n")
    _git(repo, "add", "--", "docs")
    # Without the private files the export is clean ...
    r = _export(repo, tmp_path / "pub0")
    assert r.returncode == 0, r.stdout + r.stderr
    # ... with them, both are found and the secret value is never printed.
    _write(repo, "scripts/leak_gate.deny", DENY_WORD + "\n")
    _write(repo, ".env", f"AIM_TOKEN={ENV_VALUE}\n")
    r = _export(repo, tmp_path / "pub1")
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "docs/a.md:2" in out and "docs/b.md:1" in out, out
    assert ENV_VALUE not in out
    assert not (tmp_path / "pub1" / ".env").exists()
    assert not (tmp_path / "pub1" / "scripts" / "leak_gate.deny").exists()


def test_linked_worktree_falls_back_to_main_private_files(repo: Path, tmp_path: Path) -> None:
    _write(repo, "docs/a.md", f"hello {DENY_WORD}\n")
    _git(repo, "add", "--", "docs")
    _git(repo, "commit", "-q", "--no-verify", "-m", "base")
    _write(repo, "scripts/leak_gate.deny", DENY_WORD + "\n")
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(wt))
    assert not (wt / "scripts" / "leak_gate.deny").exists()
    r = _export(wt, tmp_path / "pub")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "docs/a.md:1" in r.stdout


def test_refuses_non_empty_dest(repo: Path, tmp_path: Path) -> None:
    dest = tmp_path / "full"
    dest.mkdir()
    (dest / "keep.txt").write_text("mine")
    r = _export(repo, dest)
    assert r.returncode == 2, r.stdout + r.stderr
    assert sorted(p.name for p in dest.iterdir()) == ["keep.txt"]
    assert (dest / "keep.txt").read_text() == "mine"


def test_refuses_dest_that_is_a_file(repo: Path, tmp_path: Path) -> None:
    dest = tmp_path / "afile"
    dest.write_text("x")
    r = _export(repo, dest)
    assert r.returncode == 2
    assert dest.read_text() == "x"


def test_fills_dest_holding_only_git(repo: Path, tmp_path: Path) -> None:
    dest = tmp_path / "clone"
    (dest / ".git").mkdir(parents=True)
    (dest / ".git" / "HEAD").write_text("ref")
    r = _export(repo, dest)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (dest / "scarebox" / "__init__.py").exists()
    assert (dest / ".git" / "HEAD").read_text() == "ref"


def test_fills_empty_existing_dest(repo: Path, tmp_path: Path) -> None:
    dest = tmp_path / "empty"
    dest.mkdir()
    r = _export(repo, dest)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (dest / "README.md").exists()


def test_usage_error(repo: Path) -> None:
    r = subprocess.run(["bash", str(repo / "scripts" / "export_public.sh")],
                       env=_env(), capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "usage" in r.stderr

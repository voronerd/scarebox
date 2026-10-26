from pathlib import Path

import pytest

from scarebox.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIG = REPO_ROOT / "config" / "scarebox.example.yaml"
FIELD_CONFIG = REPO_ROOT / "config" / "field.yaml"
LOCAL_EXAMPLE = REPO_ROOT / "config" / "local.example.yaml"
FAKE_ENV = {
    "CAMERA_HOST": "192.0.2.99",
    "CAMERA_USER": "fake-user",
    "CAMERA_PASSWORD": "fake-pass",
    "AIM_TOKEN": "fake-token",
}


@pytest.fixture
def fake_env(monkeypatch):
    for key, value in FAKE_ENV.items():
        monkeypatch.setenv(key, value)
    return FAKE_ENV


def test_example_config_loads_with_fake_env(fake_env):
    cfg = load_config(EXAMPLE_CONFIG)
    assert cfg.camera.host == FAKE_ENV["CAMERA_HOST"]
    assert cfg.camera.user == FAKE_ENV["CAMERA_USER"]
    assert cfg.camera.password == FAKE_ENV["CAMERA_PASSWORD"]
    assert cfg.aim.listen.endswith(":8090")
    # Every section from config/scarebox.example.yaml is present.
    for section in (
        "camera",
        "source",
        "perceive",
        "judge",
        "quip",
        "censor",
        "voice",
        "stage",
        "orchestrator",
        "library",
        "aim",
    ):
        assert getattr(cfg, section) is not None
    assert cfg.perceive.trigger_line == ((0.55, 0.0), (0.55, 1.0))


def test_unknown_key_raises(fake_env, tmp_path):
    text = EXAMPLE_CONFIG.read_text()
    bad_text = text.replace(
        "camera:\n", "camera:\n  bogus_key: 1\n", 1
    )
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(bad_text)

    with pytest.raises(ConfigError, match="bogus_key"):
        load_config(bad_path)


def test_missing_var_raises(monkeypatch, tmp_path):
    # No process env vars set, and env_file points somewhere with nothing in it.
    for key in FAKE_ENV:
        monkeypatch.delenv(key, raising=False)
    empty_env_file = tmp_path / "empty.env"
    empty_env_file.write_text("")

    with pytest.raises(ConfigError, match="CAMERA_HOST"):
        load_config(EXAMPLE_CONFIG, env_file=empty_env_file)


def test_profile_switch_picks_env_field(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCAREBOX_PROFILE", "field")
    for key in FAKE_ENV:
        monkeypatch.delenv(key, raising=False)

    env_field = tmp_path / ".env.field"
    env_field.write_text(
        "\n".join(f"{k}={v}" for k, v in FAKE_ENV.items()) + "\n"
    )
    # A stray plain .env must NOT be picked up when the profile is "field".
    (tmp_path / ".env").write_text("CAMERA_HOST=wrong-host\n")

    cfg = load_config(EXAMPLE_CONFIG)
    assert cfg.camera.host == FAKE_ENV["CAMERA_HOST"]


def _field_copy(tmp_path, name="field.yaml"):
    """config/field.yaml copied into tmp_path, so a test controls its local.yaml overlay
    (the repo's own config/local.yaml, if any, belongs to the installation)."""
    dest = tmp_path / name
    dest.write_text(FIELD_CONFIG.parent.joinpath(name).read_text())
    return dest


def test_field_config_loads_and_locks_in_race_safety(fake_env, tmp_path):
    """config/field.yaml (04-01 Task 2): the race profile must load, and must carry the two
    settings that keep race day safe -- an enforced, fully-vetted library, and the real
    hardware audio sink (never a silent WAV fallback)."""
    cfg = load_config(_field_copy(tmp_path))
    assert cfg.library.require_approved is True
    assert cfg.stage.sink == "alsa"
    # Found by MAC, not a fixed address -- the travel router's DHCP range is unpredictable.
    # The MAC itself is an installation value: it lives in the overlay, not the tracked file.
    assert cfg.camera.host == ""
    assert cfg.camera.mac == ""
    assert cfg.aim.trusted_networks == ()


def test_field_config_with_example_overlay(fake_env, tmp_path):
    base = _field_copy(tmp_path)
    (tmp_path / "local.yaml").write_text(LOCAL_EXAMPLE.read_text())
    cfg = load_config(base)
    assert cfg.camera.mac == "02:00:00:00:00:01"
    assert cfg.aim.trusted_networks == ("198.51.100.0/24",)
    # Everything the overlay doesn't name is the base's, untouched by the merge.
    assert cfg.camera.host == ""
    assert cfg.camera.user == FAKE_ENV["CAMERA_USER"]
    assert cfg.aim.listen == "0.0.0.0:8090"
    assert cfg.library.require_approved is True


def test_overlay_applies_to_any_config_in_the_directory(fake_env, tmp_path):
    base = _field_copy(tmp_path, "replay.yaml")
    (tmp_path / "local.yaml").write_text('camera:\n  mac: "02:00:00:00:00:01"\n')
    cfg = load_config(base)
    assert cfg.camera.mac == "02:00:00:00:00:01"
    assert cfg.aim.tuning_file != "runs/tuning.json"


def test_overlay_values_get_env_expansion(fake_env, tmp_path, monkeypatch):
    base = _field_copy(tmp_path)
    monkeypatch.setenv("CAMERA_HOST", "192.0.2.50")
    (tmp_path / "local.yaml").write_text("camera:\n  host: ${CAMERA_HOST}\n")
    assert load_config(base).camera.host == "192.0.2.50"


@pytest.mark.parametrize("text", ["", "# just a comment\n", "{}\n"])
def test_empty_overlay_changes_nothing(fake_env, tmp_path, text):
    base = _field_copy(tmp_path)
    before = load_config(base)
    (tmp_path / "local.yaml").write_text(text)
    assert load_config(base) == before


@pytest.mark.parametrize("text, match", [
    ("camera:\n  no_such_key: 1\n", "no_such_key"),
    ("nosuchsection:\n  a: 1\n", "nosuchsection"),
    ("- a\n- b\n", "expected a mapping"),
    ("just a string\n", "expected a mapping"),
    ("camera: 5\n", "expected a mapping"),
    ("camera: [1, 2\n", "not valid YAML"),
])
def test_bad_overlay_raises(fake_env, tmp_path, text, match):
    base = _field_copy(tmp_path)
    (tmp_path / "local.yaml").write_text(text)
    with pytest.raises(ConfigError, match=match):
        load_config(base)


def test_overlay_wrong_value_type_fails_validation_like_base(fake_env, tmp_path):
    # The merged result goes through the same validation: a section replaced by null is
    # caught (here as "expected a mapping"), not silently accepted.
    base = _field_copy(tmp_path)
    (tmp_path / "local.yaml").write_text("camera: null\n")
    with pytest.raises(ConfigError):
        load_config(base)


def test_local_yaml_is_git_ignored():
    gitignore = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert "config/local.yaml" in gitignore


def test_example_overlay_documents_every_installation_key():
    import yaml

    data = yaml.safe_load(LOCAL_EXAMPLE.read_text())
    assert data["camera"]["mac"] == "02:00:00:00:00:01"
    assert data["aim"]["trusted_networks"] == ["198.51.100.0/24"]


def test_tuning_file_defaults_and_replay_keeps_its_own(fake_env, tmp_path):
    # The live box and a replay run side by side; sharing one settings file let the replay
    # inherit the live webcam crop (2026-10-19).
    assert load_config(_field_copy(tmp_path)).aim.tuning_file == "runs/tuning.json"
    replay = load_config(_field_copy(tmp_path, "replay.yaml"))
    assert replay.aim.tuning_file != "runs/tuning.json"

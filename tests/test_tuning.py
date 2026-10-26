"""tests/test_tuning.py (04-04): pure-logic unit tests for `Tuning` -- clamping, validation,
and the load/save round trip. Pure stdlib, same bar as `tests/test_aim.py`.
"""

from __future__ import annotations

import types

import pytest

from scarebox.tuning import (
    BOOL_FIELDS,
    RANGES,
    Tuning,
    apply_to_orchestrator,
    apply_to_stage,
    defaults_from_cfg,
    load_tuning,
    save_tuning,
    validate_and_update,
)


def _cfg():
    return types.SimpleNamespace(
        perceive=types.SimpleNamespace(min_bbox_height=0.25, min_confidence=0.5, group_window_s=1.5),
        stage=types.SimpleNamespace(cooldown_s=1.0),
        orchestrator=types.SimpleNamespace(budget_s=3.5),
        quip=types.SimpleNamespace(roast_non_costumes=True),
    )


# -- defaults from cfg -----------------------------------------------------------------------


def test_defaults_from_cfg_mirrors_config_values():
    tuning = defaults_from_cfg(_cfg())
    assert tuning.min_bbox_height == 0.25
    assert tuning.detector_min_confidence == 0.5
    assert tuning.stage_cooldown_s == 1.0
    assert tuning.orchestrator_budget_s == 3.5
    assert tuning.group_window_s == 1.5
    assert tuning.live_lines_enabled is True
    assert tuning.roast_non_costumes is True
    # Applied to the dongle at startup: must match the 80% race level, not 100 (2026-10-19).
    assert tuning.volume_pct == 80.0


def test_budget_s_property_mirrors_orchestrator_budget_s():
    tuning = defaults_from_cfg(_cfg())
    assert tuning.budget_s == tuning.orchestrator_budget_s


# -- clamping ---------------------------------------------------------------------------------


@pytest.mark.parametrize("field", list(RANGES))
def test_clamp_pulls_out_of_range_values_into_range(field):
    tuning = defaults_from_cfg(_cfg())
    setattr(tuning, field, RANGES[field].hi + 1000.0)
    tuning.clamp()
    assert getattr(tuning, field) == RANGES[field].hi
    setattr(tuning, field, RANGES[field].lo - 1000.0)
    tuning.clamp()
    assert getattr(tuning, field) == RANGES[field].lo


def test_clamp_coerces_bool_fields():
    tuning = defaults_from_cfg(_cfg())
    tuning.live_lines_enabled = 1  # truthy non-bool, as a stray JSON load might produce
    tuning.clamp()
    assert tuning.live_lines_enabled is True


# -- validate_and_update (what POST /settings drives) ------------------------------------------


def test_validate_and_update_applies_known_fields():
    tuning = defaults_from_cfg(_cfg())
    ok, err = validate_and_update(tuning, {"stage_cooldown_s": 2.5, "live_lines_enabled": False})
    assert ok is True
    assert err is None
    assert tuning.stage_cooldown_s == 2.5
    assert tuning.live_lines_enabled is False


def test_validate_and_update_clamps_out_of_range():
    tuning = defaults_from_cfg(_cfg())
    ok, _err = validate_and_update(tuning, {"min_bbox_height": 99.0})
    assert ok is True
    assert tuning.min_bbox_height == RANGES["min_bbox_height"].hi


def test_validate_and_update_rejects_unknown_key_and_leaves_tuning_unchanged():
    tuning = defaults_from_cfg(_cfg())
    before = tuning.as_dict()
    ok, err = validate_and_update(tuning, {"not_a_real_setting": 1})
    assert ok is False
    assert "not_a_real_setting" in err
    assert tuning.as_dict() == before


def test_validate_and_update_rejects_wrong_type_and_leaves_tuning_unchanged():
    tuning = defaults_from_cfg(_cfg())
    before = tuning.as_dict()
    ok, err = validate_and_update(tuning, {"live_lines_enabled": "yes"})
    assert ok is False
    assert err is not None
    assert tuning.as_dict() == before


def test_validate_and_update_rejects_non_numeric_for_range_field():
    tuning = defaults_from_cfg(_cfg())
    ok, err = validate_and_update(tuning, {"stage_cooldown_s": "not a number"})
    assert ok is False
    assert err is not None


def test_validate_and_update_partial_patch_leaves_other_fields_alone():
    tuning = defaults_from_cfg(_cfg())
    validate_and_update(tuning, {"stage_cooldown_s": 4.0})
    assert tuning.min_bbox_height == 0.25  # untouched


# -- load/save round trip (persistence, loaded over config defaults) -------------------------


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "tuning.json"
    tuning = defaults_from_cfg(_cfg())
    tuning.stage_cooldown_s = 7.0
    tuning.live_lines_enabled = False
    save_tuning(tuning, path)

    reloaded = load_tuning(_cfg(), path)
    assert reloaded.stage_cooldown_s == 7.0
    assert reloaded.live_lines_enabled is False


def test_load_with_no_saved_file_returns_cfg_defaults(tmp_path):
    tuning = load_tuning(_cfg(), tmp_path / "missing.json")
    assert tuning.stage_cooldown_s == 1.0


def test_load_with_corrupt_file_falls_back_to_cfg_defaults(tmp_path):
    path = tmp_path / "tuning.json"
    path.write_text("{not json")
    tuning = load_tuning(_cfg(), path)
    assert tuning.stage_cooldown_s == 1.0


def test_load_ignores_unknown_saved_keys(tmp_path):
    path = tmp_path / "tuning.json"
    path.write_text('{"unknown_field": 123, "stage_cooldown_s": 5.0}')
    tuning = load_tuning(_cfg(), path)
    assert tuning.stage_cooldown_s == 5.0
    assert not hasattr(tuning, "unknown_field")


def test_save_never_leaves_a_tmp_file_behind(tmp_path):
    path = tmp_path / "tuning.json"
    save_tuning(defaults_from_cfg(_cfg()), path)
    assert path.exists()
    assert not (tmp_path / f".{path.name}.tmp").exists()


# -- apply_to_* (push into the running pipeline) -----------------------------------------------


def test_apply_to_stage_sets_cooldown():
    stage = types.SimpleNamespace(cooldown_s=1.0)
    tuning = defaults_from_cfg(_cfg())
    tuning.stage_cooldown_s = 9.0
    apply_to_stage(tuning, stage)
    assert stage.cooldown_s == 9.0


def test_line_age_is_a_setting_and_reaches_the_stage():
    # Speaker distance is only known on the day, so the stale limit is a phone-page slider.
    stage = types.SimpleNamespace(cooldown_s=1.0, max_line_age_s=5.0)
    tuning = defaults_from_cfg(_cfg())
    tuning.max_line_age_s = 12.0
    apply_to_stage(tuning, stage)
    assert stage.max_line_age_s == 12.0
    ok, _ = validate_and_update(tuning, {"max_line_age_s": 40})
    assert ok and tuning.max_line_age_s == 15.0  # clamped to the slider's top


def test_apply_to_orchestrator_hands_over_the_tuning_object():
    orchestrator = types.SimpleNamespace(tuning=None)
    tuning = defaults_from_cfg(_cfg())
    apply_to_orchestrator(tuning, orchestrator)
    assert orchestrator.tuning is tuning


def test_all_fields_covers_every_range_and_bool_field():
    from scarebox.tuning import ALL_FIELDS

    assert ALL_FIELDS == frozenset(RANGES) | frozenset(BOOL_FIELDS)


# -- crop (digital pan/zoom, no PTZ on the USB webcam) ----------------------------------------


def test_default_crop_is_full_frame():
    tuning = defaults_from_cfg(_cfg())
    assert tuning.crop == (0.0, 0.0, 1.0, 1.0)


def test_validate_and_update_accepts_a_valid_crop():
    from scarebox.tuning import validate_and_update

    tuning = defaults_from_cfg(_cfg())
    ok, err = validate_and_update(tuning, {"crop": [0.1, 0.1, 0.9, 0.9]})
    assert ok is True
    assert err is None
    assert tuning.crop == (0.1, 0.1, 0.9, 0.9)


def test_validate_and_update_rejects_too_small_crop():
    from scarebox.tuning import validate_and_update

    tuning = defaults_from_cfg(_cfg())
    before = tuning.crop
    ok, err = validate_and_update(tuning, {"crop": [0.4, 0.4, 0.45, 0.45]})
    assert ok is False
    assert "minimum size" in err
    assert tuning.crop == before


def test_validate_and_update_rejects_backwards_crop():
    from scarebox.tuning import validate_and_update

    tuning = defaults_from_cfg(_cfg())
    ok, err = validate_and_update(tuning, {"crop": [0.9, 0.1, 0.1, 0.9]})
    assert ok is False
    assert err is not None


def test_validate_and_update_rejects_out_of_bounds_crop():
    from scarebox.tuning import validate_and_update

    tuning = defaults_from_cfg(_cfg())
    ok, err = validate_and_update(tuning, {"crop": [-0.1, 0.0, 0.9, 0.9]})
    assert ok is False
    assert err is not None


def test_clamp_crop_fixes_swapped_coords():
    from scarebox.tuning import clamp_crop

    assert clamp_crop((0.9, 0.1, 0.1, 0.9)) == (0.1, 0.1, 0.9, 0.9)


def test_clamp_crop_grows_a_too_small_rectangle():
    from scarebox.tuning import MIN_CROP_SIZE, clamp_crop

    x0, y0, x1, y1 = clamp_crop((0.4, 0.4, 0.45, 0.45))
    assert (x1 - x0) >= MIN_CROP_SIZE - 1e-9
    assert (y1 - y0) >= MIN_CROP_SIZE - 1e-9


def test_crop_round_trips_through_save_load(tmp_path):
    path = tmp_path / "tuning.json"
    tuning = defaults_from_cfg(_cfg())
    tuning.crop = (0.2, 0.2, 0.8, 0.8)
    save_tuning(tuning, path)
    reloaded = load_tuning(_cfg(), path)
    assert reloaded.crop == (0.2, 0.2, 0.8, 0.8)


def test_apply_crop_to_frame_shape_and_mapping():
    import numpy as np

    from scarebox.tuning import apply_crop_to_frame

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(200, dtype=np.uint8)  # a horizontal gradient in channel 0
    cropped = apply_crop_to_frame(image, (0.25, 0.0, 0.75, 1.0))
    assert cropped.shape == (100, 100, 3)
    # Cropping [0.25, 0.75) of a 200px-wide frame starts at pixel 50.
    assert cropped[0, 0, 0] == image[0, 50, 0]


def test_apply_crop_to_frame_full_frame_is_a_noop():
    import numpy as np

    from scarebox.tuning import apply_crop_to_frame

    image = np.zeros((50, 80, 3), dtype=np.uint8)
    cropped = apply_crop_to_frame(image, (0.0, 0.0, 1.0, 1.0))
    assert cropped.shape == image.shape

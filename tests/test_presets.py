"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink und Auswahlregler."""

import zkt_constants as C
import zkt_evaluation as ev
import zkt_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert preset["kind"] in C.KINDS
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.B_MIN <= preset["b"] <= C.B_MAX and C.K_MIN <= preset["k"] <= C.K_MAX
        assert C.S_MIN <= preset["s"] <= C.S_MAX and C.ALPHA_MIN <= preset["alpha"] <= C.ALPHA_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_k_never_exceeds_the_smallest_network():
    assert C.K_MAX <= C.N_MIN


def test_clamp_value_limits_numbers_and_rejects_unknown_choices():
    assert P.clamp_value(P.SETTING_SPECS["b_slider"], 5.0) == C.B_MAX and P.clamp_value(P.SETTING_SPECS["alpha_slider"], 0.1) == C.ALPHA_MIN
    assert P.clamp_value(P.SETTING_SPECS["k_slider"], 99) == C.K_MAX and P.clamp_value(P.SETTING_SPECS["s_slider"], 0.333) == 0.33
    assert P.clamp_value(P.SETTING_SPECS["kind_radio"], "ring") == "ring" and P.clamp_value(P.SETTING_SPECS["kind_radio"], "x") is None


def test_kinds_have_labels_and_colors():
    assert set(C.KIND_LABEL) == set(C.KINDS) == set(C.COLORS)


def test_the_s_grid_contains_the_values_named_in_the_readme():
    for s in (0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0):
        assert s in C.S_GRID
    assert C.S_GRID[0] == 0.0 and C.S_GRID[-1] == 2.5

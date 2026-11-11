"""--preset: cambia defaults, los flags explícitos ganan, y sin preset nada cambia."""
import pytest

import runner
from core.presets import PRESETS

BASE = ["--technique", "t3", "--dataset", "iris"]


def test_no_preset_keeps_historical_defaults():
    a = runner.parse_args(BASE)
    assert (a.preset, a.l2_form, a.lambda_l2, a.selection_metric, a.selection_se,
            a.n_sessions, a.warm_percent) == (None, "mean", 10.0, "acc", 0.0, 15, None)


def test_paper2_sets_protocol():
    a = runner.parse_args(BASE + ["--preset", "paper2"])
    for k, v in PRESETS["paper2"].items():
        assert getattr(a, k) == v, k
    assert a.warm_particles == 15          # 30 % de las 50 partículas por defecto


def test_explicit_flag_wins():
    a = runner.parse_args(BASE + ["--lambda-l2", "3", "--preset", "paper2", "--selection-se", "0"])
    assert a.lambda_l2 == 3.0 and a.selection_se == 0.0
    assert a.l2_form == "sample"


def test_unknown_preset_rejected():
    with pytest.raises(SystemExit):
        runner.parse_args(BASE + ["--preset", "nope"])


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_keys_are_runner_flags(name):
    runner.parse_args(BASE + ["--preset", name])   # lanza ValueError si alguna clave no existe

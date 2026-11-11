"""Fórmulas de core/budget.py: contra números conocidos y contra el conteo real."""
import pytest
import torch

from core.budget import METER, counted, evals_per_session, iters_for_budget

KW = dict(n_particles=100, n_layers=3, warm_particles=30)


def test_formulas_known_values():
    # Piloto MCW (output/paper2-cnn-mcw): T3 gastó 600,390 y T4 600,330.
    assert evals_per_session("t3", 1000, **KW) == 3 * (100 + 30 + 2 * 1000 * 100)
    assert evals_per_session("t3", 1000, **KW) == 600_390
    assert iters_for_budget("t4", 600_390, **KW) == 6002
    assert evals_per_session("t4", 6002, **KW) == 600_330
    assert evals_per_session("t3", 10, reevaluate_pbest=False, **KW) == 3 * (130 + 10 * 100)
    assert evals_per_session("t1", 10, n_rounds=2, **KW) == 2 * 3 * (130 + 10 * 100)
    assert evals_per_session("t2-jacobi", 10, n_rounds=2, **KW) == 3 * (130 + 2 * 100 + 2 * 10 * 100)


def test_warm_particles_clamped():
    assert evals_per_session("t4", 5, n_particles=10, n_layers=1, warm_particles=0) == 10 + 1 + 50
    assert evals_per_session("t4", 5, n_particles=10, n_layers=1, warm_particles=99) == 10 + 10 + 50


@pytest.mark.parametrize("tech", ["t1", "t2", "t2-jacobi", "t3", "t4"])
@pytest.mark.parametrize("budget", [5_000, 77_777, 600_390])
def test_iters_for_budget_is_largest_that_fits(tech, budget):
    kw = dict(KW, n_rounds=2)
    it = iters_for_budget(tech, budget, **kw)
    assert evals_per_session(tech, it, **kw) <= budget < evals_per_session(tech, it + 1, **kw)


def test_iters_for_budget_too_small():
    with pytest.raises(ValueError):
        iters_for_budget("t3", 10, **KW)


def test_unknown_technique():
    with pytest.raises(ValueError):
        evals_per_session("t9", 1, **KW)


def test_counted_wrapper():
    METER.start()
    f = counted(lambda x: x.sum(dim=-1) if x.dim() > 1 else x.sum())
    f(torch.zeros(7, 3))
    f(torch.zeros(3))
    snap = METER.snapshot()
    assert snap["fitness_evals"] == 8 and snap["fitness_calls"] == 2


@pytest.mark.parametrize("tech", ["t1", "t2", "t2-jacobi", "t3", "t4"])
def test_formula_matches_real_count(tech, tmp_path):
    """Una sesión real en CPU gasta exactamente lo que dice la fórmula."""
    import runner
    args = runner.parse_args([
        "--technique", tech, "--dataset", "iris", "--preset", "paper2",
        "--n-particles", "6", "--max-iter", "3", "--patience", "1000",
        "--device", "cpu", "--dtype", "float64", "--output-dir", str(tmp_path)])
    data = runner._prepare(args, torch.device("cpu"), torch.float64, 100)
    res = runner._build_cfg_and_run(args=args, device=torch.device("cpu"), dtype=torch.float64,
                                    data=data, output_dir=tmp_path / tech, data_seed=100)
    assert res["sessions"][0]["fitness_evals"] == res["config"]["planned_evals_per_session"]

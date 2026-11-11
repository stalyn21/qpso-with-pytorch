"""Ganancia por capa (core/gain.py): W_eff = exp(u)·W, sólo pesos, L2 sobre W_eff."""
import math

import pytest
import torch

from core.bounds import dim_bounds
from core.cnn import TinyCNN
from core.diagnostics import param_stats
from core.gain import log_gain_range, validate_gain_range
from core.mlp import MLP
from core.sessions import init_params_from_swarms

DT = torch.float64
CFG = {"bounds_mode": "fixed", "bounds": (-1.0, 1.0), "edge_tol": 1e-3,
       "gain_range": (0.1, 2.0)}


def _pair(kind):
    if kind == "mlp":
        mk = lambda g: MLP([4, 6, 3], dtype=DT, layer_gain=g)
        X = torch.randn(7, 4, dtype=DT)
    else:
        mk = lambda g: TinyCNN(in_channels=1, image_size=10, channels=(2,), n_classes=3,
                               head="flatten", dtype=DT, layer_gain=g)
        X = torch.randn(7, 1, 10, 10, dtype=DT)
    return mk(False), mk(True), X, torch.randint(0, 3, (7,))


def _with_gain(plain, gained, flat, u):
    """Inserta la coordenada u de cada capa en un vector sin ganancia."""
    parts = [torch.cat([flat[s.offset : s.offset + s.size], torch.tensor([u[k]], dtype=DT)])
             for k, s in enumerate(plain.layer_specs)]
    out = torch.cat(parts)
    assert out.numel() == gained.d_total
    return out


@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_layout(kind):
    plain, gained, _, _ = _pair(kind)
    for a, b in zip(plain.layer_specs, gained.layer_specs):
        assert b.size == a.size + 1 and b.gain and not a.gain
    assert gained.d_total == plain.d_total + plain.num_layers
    assert gained.n_net == plain.n_net == plain.d_total
    total = gained.weight_mask().sum() + gained.bias_mask().sum() + gained.gain_mask().sum()
    assert int(total) == gained.d_total
    assert not bool((gained.weight_mask() & gained.gain_mask()).any())


@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_unit_gain_is_the_plain_network(kind):
    torch.manual_seed(0)
    plain, gained, X, y = _pair(kind)
    flat = torch.rand(plain.d_total, dtype=DT) * 2 - 1
    g = _with_gain(plain, gained, flat, [0.0] * plain.num_layers)
    torch.testing.assert_close(gained.forward(X, g), plain.forward(X, flat))
    for form in ("mean", "sample"):
        plain.l2_form = gained.l2_form = form
        torch.testing.assert_close(gained.l2(g, 0.7, 7), plain.l2(flat, 0.7, 7))


@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_gain_equals_scaled_weights(kind):
    """s en la capa k da la misma red que multiplicar sus pesos por s."""
    torch.manual_seed(1)
    plain, gained, X, y = _pair(kind)
    flat = torch.rand(plain.d_total, dtype=DT) * 2 - 1
    u = [math.log(0.3), math.log(1.7)][: plain.num_layers]
    g = _with_gain(plain, gained, flat, u)
    scaled = flat.clone()
    wm = plain.weight_mask()
    for k, s in enumerate(plain.layer_specs):
        seg = slice(s.offset, s.offset + s.size)
        scaled[seg] = torch.where(wm[seg], flat[seg] * math.exp(u[k]), flat[seg])
    torch.testing.assert_close(gained.forward(X, g), plain.forward(X, scaled))
    for form in ("mean", "sample"):
        plain.l2_form = gained.l2_form = form
        torch.testing.assert_close(gained.l2(g, 0.7, 7), plain.l2(scaled, 0.7, 7))


def test_l2_sample_is_s2_sum_w2():
    m = MLP([2, 3], dtype=DT, layer_gain=True)            # 6 w + 3 b + u
    m.l2_form = "sample"
    flat = torch.zeros(m.d_total, dtype=DT)
    flat[:6] = 0.5
    flat[6:9] = 9.0                                         # sesgos: no entran
    flat[9] = math.log(0.2)
    expected = 0.7 / 2 * (0.2 ** 2) * 6 * 0.25 / 11
    assert float(m.l2(flat, 0.7, 11)) == pytest.approx(expected)


def test_bounds_and_init():
    m = MLP([3, 4, 2], dtype=DT, layer_gain=True)
    b = dim_bounds(m, CFG)
    assert len(b) == m.d_total
    lo, hi = log_gain_range((0.1, 2.0))
    for s in m.layer_specs:
        assert b[s.offset + s.size - 1] == (lo, hi)
        assert b[s.offset] == (-1.0, 1.0)
    init = init_params_from_swarms(m, CFG, 5, torch.device("cpu"), DT)
    plain = init_params_from_swarms(MLP([3, 4, 2], dtype=DT), CFG, 5, torch.device("cpu"), DT)
    # Misma red de partida que sin ganancia, con u = 0.
    assert bool((init[m.gain_mask()] == 0).all())
    torch.testing.assert_close(init[~m.gain_mask()], plain)


def test_param_stats_with_gain():
    m = MLP([2, 3], dtype=DT, layer_gain=True)
    flat = torch.zeros(m.d_total, dtype=DT)
    flat[:6] = 1.0                                          # todos en el borde
    flat[9] = math.log(0.5)
    st = param_stats(m, flat, CFG)["layers"][0]
    assert st["gain"] == pytest.approx(0.5)
    assert st["w_sq_mean"] == pytest.approx(1.0)
    assert st["w_sq_mean_eff"] == pytest.approx(0.25)
    assert st["w_edge_frac"] == 1.0 and st["b_edge_frac"] == 0.0
    assert st["n_weights"] == 6 and st["n_biases"] == 3


@pytest.mark.parametrize("bad", [(0.0, 2.0), (1.5, 2.0), (0.1, 0.9), (1.0, 1.0)])
def test_invalid_gain_range(bad):
    with pytest.raises(ValueError):
        validate_gain_range(bad)

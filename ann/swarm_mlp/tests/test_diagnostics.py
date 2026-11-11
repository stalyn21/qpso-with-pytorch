"""core/diagnostics.py: escala y pesos en el borde por capa."""
import math

import pytest
import torch

from core.cnn import TinyCNN
from core.diagnostics import format_param_stats, param_stats
from core.mlp import MLP

CFG = {"bounds_mode": "fixed", "bounds": (-1.0, 1.0), "edge_tol": 1e-3}


def test_known_vector_mlp():
    m = MLP([2, 3, 2], dtype=torch.float64)          # capa 0: 6 w + 3 b; capa 1: 6 w + 2 b
    flat = torch.zeros(m.d_total, dtype=torch.float64)
    flat[0:6] = torch.tensor([1.0, -1.0, 0.9995, 0.5, 0.0, 0.0])   # 3 en el borde
    flat[6:9] = torch.tensor([2.0, 0.0, 0.0])                         # sesgo fuera de la caja
    s = param_stats(m, flat, CFG)
    l0, l1 = s["layers"]
    assert (l0["n_weights"], l0["n_biases"], l1["n_weights"], l1["n_biases"]) == (6, 3, 6, 2)
    assert l0["w_sq_mean"] == pytest.approx((1 + 1 + 0.9995 ** 2 + 0.25) / 6)
    assert l0["w_edge_frac"] == pytest.approx(3 / 6)
    assert l0["b_sq_mean"] == pytest.approx(4 / 3)
    assert l0["b_edge_frac"] == pytest.approx(1 / 3)
    assert l1["w_sq_mean"] == 0.0 and l1["w_edge_frac"] == 0.0
    assert s["total"]["n_weights"] == 12
    assert s["total"]["w_edge_frac"] == pytest.approx(3 / 12)
    assert s["total"]["w_sq_mean"] == pytest.approx(l0["w_sq_mean"] / 2)


def test_uniform_reference_value():
    torch.manual_seed(0)
    m = MLP([30, 40, 5], dtype=torch.float64)
    flat = torch.rand(m.d_total, dtype=torch.float64) * 2 - 1
    s = param_stats(m, flat, CFG)
    assert s["total"]["w_sq_mean"] == pytest.approx(1 / 3, abs=0.02)
    assert s["total"]["w_edge_frac"] == pytest.approx(1e-3, abs=2e-3)


def test_fanin_bounds_per_layer():
    m = MLP([16, 4, 2], dtype=torch.float64)
    cfg = {"bounds_mode": "fanin", "bounds_scale": math.sqrt(6), "edge_tol": 1e-3}
    flat = torch.zeros(m.d_total, dtype=torch.float64)
    b0 = math.sqrt(6) / 4                            # fan_in 16
    flat[0] = b0
    s = param_stats(m, flat, cfg)
    assert s["layers"][0]["high"] == pytest.approx(b0)
    assert s["layers"][1]["high"] == pytest.approx(math.sqrt(6) / 2)
    assert s["layers"][0]["w_edge_frac"] == pytest.approx(1 / 64)


def test_cnn_weight_bias_split():
    m = TinyCNN(in_channels=1, image_size=12, channels=(2, 3), n_classes=4, dtype=torch.float64)
    s = param_stats(m, torch.zeros(m.d_total, dtype=torch.float64), CFG)
    for spec, l in zip(m.layer_specs, s["layers"]):
        assert l["n_biases"] == spec.n_out
        assert l["n_weights"] + l["n_biases"] == spec.size
    assert "media w²" in format_param_stats(s)

"""El costo vectorizado (por capa y completo) coincide con el forward, y la L2 por
capa suma la total. MLP y CNN, en float64 y en CPU."""
import pytest
import torch

from core.cnn import TinyCNN
from core.loss import cross_entropy_from_logits
from core.mlp import MLP

DT = torch.float64
LAMBDA = 0.7
P = 5


def _mlp(l2_form, gain=False):
    m = MLP([4, 6, 5, 3], dtype=DT, layer_gain=gain)
    m.l2_form = l2_form
    return m, torch.randn(11, 4, dtype=DT), torch.randint(0, 3, (11,))


def _cnn(l2_form, head, act_norm, max_elems=1.2e8, gain=False):
    m = TinyCNN(in_channels=2, image_size=12, channels=(2, 3), n_classes=4, head=head,
                act_norm=act_norm, dtype=DT, max_elems=max_elems, layer_gain=gain)
    m.l2_form = l2_form
    return m, torch.randn(9, 2, 12, 12, dtype=DT), torch.randint(0, 4, (9,))


def _models():
    out = []
    for form in ("mean", "sample"):
        out.append(("mlp", form, _mlp(form)))
        for head in ("gap", "flatten"):
            for norm in ("none", "layer"):
                out.append((f"cnn-{head}-{norm}", form, _cnn(form, head, norm)))
        # max_elems mínimo: fuerza un chunk por imagen
        out.append(("cnn-chunked", form, _cnn(form, "flatten", "none", max_elems=1)))
        # Ganancia por capa (core/gain.py)
        out.append(("mlp-gain", form, _mlp(form, gain=True)))
        out.append(("cnn-gain", form, _cnn(form, "flatten", "none", gain=True)))
        out.append(("cnn-gain-chunked", form, _cnn(form, "gap", "layer", max_elems=1, gain=True)))
    return out


MODELS = _models()
IDS = [f"{name}-{form}" for name, form, _ in MODELS]


def _ce(model, X, y, flat):
    return cross_entropy_from_logits(model.forward(X, flat), y, dim_classes=-1).mean()


@pytest.mark.parametrize("name,form,case", MODELS, ids=IDS)
def test_layer_cost_matches_forward(name, form, case):
    torch.manual_seed(0)
    model, X, y = case
    frozen = torch.rand(model.d_total, dtype=DT) * 2 - 1
    for k, spec in enumerate(model.layer_specs):
        pos = torch.rand(P, spec.size, dtype=DT) * 2 - 1
        cost = model.build_layer_cost_fn(X, y, k, frozen, lambda_l2=LAMBDA)(pos)
        for p in range(P):
            flat = frozen.clone()
            flat[spec.offset : spec.offset + spec.size] = pos[p]
            expected = _ce(model, X, y, flat) + model.l2(pos[p], LAMBDA, X.shape[0], k=k)
            torch.testing.assert_close(cost[p], expected)


@pytest.mark.parametrize("name,form,case", MODELS, ids=IDS)
def test_full_cost_matches_forward(name, form, case):
    torch.manual_seed(1)
    model, X, y = case
    pos = torch.rand(P, model.d_total, dtype=DT) * 2 - 1
    cost = model.build_full_cost_fn(X, y, lambda_l2=LAMBDA)(pos)
    for p in range(P):
        expected = _ce(model, X, y, pos[p]) + model.l2(pos[p], LAMBDA, X.shape[0])
        torch.testing.assert_close(cost[p], expected)


@pytest.mark.parametrize("name,form,case", MODELS, ids=IDS)
def test_single_vector_input(name, form, case):
    model, X, y = case
    pos = torch.rand(3, model.d_total, dtype=DT) * 2 - 1
    f = model.build_full_cost_fn(X, y, lambda_l2=LAMBDA)
    torch.testing.assert_close(f(pos[1]), f(pos)[1])


@pytest.mark.parametrize("name,form,case", MODELS, ids=IDS)
def test_l2_per_layer_sums_to_total(name, form, case):
    model, X, _ = case
    flat = torch.rand(model.d_total, dtype=DT) * 2 - 1
    per_layer = sum(model.l2(flat[s.offset : s.offset + s.size], LAMBDA, X.shape[0], k=k)
                    for k, s in enumerate(model.layer_specs))
    torch.testing.assert_close(per_layer, model.l2(flat, LAMBDA, X.shape[0]))


def test_sample_form_ignores_biases():
    model, X, _ = _mlp("sample")
    flat = torch.zeros(model.d_total, dtype=DT)
    flat[~model.weight_mask()] = 5.0          # sólo sesgos
    assert float(model.l2(flat, LAMBDA, X.shape[0])) == 0.0


def test_lambda_zero_is_zero():
    model, X, _ = _mlp("mean")
    assert float(model.l2(torch.ones(model.d_total, dtype=DT), 0.0, X.shape[0])) == 0.0

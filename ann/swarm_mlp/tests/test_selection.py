"""Regla de selección: la poda del Selector da lo mismo que aplicar la regla a todos."""
import math
import random

import pytest
import torch

from core.selection import (Selector, ValScore, _EPS, criterion_name, standard_error,
                            standard_error_max)

SUPPORTS = [5, 7, 6]


def _score(rng):
    hits = [rng.randint(0, n) for n in SUPPORTS]
    recalls = tuple(h / n for h, n in zip(hits, SUPPORTS))
    acc = sum(hits) / sum(SUPPORTS)
    ce = rng.choice([0.3, 0.5, 0.5, 0.7, rng.random()])   # repetidos a propósito
    return ValScore(acc=acc, bacc=sum(recalls) / len(recalls), ce=ce, recalls=recalls)


def _brute_force(scores, metric, n_se):
    """La regla tal como la documenta core/selection.py, sobre TODOS los candidatos."""
    m = [s.metric(metric) for s in scores]
    ref = max(range(len(scores)), key=lambda i: (m[i], -scores[i].ce, -i))
    thr = m[ref] - n_se * standard_error(metric, scores[ref], SUPPORTS)
    tied = [i for i in range(len(scores)) if m[i] >= thr - _EPS]
    return min(tied, key=lambda i: (scores[i].ce, i))


def _y_val():
    return torch.cat([torch.full((n,), c) for c, n in enumerate(SUPPORTS)])


@pytest.mark.parametrize("metric", ["acc", "bacc"])
@pytest.mark.parametrize("n_se", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("seed", range(40))
def test_pruning_matches_brute_force(metric, n_se, seed):
    rng = random.Random(seed)
    scores = [_score(rng) for _ in range(rng.randint(1, 60))]
    sel = Selector(metric, n_se, _y_val(), len(SUPPORTS))
    for i, s in enumerate(scores):
        sel.offer(s, i)
    win, _, info = sel.choose()
    assert win == _brute_force(scores, metric, n_se)
    assert info["n_candidates"] == len(scores)


def test_se_never_zero_and_bounded():
    perfect = ValScore(acc=1.0, bacc=1.0, ce=0.0, recalls=(1.0, 1.0, 1.0))
    for metric in ("acc", "bacc"):
        se = standard_error(metric, perfect, SUPPORTS)
        assert 0 < se <= standard_error_max(metric, SUPPORTS)


def test_empty_selector_raises():
    with pytest.raises(RuntimeError):
        Selector("acc", 1.0, _y_val(), 3).choose()


def test_criterion_names():
    assert criterion_name("acc", 0) == "val_acc>val_ce"
    assert criterion_name("bacc", 1.0) == "val_bacc(1SE)>val_ce"

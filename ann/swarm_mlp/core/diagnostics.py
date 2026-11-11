"""Diagnóstico de los parámetros de un modelo: escala y pesos contra la caja, por capa.

No cambia el entrenamiento. `run_sessions` lo calcula sobre los parámetros de cada
sesión y sobre el ganador, y lo guarda en run.json (`param_stats`).

Por capa k (y para el modelo completo en `total`):
- `w_sq_mean`, `b_sq_mean`: media de w² de los pesos y de b² de los sesgos. Una
  uniforme en [−a, a] da a²/3 (0.333 en [−1, 1]): si la media no se aleja de ese
  valor, la búsqueda no movió la escala de la capa.
- `w_edge_frac`, `b_edge_frac`: fracción de pesos y de sesgos pegados a un borde de
  la caja de la capa (`core.bounds.layer_range`), a menos de `edge_tol`·(ancho / 2).
  Con la caja [−1, 1] y edge_tol = 1e-3 es |w| ≥ 0.999.
- `low`, `high`: la caja de la capa.

Pesos y sesgos se separan con `model.weight_mask(k)`, así que sirve igual para el
MLP y para la CNN.
"""
from typing import Any, Dict

import torch

from core.bounds import layer_range


def param_stats(model, flat: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Escala y fracción en el borde de `flat`, por capa y total."""
    edge_tol = float(cfg["edge_tol"])
    layers = []
    tot = {"n_weights": 0, "n_biases": 0, "w_sq_sum": 0.0, "b_sq_sum": 0.0,
           "w_edge": 0, "b_edge": 0}
    with torch.no_grad():
        for k, spec in enumerate(model.layer_specs):
            seg = flat[spec.offset : spec.offset + spec.size].double()
            mask = model.weight_mask(k).to(seg.device)
            low, high = layer_range(model, cfg, k)
            margin = edge_tol * (high - low) / 2.0
            at_edge = (seg <= low + margin) | (seg >= high - margin)
            w, b = seg[mask], seg[~mask]
            n_w, n_b = int(w.numel()), int(b.numel())
            w_sq, b_sq = float((w ** 2).sum()), float((b ** 2).sum())
            w_edge, b_edge = int(at_edge[mask].sum()), int(at_edge[~mask].sum())
            layers.append({
                "layer": k,
                "n_weights": n_w,
                "n_biases": n_b,
                "w_sq_mean": w_sq / n_w if n_w else 0.0,
                "b_sq_mean": b_sq / n_b if n_b else 0.0,
                "w_edge_frac": w_edge / n_w if n_w else 0.0,
                "b_edge_frac": b_edge / n_b if n_b else 0.0,
                "low": float(low),
                "high": float(high),
            })
            tot["n_weights"] += n_w
            tot["n_biases"] += n_b
            tot["w_sq_sum"] += w_sq
            tot["b_sq_sum"] += b_sq
            tot["w_edge"] += w_edge
            tot["b_edge"] += b_edge
    n_w, n_b = tot["n_weights"], tot["n_biases"]
    total = {
        "n_weights": n_w,
        "n_biases": n_b,
        "w_sq_mean": tot["w_sq_sum"] / n_w if n_w else 0.0,
        "b_sq_mean": tot["b_sq_sum"] / n_b if n_b else 0.0,
        "w_edge_frac": tot["w_edge"] / n_w if n_w else 0.0,
        "b_edge_frac": tot["b_edge"] / n_b if n_b else 0.0,
    }
    return {"edge_tol": edge_tol, "layers": layers, "total": total}


def format_param_stats(stats: Dict[str, Any]) -> str:
    """Una línea para la consola: media de w² y fracción de pesos en el borde por capa."""
    w_sq = "/".join(f"{l['w_sq_mean']:.3f}" for l in stats["layers"])
    edge = "/".join(f"{l['w_edge_frac']:.3f}" for l in stats["layers"])
    return f"pesos por capa: media w²={w_sq}  en el borde={edge}"

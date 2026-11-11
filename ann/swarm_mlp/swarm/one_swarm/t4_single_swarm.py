"""T4 — Un solo enjambre para toda la ANN (baseline clásico).

Sin descomposición por capa, sin paralelización, sin concurrencia. Es el approach
"clásico" contra el que se comparan T1/T2/T3. Un único QDPSO con dimensionalidad
`d_total` (suma de pesos+biases de todas las capas) optimiza el modelo completo.

Cadena inter-sesión:
    No hay contrapartes que intercambiar; en su lugar usamos warm-start: las primeras
    K partículas se inicializan como `params_init + ruido pequeño` formando un cluster
    denso cerca del óptimo conocido. Las P-K restantes son random para preservar
    diversidad/exploración. K=1 con noise=0 reproduce el warm-start original (sólo
    seed exacto).
"""
import time
from typing import Any, Dict

import torch

from core.bounds import dim_bounds
from core.selection import Selector, val_score
from core.mlp import MLP
from qpso_torch import CallbackEvent
from swarm.optimizer_factory import build_swarm_optimizer
from swarm.warm_start import warm_start_cluster


def run_session_t4(
    mlp: MLP,
    data: Dict[str, Any],
    params_init: torch.Tensor,
    cfg: Dict[str, Any],
    session_seed: int,
) -> Dict[str, Any]:
    bounds = dim_bounds(mlp, cfg)

    cost_fn = mlp.build_full_cost_fn(
        X=data["X_train"], y=data["y_train"],
        lambda_l2=cfg["lambda_l2"],
    )

    # Candidatos: cada mejora del gbest (ON_NEW_BEST dispara cuando gbest_value =
    # train + L2 mejora). Se elige con la regla de core.selection al terminar.
    X_val, y_val = data["X_val"], data["y_val"]
    selector = Selector.from_cfg(cfg, data)

    def on_new_best(opt):
        score = val_score(mlp, opt.gbest, X_val, y_val, data["n_classes"])
        selector.offer(score, opt.gbest.clone(), {"iter_at_best": opt.iters})

    # Configuración del warm-start cluster (CLI configurable, defaults razonables).
    k_seeded = int(cfg.get("warm_particles", 10))
    noise_scale = float(cfg.get("warm_noise", 0.05))
    # Asegurar K <= n_particles
    k_seeded = max(1, min(k_seeded, cfg["n_particles"]))

    session_start = time.time()
    with build_swarm_optimizer(
        cfg,
        cf=cost_fn,
        dim=mlp.d_total,
        bounds=bounds,
        seed=session_seed,
        dtype=mlp.dtype,
    ) as opt:
        # Warm-start cluster: K partículas alrededor de params_init.
        warm_start_cluster(opt, params_init, k_seeded=k_seeded, noise_scale=noise_scale)

        opt.callbacks.register(CallbackEvent.ON_NEW_BEST, on_new_best)
        result = opt.optimize()
    session_time = time.time() - session_start

    # Sin ninguna mejora del gbest (no ocurre con el QDPSO actual) se devuelve el
    # punto de partida, medido en val.
    if selector.n_candidates == 0:
        selector.offer(val_score(mlp, params_init, X_val, y_val, data["n_classes"]),
                       params_init.clone(), {"iter_at_best": 0})
    params_best, score_best, sel_info = selector.choose()
    best_val_acc, best_val_ce = score_best.acc, score_best.ce

    print(f"    🎯 single-swarm dim={mlp.d_total}: "
          f"train_loss={float(result.best_value):.4E}  "
          f"iters={result.iterations}  "
          f"best_val: acc={best_val_acc:.4f} bacc={score_best.bacc:.4f} ce={best_val_ce:.4f} "
          f"en iter {sel_info['iter_at_best']} "
          f"(franja: no dominados {sel_info['n_tied_nondominated']}/{sel_info['n_candidates']})  "
          f"warm-cluster=K{k_seeded}/σ{noise_scale}  "
          f"⏱ {session_time:.2f}s  "
          f"({result.convergence_reason})")

    per_layer = [{
        "layer": "all",
        "position": "single_swarm",
        "dim": mlp.d_total,
        "final_train_loss": float(result.best_value),
        "iterations_run": int(result.iterations),
        "converged": bool(result.converged),
        "convergence_reason": result.convergence_reason,
        "val_best_acc": float(best_val_acc),
        "val_best_bacc": float(score_best.bacc),
        "val_best_ce": float(best_val_ce),
        "iter_at_val_best": int(sel_info["iter_at_best"]),
        "layer_time": float(session_time),
        "warm_k_seeded": k_seeded,
        "warm_noise_scale": noise_scale,
    }]

    return {
        "params_best": params_best,
        "per_layer": per_layer,
        "session_time": session_time,
        "best_val_acc_during_session": float(best_val_acc),
        "best_val_ce_during_session": float(best_val_ce),
        "best_val_bacc_during_session": float(score_best.bacc),
        "iter_at_best": int(sel_info["iter_at_best"]),
        "selection": sel_info,
    }

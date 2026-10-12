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

from core.metrics import selection_key
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
    low, high = cfg["bounds"]
    bounds = [(low, high)] * mlp.d_total

    cost_fn = mlp.build_full_cost_fn(
        X=data["X_train"], y=data["y_train"],
        lambda_l2=cfg["lambda_l2"],
    )

    # Tracking de mejor val durante la sesión (criterio: core.metrics.selection_key,
    # más val_acc y, con la misma, menos val_ce). ON_NEW_BEST sólo dispara cuando
    # opt.gbest_value (= train+L2) mejora.
    val_best = {
        "params": params_init.clone(),
        "key": (float("inf"), float("inf")),
        "iter_at_best": 0,
    }
    X_val, y_val = data["X_val"], data["y_val"]

    def on_new_best(opt):
        key = selection_key(mlp, opt.gbest, X_val, y_val)
        if key < val_best["key"]:
            val_best["key"] = key
            val_best["params"] = opt.gbest.clone()
            val_best["iter_at_best"] = opt.iters

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

    # Si el QDPSO terminó sin mejorar val (raro pero posible), val_best.params
    # podría estar en su valor inicial. Aseguramos que devolvemos el mejor visto.
    params_best = val_best["params"]
    best_val_acc, best_val_ce = 1.0 - val_best["key"][0], val_best["key"][1]

    print(f"    🎯 single-swarm dim={mlp.d_total}: "
          f"train_loss={float(result.best_value):.4E}  "
          f"iters={result.iterations}  "
          f"best_val: acc={best_val_acc:.4f} ce={best_val_ce:.4f} en iter {val_best['iter_at_best']}  "
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
        "val_best_ce": float(best_val_ce),
        "iter_at_val_best": int(val_best["iter_at_best"]),
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
        "iter_at_best": int(val_best["iter_at_best"]),
    }

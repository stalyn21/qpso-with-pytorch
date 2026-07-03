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

from core.mlp import MLP
from tensor_qpso import QDPSOTensorOptimized, CallbackEvent
from techniques.warm_start import warm_start_cluster


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

    # Tracking de mejor val durante la sesión.
    # NOTA: ON_NEW_BEST sólo dispara cuando opt.gbest_value (= train+L2) mejora,
    # así que train_loss decrece monótonamente entre disparos. Usar `<=` (no `<`)
    # garantiza que ante empate de val nos quedamos con el ÚLTIMO snapshot, que
    # tiene el menor train_loss → modelo más convergido y más generalizable.
    val_best = {
        "params": params_init.clone(),
        "val_cost": float("inf"),
        "iter_at_best": 0,
    }
    X_val = data["X_val"]
    y_val_f = data["y_val"].float()

    def on_new_best(opt):
        with torch.no_grad():
            y_pred = mlp.predict(X_val, opt.gbest)
        val_cost = ((y_val_f - y_pred.float()) ** 2).mean().item()
        # `<=` permite que el último snapshot con val mínimo gane (más maduro
        # porque opt.gbest_value es monótono decreciente entre disparos).
        if val_cost <= val_best["val_cost"]:
            val_best["val_cost"] = val_cost
            val_best["params"] = opt.gbest.clone()
            val_best["iter_at_best"] = opt.iters

    # Configuración del warm-start cluster (CLI configurable, defaults razonables).
    k_seeded = int(cfg.get("warm_particles", 10))
    noise_scale = float(cfg.get("warm_noise", 0.05))
    # Asegurar K <= n_particles
    k_seeded = max(1, min(k_seeded, cfg["n_particles"]))

    session_start = time.time()
    with QDPSOTensorOptimized(
        cf=cost_fn,
        size=cfg["n_particles"],
        dim=mlp.d_total,
        bounds=bounds,
        maxIters=cfg["max_iter"],
        g=cfg["g"],
        device=cfg["device"],
        dtype=mlp.dtype,
        seed=session_seed,
        boundary_strategy=cfg["boundary_strategy"],
        tol=cfg["tol"],
        patience=cfg["patience"],
        track_history=False,
        minimize=True,
    ) as opt:
        # Warm-start cluster: K partículas alrededor de params_init.
        warm_start_cluster(opt, params_init, k_seeded=k_seeded, noise_scale=noise_scale)

        opt.callbacks.register(CallbackEvent.ON_NEW_BEST, on_new_best)
        result = opt.optimize()
    session_time = time.time() - session_start

    # Si el QDPSO terminó sin mejorar val (raro pero posible), val_best.params
    # podría estar en su valor inicial. Aseguramos que devolvemos el mejor visto.
    params_best = val_best["params"]

    print(f"    🎯 single-swarm dim={mlp.d_total}: "
          f"train_loss={float(result.best_value):.4E}  "
          f"iters={result.iterations}  "
          f"best_val={val_best['val_cost']:.4f} en iter {val_best['iter_at_best']}  "
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
        "val_best_cost": float(val_best["val_cost"]),
        "iter_at_val_best": int(val_best["iter_at_best"]),
        "layer_time": float(session_time),
        "warm_k_seeded": k_seeded,
        "warm_noise_scale": noise_scale,
    }]

    return {
        "params_best": params_best,
        "per_layer": per_layer,
        "session_time": session_time,
        "best_val_cost_during_session": float(val_best["val_cost"]),
        "iter_at_best": int(val_best["iter_at_best"]),
    }

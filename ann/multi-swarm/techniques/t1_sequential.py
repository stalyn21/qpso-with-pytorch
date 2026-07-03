"""T1 — Secuencial sin paralelización ni concurrencia (Block Coordinate Descent).

Orden de entrenamiento dentro de cada round: de ATRÁS hacia ADELANTE
(capa de salida → capa de entrada).

Mejora sobre la versión inicial: las contrapartes ya NO permanecen congeladas en
`params_init` durante toda la sesión. Después de entrenar la capa k, su mejor
segmento (por val) se PROPAGA al vector `current` y la siguiente capa lo ve como
parte de su contexto. Esto es block coordinate descent (Gauss-Seidel a nivel
de capa, sin lockstep — cada capa converge completamente antes de pasar a la
siguiente).

Cuando `n_rounds > 1`, la sesión hace múltiples sweeps back-to-front sobre las
capas, cada uno usando como contexto el `current` actualizado por el sweep
previo. Esto es el caso general del BCD.

Diferencias con T3:
- T3 hace lockstep iter-por-iter (1 paso de cada swarm por outer iter).
- T1 hace QDPSO completo por capa antes de pasar a la siguiente (round-level GS).
"""
import time
from typing import Any, Dict

import torch

from core.mlp import MLP
from tensor_qpso import QDPSOTensorOptimized, CallbackEvent
from techniques.warm_start import warm_start_cluster


def run_session_t1(
    mlp: MLP,
    data: Dict[str, Any],
    params_init: torch.Tensor,
    cfg: Dict[str, Any],
    session_seed: int,
) -> Dict[str, Any]:
    low, high = cfg["bounds"]
    n_rounds = int(cfg.get("n_rounds", 1))
    per_layer = []

    # Warm-start cluster: K partículas concentradas alrededor del segmento actual,
    # P-K random para preservar exploración. Reduce la varianza por cold start
    # de cada (round, capa). Default K=10, σ=0.05 (configurable via CLI).
    k_seeded = int(cfg.get("warm_particles", 10))
    noise_scale = float(cfg.get("warm_noise", 0.05))
    k_seeded_eff = max(1, min(k_seeded, cfg["n_particles"]))

    # `current` evoluciona durante la sesión.
    current = params_init.clone()

    X_val = data["X_val"]
    y_val = data["y_val"]
    y_val_f = y_val.float()

    training_order = list(range(mlp.num_layers - 1, -1, -1))

    for round_idx in range(n_rounds):
        for k in training_order:
            spec = mlp.layer_specs[k]
            position_label = (
                "salida" if k == mlp.num_layers - 1 else
                ("entrada" if k == 0 else f"oculta{k}")
            )
            dim_k = spec.size
            bounds = [(low, high)] * dim_k

            # Cost fn usa el `current` actual: capas != k congeladas a su valor reciente.
            cost_fn = mlp.build_layer_cost_fn(
                X=data["X_train"], y=data["y_train"],
                k=k, frozen_flat=current,
                lambda_l2=cfg["lambda_l2"],
            )

            # Snapshot del contexto actual para el callback de val.
            # Importante: clonamos para que el callback evalúe contra el mismo contexto
            # que usa la cost fn (incluso si `current` se mutara — no lo hacemos durante
            # el QDPSO, pero la consistencia conceptual es valiosa).
            frozen_snapshot = current.clone()

            val_best = {
                "params": current[spec.offset : spec.offset + spec.size].clone(),
                "val_cost": float("inf"),
                "iter_at_best": 0,
            }

            def make_callback(val_best_ref, spec_ref, frozen_ref):
                def on_new_best(opt):
                    candidate = frozen_ref.clone()
                    candidate[spec_ref.offset : spec_ref.offset + spec_ref.size] = opt.gbest
                    with torch.no_grad():
                        y_pred = mlp.predict(X_val, candidate)
                    val_cost = ((y_val_f - y_pred.float()) ** 2).mean().item()
                    if val_cost < val_best_ref["val_cost"]:
                        val_best_ref["val_cost"] = val_cost
                        val_best_ref["params"] = opt.gbest.clone()
                        val_best_ref["iter_at_best"] = opt.iters
                return on_new_best

            layer_start = time.time()
            with QDPSOTensorOptimized(
                cf=cost_fn,
                size=cfg["n_particles"],
                dim=dim_k,
                bounds=bounds,
                maxIters=cfg["max_iter"],
                g=cfg["g"],
                device=cfg["device"],
                dtype=mlp.dtype,
                seed=session_seed * 10000 + round_idx * 100 + k,
                boundary_strategy=cfg["boundary_strategy"],
                tol=cfg["tol"],
                patience=cfg["patience"],
                track_history=False,
                minimize=True,
            ) as opt:
                # Warm-start: ancla la búsqueda alrededor del segmento actual de la capa
                # k en `current`. Reduce la varianza vs random init puro (cold start).
                seed_segment = current[spec.offset : spec.offset + spec.size]
                warm_start_cluster(opt, seed_segment, k_seeded_eff, noise_scale)

                opt.callbacks.register(
                    CallbackEvent.ON_NEW_BEST,
                    make_callback(val_best, spec, frozen_snapshot),
                )
                result = opt.optimize()
            layer_time = time.time() - layer_start

            # Propagar el mejor segmento (por val) al vector combinado.
            current[spec.offset : spec.offset + spec.size] = val_best["params"]

            per_layer.append({
                "round": round_idx,
                "layer": k,
                "position": position_label,
                "n_in": spec.n_in,
                "n_out": spec.n_out,
                "dim": dim_k,
                "final_train_loss": float(result.best_value),
                "val_best_cost": float(val_best["val_cost"]),
                "iter_at_val_best": int(val_best["iter_at_best"]),
                "iterations_run": int(result.iterations),
                "converged": bool(result.converged),
                "convergence_reason": result.convergence_reason,
                "layer_time": float(layer_time),
            })

            print(f"    [r{round_idx}] capa {k} ({position_label}) "
                  f"[{spec.n_in}→{spec.n_out}, dim={dim_k}]: "
                  f"train_loss={float(result.best_value):.4E}  "
                  f"val_best={val_best['val_cost']:.4f}  "
                  f"iters={result.iterations}  {layer_time:.2f}s")

    return {
        "params_best": current,
        "per_layer": per_layer,
        "n_rounds": n_rounds,
    }

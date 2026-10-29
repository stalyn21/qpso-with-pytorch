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

from core.selection import Selector, val_score
from core.mlp import MLP
from qpso_torch import CallbackEvent
from swarm.optimizer_factory import build_swarm_optimizer
from swarm.warm_start import warm_start_cluster


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

            # Candidatos de esta capa: la red completa con el gbest de la capa k sobre
            # el contexto congelado. Se elige con la regla de core.selection al terminar.
            selector = Selector.from_cfg(cfg, data)

            def make_callback(selector_ref, spec_ref, frozen_ref):
                def on_new_best(opt):
                    candidate = frozen_ref.clone()
                    candidate[spec_ref.offset : spec_ref.offset + spec_ref.size] = opt.gbest
                    score = val_score(mlp, candidate, X_val, y_val, data["n_classes"])
                    selector_ref.offer(score, opt.gbest.clone(), {"iter_at_best": opt.iters})
                return on_new_best

            layer_start = time.time()
            with build_swarm_optimizer(
                cfg,
                cf=cost_fn,
                dim=dim_k,
                bounds=bounds,
                seed=session_seed * 10000 + round_idx * 100 + k,
                dtype=mlp.dtype,
            ) as opt:
                # Warm-start: ancla la búsqueda alrededor del segmento actual de la capa
                # k en `current`. Reduce la varianza vs random init puro (cold start).
                seed_segment = current[spec.offset : spec.offset + spec.size]
                warm_start_cluster(opt, seed_segment, k_seeded_eff, noise_scale)

                opt.callbacks.register(
                    CallbackEvent.ON_NEW_BEST,
                    make_callback(selector, spec, frozen_snapshot),
                )
                result = opt.optimize()
            layer_time = time.time() - layer_start

            # Propagar el segmento elegido (por val) al vector combinado.
            if selector.n_candidates == 0:
                # Sin mejoras del gbest (no ocurre con el QDPSO actual): se mide el
                # contexto tal cual y la capa conserva su segmento.
                selector.offer(val_score(mlp, frozen_snapshot, X_val, y_val, data["n_classes"]),
                               current[spec.offset : spec.offset + spec.size].clone(),
                               {"iter_at_best": 0})
            seg_best, score_best, sel_info = selector.choose()
            current[spec.offset : spec.offset + spec.size] = seg_best

            per_layer.append({
                "round": round_idx,
                "layer": k,
                "position": position_label,
                "n_in": spec.n_in,
                "n_out": spec.n_out,
                "dim": dim_k,
                "final_train_loss": float(result.best_value),
                "val_best_acc": float(score_best.acc),
                "val_best_bacc": float(score_best.bacc),
                "val_best_ce": float(score_best.ce),
                "iter_at_val_best": int(sel_info["iter_at_best"]),
                "selection": sel_info,
                "iterations_run": int(result.iterations),
                "converged": bool(result.converged),
                "convergence_reason": result.convergence_reason,
                "layer_time": float(layer_time),
            })

            print(f"    [r{round_idx}] capa {k} ({position_label}) "
                  f"[{spec.n_in}→{spec.n_out}, dim={dim_k}]: "
                  f"train_loss={float(result.best_value):.4E}  "
                  f"val_best: acc={score_best.acc:.4f} bacc={score_best.bacc:.4f} ce={score_best.ce:.4f} "
                  f"(franja: no dominados {sel_info['n_tied_nondominated']}/{sel_info['n_candidates']})  "
                  f"iters={result.iterations}  {layer_time:.2f}s")

    return {
        "params_best": current,
        "per_layer": per_layer,
        "n_rounds": n_rounds,
    }

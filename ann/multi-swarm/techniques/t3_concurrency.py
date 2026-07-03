"""T3 — Concurrencia: intercambio de parámetros DURANTE cada iteración.

Diferencia clave con T1/T2:
    - T1/T2: contrapartes de cada capa CONGELADAS a `params_init` durante toda la sesión.
              El intercambio ocurre solo después de la sesión.
    - T3:    contrapartes ACTUALIZADAS en cada iteración del QDPSO con los gBests
              actuales de los otros swarms. No requiere intercambio post-sesión porque
              ya se hizo durante el entrenamiento.

Implementación lockstep (Gauss-Seidel back-to-front):
    1. Crear N swarms QDPSO, uno por capa.
    2. Por cada iteración i en 1..max_iter:
        a. Tomar snapshot `current` = combinación de gBests al inicio de la iteración.
        b. Iterar swarms de ATRÁS hacia ADELANTE (salida → ocultas → entrada). Para cada k:
           - Construir cost_fn con `current` como contrapartes (la capa k varía).
           - Avanzar el swarm un paso (kernel_update + boundary + update_best).
           - Actualizar `current[layer k]` con el nuevo gBest → el siguiente swarm en
             ESTA iteración ya ve esta actualización (Gauss-Seidel).
        c. Evaluar el modelo combinado en val. Guardar snapshot si mejoró.
    3. Devolver el mejor snapshot por val.

Por qué Gauss-Seidel back-to-front: el swarm de salida se entrena primero contra el
contexto más antiguo (entrada del paso previo); cuando le toca a entrada, ya tiene el
gBest fresco de salida como guía. Eso convierte el flujo en información atrás→adelante
dentro de cada iteración del QDPSO.

El bucle se ejecuta en serie (Python lockstep) porque cada paso depende del paso previo
del propio QDPSO. La "concurrencia" es lógica: los swarms cooperan compartiendo estado.
"""
import time
from typing import Any, Dict

import torch

from core.mlp import MLP
from core.metrics import mse_int
from core.loss import cross_entropy_from_logits, l2_penalty
from tensor_qpso import QDPSOTensorOptimized
from techniques.warm_start import warm_start_cluster


def _train_loss_combined(mlp, current, X_train, y_train, lambda_l2):
    """CE + L2 sobre train con el modelo combinado (tie-breaker maduro).

    L2 normalizado por (d_total · n_train) — ver core/loss.py:l2_penalty.
    """
    with torch.no_grad():
        logits = mlp.forward(X_train, current)
        ce = cross_entropy_from_logits(logits, y_train, dim_classes=-1).mean().item()
        l2 = float(l2_penalty(
            current, lambda_l2,
            d_total=mlp.d_total, n_train=X_train.shape[0],
        ).item())
    return ce + l2


def _combine_gbests(
    base_params: torch.Tensor,
    swarms,
    mlp: MLP,
) -> torch.Tensor:
    """Construye un vector plano con los gBests actuales de cada swarm en su capa."""
    out = base_params.clone()
    for k, opt in enumerate(swarms):
        spec = mlp.layer_specs[k]
        out[spec.offset : spec.offset + spec.size] = opt.gbest
    return out


def run_session_t3(
    mlp: MLP,
    data: Dict[str, Any],
    params_init: torch.Tensor,
    cfg: Dict[str, Any],
    session_seed: int,
) -> Dict[str, Any]:
    low, high = cfg["bounds"]
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    y_val_f = y_val.float()

    # Configuración del warm-start cluster para cada swarm (CLI configurable).
    # Cada swarm de capa k recibe como semilla el segmento de params_init correspondiente
    # a esa capa: K partículas refinan ese punto, P-K random exploran. Esto traslada el
    # beneficio de warm-start de T4 a T3 (estabiliza la varianza entre sesiones porque
    # la cadena monótona ahora SÍ aprovecha params_init dentro de cada swarm).
    k_seeded = int(cfg.get("warm_particles", 10))
    noise_scale = float(cfg.get("warm_noise", 0.05))
    k_seeded_eff = max(1, min(k_seeded, cfg["n_particles"]))

    # 1) Crear N swarms con el cost_fn inicial (counterparts = params_init).
    swarms = []
    for k in range(mlp.num_layers):
        spec = mlp.layer_specs[k]
        init_cost_fn = mlp.build_layer_cost_fn(
            X=X_train, y=y_train, k=k,
            frozen_flat=params_init, lambda_l2=cfg["lambda_l2"],
        )
        opt = QDPSOTensorOptimized(
            cf=init_cost_fn,
            size=cfg["n_particles"],
            dim=spec.size,
            bounds=[(low, high)] * spec.size,
            maxIters=cfg["max_iter"],   # informativo; el bucle lo manejamos nosotros
            g=cfg["g"],
            device=cfg["device"],
            dtype=mlp.dtype,
            seed=session_seed * 1000 + k,
            boundary_strategy=cfg["boundary_strategy"],
            tol=cfg["tol"],
            patience=cfg["patience"],
            track_history=False,
            minimize=True,
        )
        # Warm-start cluster: cada swarm refina alrededor de params_init[layer_k].
        seed_segment = params_init[spec.offset : spec.offset + spec.size]
        warm_start_cluster(opt, seed_segment, k_seeded_eff, noise_scale)
        swarms.append(opt)

    # Snapshot inicial: evaluamos val con el combinado de gBests iniciales (post warm-start).
    best_combined = _combine_gbests(params_init, swarms, mlp)
    with torch.no_grad():
        y_pred = mlp.predict(X_val, best_combined)
    best_val_cost = mse_int(y_val, y_pred)
    best_train_loss = _train_loss_combined(
        mlp, best_combined, X_train, y_train, cfg["lambda_l2"]
    )
    iter_at_best = 0

    # Métricas por capa (gBest final de cada swarm + tiempo)
    layer_iter_times = [0.0] * mlp.num_layers
    layer_train_loss = [None] * mlp.num_layers

    session_start = time.time()

    # 2) Bucle lockstep — Gauss-Seidel back-to-front
    update_order = list(range(mlp.num_layers - 1, -1, -1))  # salida → entrada

    with torch.no_grad():
        for it in range(cfg["max_iter"]):
            # a) Snapshot inicial de la iteración (gBests al cierre de la iteración previa)
            current = _combine_gbests(params_init, swarms, mlp)

            # b) Para cada swarm en orden back-to-front:
            for k in update_order:
                opt = swarms[k]
                spec = mlp.layer_specs[k]
                t0 = time.time()

                fresh_cost_fn = mlp.build_layer_cost_fn(
                    X=X_train, y=y_train, k=k,
                    frozen_flat=current, lambda_l2=cfg["lambda_l2"],
                )
                opt._cf = fresh_cost_fn
                opt._vectorized_cf = None

                opt.kernel_update()
                opt._positions = opt._apply_boundary(opt._positions)
                opt.update_best()
                opt._iters += 1

                # Gauss-Seidel: el siguiente swarm en ESTA iteración ya ve el gBest fresco.
                current[spec.offset : spec.offset + spec.size] = opt.gbest

                layer_iter_times[k] += time.time() - t0
                layer_train_loss[k] = float(opt.gbest_value)

            # c) Evaluar el modelo combinado completo en val
            y_pred = mlp.predict(X_val, current)
            val_cost = mse_int(y_val, y_pred)

            # Snapshot selection con tie-breaker train_loss:
            # 1. menor val_cost gana
            # 2. con mismo val_cost, menor train_loss gana (modelo más convergido)
            # Evita el "first-hit cherry-pick" que castigaba a T3 en datasets pequeños.
            if val_cost <= best_val_cost:
                train_loss_now = _train_loss_combined(
                    mlp, current, X_train, y_train, cfg["lambda_l2"]
                )
                is_better = (val_cost < best_val_cost) or (
                    val_cost == best_val_cost and train_loss_now < best_train_loss
                )
                if is_better:
                    best_val_cost = val_cost
                    best_train_loss = train_loss_now
                    best_combined = current.clone()
                    iter_at_best = it + 1

    session_time = time.time() - session_start

    # Cleanup de swarms (libera memory pool, GPU cache)
    for opt in swarms:
        opt.__exit__(None, None, None)

    # Reporte por capa (orden de impresión: back-to-front; almacenado en orden natural)
    per_layer = []
    for k in range(mlp.num_layers):
        spec = mlp.layer_specs[k]
        label = ("salida" if k == mlp.num_layers - 1 else
                 ("entrada" if k == 0 else f"oculta{k}"))
        per_layer.append({
            "layer": k,
            "position": label,
            "n_in": spec.n_in,
            "n_out": spec.n_out,
            "dim": spec.size,
            "final_train_loss": layer_train_loss[k],
            "iterations_run": cfg["max_iter"],
            "layer_time": float(layer_iter_times[k]),
        })

    for k in update_order:
        spec = mlp.layer_specs[k]
        label = per_layer[k]["position"]
        print(f"    capa {k} ({label}) [{spec.n_in}→{spec.n_out}, dim={spec.size}]: "
              f"train_loss={layer_train_loss[k]:.4E}  "
              f"iters={cfg['max_iter']}  acumulado={layer_iter_times[k]:.2f}s")

    print(f"    🔁 lockstep: best_val={best_val_cost:.4f} en iter {iter_at_best}/{cfg['max_iter']}  "
          f"warm-cluster=K{k_seeded_eff}/σ{noise_scale}  "
          f"⏱ sesión={session_time:.2f}s")

    return {
        "params_best": best_combined,
        "per_layer": per_layer,
        "session_time": session_time,
        "best_val_cost_during_session": float(best_val_cost),
        "iter_at_best": int(iter_at_best),
    }

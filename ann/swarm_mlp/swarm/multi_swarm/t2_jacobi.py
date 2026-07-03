"""T2-jacobi — Variante Jacobi con sub-rondas y WARM-START de QDPSO.

Variante de T2 que NO comparte estado entre capas durante el round (Jacobi
puro). La canónica T2 es la versión asíncrona en `t2_async.py`; esta vive
como baseline para comparar el costo de oscilación de Jacobi vs la propagación
inmediata del async.

Cada capa tiene su propio QDPSO que persiste durante toda la sesión. En cada
round, las N capas reciben una NUEVA cost function (basada en el `current`
sincronizado del round anterior) pero MANTIENEN su estado espacial: positions
y la geometría de pbest. Esto permite que el QDPSO acumule progreso entre
rounds en lugar de hacer búsqueda random nueva en cada uno.

Sin warm-start, T2-jacobi puro es estructuralmente incapaz de converger
porque cada round es una lotería independiente. Con warm-start, las partículas
de cada capa se "adaptan" a la nueva cost function manteniendo la información
de la búsqueda previa.

Política de sincronización (Jacobi puro): dentro de un round, las N capas
entrenan en paralelo contra el MISMO `current` (snapshot). Al terminar el
round, se sincronizan todos los mejores segmentos y se actualiza `current`.
Diferencia con T1 (Gauss-Seidel: capas se ven entre sí dentro del round),
con T2 async (lock-free, sin barrera por round) y con T3 (lockstep iter-por-iter).

- En GPU: cada thread corre sobre su propio `torch.cuda.Stream` para que los
  kernels de capas distintas se solapen en el scheduler de CUDA.
- En CPU: el GIL serializa el trabajo Python puro; ganancia marginal en redes
  pequeñas.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch

from core.mlp import MLP
from tensor_qpso import QDPSOTensorOptimized, CallbackEvent
from swarm.warm_start import warm_start_cluster


def _position_label(k: int, num_layers: int) -> str:
    if k == num_layers - 1:
        return "salida"
    if k == 0:
        return "entrada"
    return f"oculta{k}"


def _train_layer_round(
    *,
    mlp: MLP,
    data: Dict[str, Any],
    frozen_flat: torch.Tensor,
    cfg: Dict[str, Any],
    k: int,
    opt: QDPSOTensorOptimized,
    is_first_round: bool,
    cuda_stream: Optional["torch.cuda.Stream"] = None,
) -> Dict[str, Any]:
    """Worker que entrena la capa k en un round, usando un opt persistente."""
    spec = mlp.layer_specs[k]
    X_val = data["X_val"]
    y_val_f = data["y_val"].float()

    cost_fn = mlp.build_layer_cost_fn(
        X=data["X_train"], y=data["y_train"],
        k=k, frozen_flat=frozen_flat,
        lambda_l2=cfg["lambda_l2"],
    )

    val_best = {
        "params": frozen_flat[spec.offset : spec.offset + spec.size].clone(),
        "val_cost": float("inf"),
        "iter_at_best": 0,
    }

    def on_new_best(_opt):
        candidate = frozen_flat.clone()
        candidate[spec.offset : spec.offset + spec.size] = _opt.gbest
        with torch.no_grad():
            y_pred = mlp.predict(X_val, candidate)
        val_cost = ((y_val_f - y_pred.float()) ** 2).mean().item()
        if val_cost < val_best["val_cost"]:
            val_best["val_cost"] = val_cost
            val_best["params"] = _opt.gbest.clone()
            val_best["iter_at_best"] = _opt.iters

    def _run():
        # Round 0: opt recién creado con cost_fn correcta — solo registramos callback y optimizamos.
        # Round r>0: re-armamos el opt con la nueva cost_fn (warm-start) y re-registramos callback.
        if not is_first_round:
            opt.reuse_with(cost_fn)
        opt.callbacks.clear(CallbackEvent.ON_NEW_BEST)
        opt.callbacks.register(CallbackEvent.ON_NEW_BEST, on_new_best)

        layer_start = time.time()
        result = opt.optimize()
        layer_time = time.time() - layer_start
        return result, layer_time

    if cuda_stream is not None:
        with torch.cuda.stream(cuda_stream):
            result, layer_time = _run()
    else:
        result, layer_time = _run()

    return {
        "k": k,
        "spec": spec,
        "best_segment": val_best["params"],
        "val_best_cost": val_best["val_cost"],
        "iter_at_val_best": val_best["iter_at_best"],
        "result": result,
        "layer_time": layer_time,
    }


def run_session_t2_jacobi(
    mlp: MLP,
    data: Dict[str, Any],
    params_init: torch.Tensor,
    cfg: Dict[str, Any],
    session_seed: int,
) -> Dict[str, Any]:
    device_str = cfg.get("device_resolved", cfg.get("device", "cpu"))
    use_cuda = torch.device(device_str).type == "cuda"
    n_rounds = int(cfg.get("n_rounds", 1))
    low, high = cfg["bounds"]

    current = params_init.clone()
    per_layer = []
    parallel_wall_times = []
    sequential_equiv_times = []

    submission_order = list(range(mlp.num_layers - 1, -1, -1))

    # Warm-start cluster aplicado UNA vez al inicio de la sesión (antes del round 0).
    # `reuse_with` entre rounds preserva positions, así que el ancla del cluster persiste
    # durante toda la sesión. Resuelve el problema de cold start del round 0 que con sólo
    # `reuse_with` no se compensaba.
    k_seeded = int(cfg.get("warm_particles", 10))
    noise_scale = float(cfg.get("warm_noise", 0.05))
    k_seeded_eff = max(1, min(k_seeded, cfg["n_particles"]))

    # Crear UN QDPSO persistente por capa para toda la sesión.
    # Round 0 arranca con cluster warm-started; rounds siguientes hacen reuse_with(new_cf).
    opts: List[QDPSOTensorOptimized] = []
    for k in range(mlp.num_layers):
        spec = mlp.layer_specs[k]
        bounds = [(low, high)] * spec.size
        # cost_fn inicial: contra params_init. Será reemplazada via reuse_with en rounds posteriores.
        cost_fn_init = mlp.build_layer_cost_fn(
            X=data["X_train"], y=data["y_train"],
            k=k, frozen_flat=params_init,
            lambda_l2=cfg["lambda_l2"],
        )
        opt = QDPSOTensorOptimized(
            cf=cost_fn_init,
            size=cfg["n_particles"],
            dim=spec.size,
            bounds=bounds,
            maxIters=cfg["max_iter"],
            g=cfg["g"],
            device=cfg["device"],
            dtype=mlp.dtype,
            seed=session_seed * 10000 + k,
            boundary_strategy=cfg["boundary_strategy"],
            tol=cfg["tol"],
            patience=cfg["patience"],
            track_history=False,
            minimize=True,
        )
        # Ancla la búsqueda alrededor de params_init[layer_k]. El cluster persiste a través
        # de los rounds vía reuse_with (que mantiene positions y re-evalúa pbest).
        seed_segment = params_init[spec.offset : spec.offset + spec.size]
        warm_start_cluster(opt, seed_segment, k_seeded_eff, noise_scale)
        opts.append(opt)

    try:
        for round_idx in range(n_rounds):
            # Snapshot del contexto al inicio del round (todos los threads lo comparten).
            frozen_for_round = current.clone()

            streams = (
                [torch.cuda.Stream() for _ in range(mlp.num_layers)]
                if use_cuda else [None] * mlp.num_layers
            )

            round_start = time.time()
            with ThreadPoolExecutor(max_workers=mlp.num_layers) as pool:
                future_by_k: Dict[int, Any] = {}
                for k in submission_order:
                    future_by_k[k] = pool.submit(
                        _train_layer_round,
                        mlp=mlp, data=data, frozen_flat=frozen_for_round,
                        cfg=cfg, k=k, opt=opts[k],
                        is_first_round=(round_idx == 0),
                        cuda_stream=streams[k],
                    )
                results = [future_by_k[k].result() for k in submission_order]

            if use_cuda:
                torch.cuda.synchronize()
            round_wall_time = time.time() - round_start

            seq_equiv = sum(r["layer_time"] for r in results)
            speedup = seq_equiv / round_wall_time if round_wall_time > 0 else 0.0
            parallel_wall_times.append(round_wall_time)
            sequential_equiv_times.append(seq_equiv)

            # Sincronización Jacobi: aplicar todos los mejores segmentos a `current` al final del round.
            for r in results:
                spec = r["spec"]
                current[spec.offset : spec.offset + spec.size] = r["best_segment"]
                per_layer.append({
                    "round": round_idx,
                    "layer": r["k"],
                    "position": _position_label(r["k"], mlp.num_layers),
                    "n_in": spec.n_in,
                    "n_out": spec.n_out,
                    "dim": spec.size,
                    "final_train_loss": float(r["result"].best_value),
                    "val_best_cost": float(r["val_best_cost"]),
                    "iter_at_val_best": int(r["iter_at_val_best"]),
                    "iterations_run": int(r["result"].iterations),
                    "converged": bool(r["result"].converged),
                    "convergence_reason": r["result"].convergence_reason,
                    "layer_time": float(r["layer_time"]),
                })
                print(f"    [r{round_idx}] capa {r['k']} ({_position_label(r['k'], mlp.num_layers)}) "
                      f"[{spec.n_in}→{spec.n_out}, dim={spec.size}]: "
                      f"train_loss={float(r['result'].best_value):.4E}  "
                      f"val_best={r['val_best_cost']:.4f}  "
                      f"iters={r['result'].iterations}  "
                      f"{r['layer_time']:.2f}s")

            print(f"    ⏱ [r{round_idx}] paralelo={round_wall_time:.2f}s   "
                  f"Σ secuencial={seq_equiv:.2f}s   speedup≈{speedup:.2f}×")

    finally:
        # Liberar tensores de los opts persistentes (equivalente a salir del context manager).
        for opt in opts:
            opt.__exit__(None, None, None)

    total_parallel = sum(parallel_wall_times)
    total_sequential = sum(sequential_equiv_times)
    total_speedup = total_sequential / total_parallel if total_parallel > 0 else 0.0

    return {
        "params_best": current,
        "per_layer": per_layer,
        "n_rounds": n_rounds,
        "parallel_wall_time": total_parallel,
        "sequential_equivalent_time": total_sequential,
        "speedup": total_speedup,
    }

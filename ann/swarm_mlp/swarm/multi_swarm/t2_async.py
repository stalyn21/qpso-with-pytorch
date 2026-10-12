"""T2 — BCD asíncrono lock-free (Hogwild-style).

Versión paralelizada de T1 que preserva la semántica de propagación inmediata
del block coordinate descent Gauss-Seidel, pero corriendo las capas en threads
concurrentes contra un estado compartido sin barreras de sincronización.

Cada capa tiene su propio QDPSO persistente y un thread dedicado. Los threads
leen y escriben sobre el tensor compartido `current` sin locks: cada step lee
un snapshot ad-hoc de `current` para construir su cost_fn y, después de avanzar
un paso del QDPSO, escribe su nuevo `gbest` en su slice `current[layer_k]`.
La siguiente lectura desde otro thread verá ese update — es Gauss-Seidel
relajado, no Jacobi (donde todo el round corre contra un snapshot fijo).

Por qué lock-free funciona (Hogwild!-style):
- Las escrituras de threads distintos van a slices DISJUNTOS del `current`
  (cada capa escribe sobre su offset). No hay write-write conflict real.
- Las lecturas (`current.clone()`) pueden leer un estado intermedio donde
  algunas capas son recientes y otras están un step atrás, pero ningún byte
  es "garbage": son siempre floats válidos. El paper Hogwild! mostró que
  para SGD-like es benigno; para QDPSO la semántica es similar (búsqueda
  estocástica robusta a ruido en el contexto de la fitness).

Diferencias clave con las otras técnicas:
- vs T1 (BCD GS secuencial): T2 es la misma semántica de propagación inmediata
  pero corre las capas en paralelo. Sin barreras → no es GS estricto sino una
  relajación asíncrona, pero la información de cada capa fluye igual a las demás.
- vs T2-jacobi: T2-jacobi congela `current` durante todo el round; T2 nunca
  congela — siempre lee lo más reciente.
- vs T3 (lockstep GS iter-por-iter): T3 sincroniza al final de cada iter
  (todos avanzan 1 paso, luego intercambian). T2 nunca sincroniza durante la
  sesión; cada thread avanza a su ritmo.

En GPU usamos un `cuda.Stream` por thread para que los kernels se solapen
en el scheduler de CUDA. En CPU el GIL serializa el trabajo Python pero las
operaciones tensoriales liberan el GIL durante kernels nativos, así que sigue
habiendo solapamiento parcial.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch

from core.mlp import MLP
from core.sessions import evaluate_train_loss
from swarm.optimizer_factory import SwarmOptimizer, build_swarm_optimizer, swap_cost_fn
from swarm.warm_start import warm_start_cluster


def _position_label(k: int, num_layers: int) -> str:
    if k == num_layers - 1:
        return "salida"
    if k == 0:
        return "entrada"
    return f"oculta{k}"


def _async_layer_worker(
    *,
    mlp: MLP,
    data: Dict[str, Any],
    current: torch.Tensor,    # tensor COMPARTIDO entre threads (lock-free)
    cfg: Dict[str, Any],
    k: int,
    opt: SwarmOptimizer,
    cuda_stream: Optional["torch.cuda.Stream"] = None,
) -> Dict[str, Any]:
    """Worker async para la capa k. Avanza max_iter steps lock-free.

    Cada step:
        1. Lee `frozen = current.clone()` — snapshot ad-hoc del estado global
           (puede traer updates parciales de otras capas, eso es correcto).
        2. Construye cost_fn vs `frozen` y la inyecta en el opt persistente.
        3. Avanza un paso del QDPSO (kernel_update + boundary + update_best).
        4. Escribe `current[layer_k] = opt.gbest` — visible para los otros threads.
        5. Si gbest mejoró: evalúa val con un snapshot ad-hoc + el nuevo gbest;
           si es el mejor modelo que ESTE worker ha visto, guarda el snapshot
           COMPLETO (todas las capas), no sólo su segmento.

    El worker devuelve su mejor modelo completo. `run_session_t2_async` elige
    entre los de todos los workers: así el modelo retornado es siempre uno que se
    evaluó tal cual en val. (Antes se recombinaba el mejor segmento de cada capa,
    cada uno medido contra un contexto distinto: una combinación nunca evaluada.)
    Cada worker escribe sólo su propio registro, así que la selección no agrega
    locks ni estado compartido.
    """
    spec = mlp.layer_specs[k]
    X_train, y_train = data["X_train"], data["y_train"]
    X_val = data["X_val"]
    y_val_f = data["y_val"].float()
    lambda_l2 = cfg["lambda_l2"]
    max_iter = int(cfg["max_iter"])

    # Mejor modelo COMPLETO (por val, desempate por train_loss) visto por este worker.
    val_best = {
        "params": None,
        "val_cost": float("inf"),
        "train_loss": float("inf"),
        "iter_at_best": 0,
    }

    def _step():
        # 1. Lectura lock-free del estado global.
        frozen = current.clone()

        # 2. Cost_fn fresca contra frozen.
        fresh_cf = mlp.build_layer_cost_fn(
            X=X_train, y=y_train, k=k,
            frozen_flat=frozen, lambda_l2=lambda_l2,
        )
        swap_cost_fn(opt, fresh_cf)

        # 3. Un paso del QDPSO (kernel_update + boundary + update_best + contador).
        improved = opt.step()

        # 4. Escritura lock-free al slice de esta capa.
        current[spec.offset : spec.offset + spec.size] = opt.gbest

        return improved

    def _track_best(step: int) -> None:
        # Eval val con snapshot ad-hoc + gbest fresco de esta capa.
        candidate = current.clone()
        candidate[spec.offset : spec.offset + spec.size] = opt.gbest
        with torch.no_grad():
            y_pred = mlp.predict(X_val, candidate)
        val_cost = ((y_val_f - y_pred.float()) ** 2).mean().item()
        if val_cost > val_best["val_cost"]:
            return
        # Misma regla que T3: menor val gana; con val igual, menor train_loss
        # (modelo más convergido). El train_loss sólo se calcula si hace falta.
        train_loss = evaluate_train_loss(mlp, candidate, X_train, y_train, lambda_l2)
        if val_cost < val_best["val_cost"] or train_loss < val_best["train_loss"]:
            val_best["val_cost"] = val_cost
            val_best["train_loss"] = train_loss
            val_best["params"] = candidate
            val_best["iter_at_best"] = step + 1

    layer_start = time.time()

    stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream is not None else nullcontext()
    with stream_ctx:
        for step in range(max_iter):
            if _step():
                _track_best(step)

    layer_time = time.time() - layer_start

    return {
        "k": k,
        "spec": spec,
        "best_params": val_best["params"],
        "val_best_cost": val_best["val_cost"],
        "val_best_train_loss": val_best["train_loss"],
        "iter_at_val_best": val_best["iter_at_best"],
        "final_train_loss": float(opt.gbest_value),
        "iterations_run": int(opt.iters),
        "layer_time": layer_time,
    }


def run_session_t2_async(
    mlp: MLP,
    data: Dict[str, Any],
    params_init: torch.Tensor,
    cfg: Dict[str, Any],
    session_seed: int,
) -> Dict[str, Any]:
    device_str = cfg.get("device_resolved", cfg.get("device", "cpu"))
    use_cuda = torch.device(device_str).type == "cuda"
    low, high = cfg["bounds"]

    # Estado compartido entre threads. Las escrituras a slices disjuntos son
    # lock-free; las lecturas (clone) son ad-hoc por step.
    current = params_init.clone()

    # Warm-start cluster: K partículas concentradas alrededor de params_init[layer_k],
    # P-K random para preservar exploración. Aplicado UNA vez al crear cada opt persistente.
    # Reduce dramáticamente la varianza por cold start (sin esto, ~1 de cada 5 inicializaciones
    # producía resultados catastróficos en CV con datasets pequeños).
    k_seeded = int(cfg.get("warm_particles", 10))
    noise_scale = float(cfg.get("warm_noise", 0.05))
    k_seeded_eff = max(1, min(k_seeded, cfg["n_particles"]))

    # Crear UN QDPSO persistente por capa.
    opts: List[SwarmOptimizer] = []
    for k in range(mlp.num_layers):
        spec = mlp.layer_specs[k]
        bounds = [(low, high)] * spec.size
        # cost_fn inicial vs params_init; se reemplaza en cada step desde el worker.
        init_cf = mlp.build_layer_cost_fn(
            X=data["X_train"], y=data["y_train"],
            k=k, frozen_flat=params_init,
            lambda_l2=cfg["lambda_l2"],
        )
        opt = build_swarm_optimizer(
            cfg,
            cf=init_cf,
            dim=spec.size,
            bounds=bounds,
            seed=session_seed * 10000 + k,
            dtype=mlp.dtype,
        )
        # Ancla la búsqueda en una región conocida del espacio de pesos.
        seed_segment = params_init[spec.offset : spec.offset + spec.size]
        warm_start_cluster(opt, seed_segment, k_seeded_eff, noise_scale)
        opts.append(opt)

    streams = (
        [torch.cuda.Stream() for _ in range(mlp.num_layers)]
        if use_cuda else [None] * mlp.num_layers
    )

    submission_order = list(range(mlp.num_layers - 1, -1, -1))
    session_start = time.time()

    try:
        with ThreadPoolExecutor(max_workers=mlp.num_layers) as pool:
            futures = {
                k: pool.submit(
                    _async_layer_worker,
                    mlp=mlp, data=data, current=current,
                    cfg=cfg, k=k, opt=opts[k],
                    cuda_stream=streams[k],
                )
                for k in submission_order
            }
            results = [futures[k].result() for k in submission_order]

        if use_cuda:
            torch.cuda.synchronize()
    finally:
        # Liberar tensores de los opts persistentes.
        for opt in opts:
            opt.release()

    session_time = time.time() - session_start

    # Elegir el mejor modelo COMPLETO entre el punto de partida y el mejor de cada
    # worker (menor val; con val igual, menor train_loss). `params_init` entra como
    # candidato para que la sesión nunca devuelva algo peor en val que su arranque.
    # El orden de los candidatos (init, luego capa 0..N-1) fija el desempate final.
    X_val, y_val_f = data["X_val"], data["y_val"].float()
    with torch.no_grad():
        y_pred_init = mlp.predict(X_val, params_init)
    best_val_cost = ((y_val_f - y_pred_init.float()) ** 2).mean().item()
    best_train_loss = evaluate_train_loss(
        mlp, params_init, data["X_train"], data["y_train"], cfg["lambda_l2"]
    )
    params_best = params_init.clone()
    winner_layer = None
    iter_at_best = 0
    for r in sorted(results, key=lambda x: x["k"]):
        if r["best_params"] is None:
            continue
        candidate_key = (r["val_best_cost"], r["val_best_train_loss"])
        if candidate_key < (best_val_cost, best_train_loss):
            best_val_cost, best_train_loss = candidate_key
            params_best = r["best_params"]
            winner_layer = r["k"]
            iter_at_best = r["iter_at_val_best"]

    # Métricas: speedup vs costo secuencial-equivalente.
    seq_equiv = sum(r["layer_time"] for r in results)
    speedup = seq_equiv / session_time if session_time > 0 else 0.0

    per_layer = []
    # Ordenar de capa 0 a N-1 para output natural (igual que T3).
    for r in sorted(results, key=lambda x: x["k"]):
        spec = r["spec"]
        per_layer.append({
            "layer": r["k"],
            "position": _position_label(r["k"], mlp.num_layers),
            "n_in": spec.n_in,
            "n_out": spec.n_out,
            "dim": spec.size,
            "final_train_loss": r["final_train_loss"],
            "val_best_cost": float(r["val_best_cost"]),
            "iter_at_val_best": int(r["iter_at_val_best"]),
            "iterations_run": r["iterations_run"],
            "layer_time": float(r["layer_time"]),
        })

    # Imprimir en orden back-to-front para coherencia con T1/T3.
    for k in submission_order:
        info = per_layer[k]
        print(f"    capa {k} ({info['position']}) "
              f"[{info['n_in']}→{info['n_out']}, dim={info['dim']}]: "
              f"train_loss={info['final_train_loss']:.4E}  "
              f"val_best={info['val_best_cost']:.4f}  "
              f"iters={info['iterations_run']}  "
              f"{info['layer_time']:.2f}s")

    winner_label = "params_init" if winner_layer is None else f"capa {winner_layer}"
    print(f"    🔀 async lock-free: best_val={best_val_cost:.4f} "
          f"(snapshot de {winner_label}, iter {iter_at_best})   "
          f"paralelo={session_time:.2f}s   "
          f"Σ secuencial={seq_equiv:.2f}s   speedup≈{speedup:.2f}×")

    return {
        "params_best": params_best,
        "per_layer": per_layer,
        "best_val_cost_during_session": float(best_val_cost),
        "iter_at_best": int(iter_at_best),
        "session_time": session_time,
        "parallel_wall_time": session_time,
        "sequential_equivalent_time": seq_equiv,
        "speedup": speedup,
    }

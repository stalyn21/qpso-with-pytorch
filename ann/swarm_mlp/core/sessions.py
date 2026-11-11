"""Infraestructura común de sesiones:
- Loop de sesiones con selección por VAL (nunca por test).
- Criterio de selección (el mismo que usan las técnicas, `core.selection.Selector`):
  regla de k errores estándar sobre la métrica principal de validación (acc o
  balanced acc); entre los empatados, menor cross-entropy de validación. Con
  acc y k = 0 es la regla histórica. El MSE entre etiquetas enteras se sigue
  reportando, pero NO decide.
- Cadena inter-sesión: el `params_init` de la sesión k+1 es SIEMPRE el mejor global
  hasta ahora (no el de la sesión inmediata anterior). Una sesión peor no envenena
  la cadena.
- Test SOLO al final, UNA vez, sobre la sesión ganadora (medición no sesgada).
- Costo de cada sesión: evaluaciones de la función de costo, tiempo de pared (con
  la GPU sincronizada), memoria pico de GPU y traza de validación (`core.budget`).
- Persistencia estructurada (JSON + npy), con la procedencia de la corrida.
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List
import json
import time
import numpy as np
import torch

from core.bounds import layer_range
from core.budget import METER, TRACE_COLUMNS
from core.diagnostics import format_param_stats, param_stats
from core.mlp import MLP
from core.metrics import accuracy, classification_metrics, confusion_matrix, mse_int
from core.selection import Selector, val_score
from core.loss import cross_entropy_from_logits, l2_penalty
from core.provenance import collect_provenance


# Tipo del callable de una técnica: recibe contexto y retorna dict con resultados de la sesión.
TechniqueFn = Callable[..., Dict[str, Any]]


@dataclass
class SessionResult:
    session: int
    seed: int
    val_acc: float           # accuracy de validación (métrica principal si selection_metric=acc)
    val_ce: float            # desempate: cross-entropy de validación (menor es mejor)
    val_cost: float          # reporte: MSE entre etiquetas enteras; NO usado para selección
    train_loss: float        # reporte: CE + L2 sobre train con params_best
    test_acc: float          # reporte; NO usado para selección
    test_ce: float           # reporte
    test_cost: float         # reporte: MSE entre etiquetas enteras
    val_test_acc_gap: float  # diagnóstico: val_acc - test_acc (positivo = posible overfit a val)
    l2_norm_sq: float = 0.0  # ||params||² — reporte
    guard_triggered: bool = False  # True si el guard de no-regresión hizo fallback a params_init
    per_layer: list = field(default_factory=list)
    session_time: float = 0.0
    train_acc: float = 0.0          # reporte: accuracy sobre el conjunto de fitness
    val_balanced_acc: float = 0.0   # métrica principal si selection_metric=bacc
    val_macro_f1: float = 0.0
    test_macro_f1: float = 0.0
    test_balanced_acc: float = 0.0
    # Costo de la sesión (ver core/budget.py). Una evaluación = una partícula
    # evaluada sobre todo el conjunto de fitness.
    fitness_evals: int = 0
    fitness_calls: int = 0
    selection_evals: int = 0        # modelos evaluados en validación para seleccionar
    selection_time: float = 0.0     # segundos de session_time gastados en eso
    gpu_peak_mb: float = 0.0        # memoria pico de GPU durante la sesión (0 en CPU)
    trace: list = field(default_factory=list)   # puntos [fitness_evals, seconds, val_acc, val_ce]
    selection: dict = field(default_factory=dict)  # diagnóstico de la regla dentro de la técnica
    param_stats: dict = field(default_factory=dict)  # escala y borde por capa (core/diagnostics.py)


def evaluate_full(mlp: MLP, flat_params: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
    """(acc, ce, mse_int, y_pred) con un solo forward."""
    with torch.no_grad():
        logits = mlp.forward(X, flat_params)
        y_pred = logits.argmax(dim=-1)
        ce = cross_entropy_from_logits(logits, y, dim_classes=-1).mean().item()
    return accuracy(y, y_pred), ce, mse_int(y, y_pred), y_pred


def evaluate_train_loss(
    mlp: MLP,
    flat_params: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    lambda_l2: float,
) -> float:
    """CE + L2 sobre train, evaluado con flat_params. Es lo que el QDPSO optimiza.

    La L2 usa la forma del modelo (`mlp.l2`, ver core/loss.py:L2_FORMS). La CNN no
    tiene `l2`: usa la forma "mean".
    """
    with torch.no_grad():
        logits = mlp.forward(X_train, flat_params)
        ce = cross_entropy_from_logits(logits, y_train, dim_classes=-1).mean().item()
        if hasattr(mlp, "l2"):
            l2 = float(mlp.l2(flat_params, lambda_l2, n_train=X_train.shape[0]).item())
        else:
            l2 = float(l2_penalty(
                flat_params, lambda_l2,
                d_total=mlp.d_total, n_train=X_train.shape[0],
            ).item())
    return ce + l2


def _sync(device: torch.device) -> None:
    """Espera a que la GPU termine lo encolado, para que el reloj mida trabajo real."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def l2_norm_squared(params: torch.Tensor) -> float:
    """||params||² — reporte por sesión."""
    with torch.no_grad():
        return float((params ** 2).sum().item())


def init_params_from_swarms(
    mlp: MLP,
    cfg: Dict[str, Any],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Inicializa los parámetros como concatenación de muestras random por capa.

    Cada capa se muestrea uniforme en su rango de búsqueda (`core.bounds`): con
    `bounds_mode="fixed"` es el mismo rango para todas, igual que antes.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    segments = []
    for k in range(mlp.num_layers):
        dim_k = mlp.layer_specs[k].size
        low, high = layer_range(mlp, cfg, k)
        r = torch.rand(dim_k, generator=g, dtype=torch.float64)
        segments.append(r * (high - low) + low)
    flat = torch.cat(segments)
    return flat.to(device=device, dtype=dtype)


def run_sessions(
    technique_fn: TechniqueFn,
    mlp: MLP,
    data: Dict[str, Any],
    cfg: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    device = data["X_train"].device
    dtype = mlp.dtype

    params_init = init_params_from_swarms(mlp, cfg, cfg["seed"], device, dtype)

    sessions: List[SessionResult] = []
    best_idx = 0
    best_params = params_init.clone()
    # Selección entre sesiones con la misma regla que dentro de las técnicas.
    session_selector = Selector.from_cfg(cfg, data)
    criterion = session_selector.criterion

    total_start = time.time()

    for k in range(cfg["n_sessions"]):
        session_seed = cfg["seed"] + k + 1
        print(f"\n━━━ Sesión {k+1}/{cfg['n_sessions']}  (seed={session_seed}) ━━━")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        METER.start()
        t0 = time.perf_counter()
        tech_out = technique_fn(
            mlp=mlp,
            data=data,
            params_init=params_init,
            cfg=cfg,
            session_seed=session_seed,
        )
        _sync(device)
        session_time = time.perf_counter() - t0
        cost = METER.snapshot()
        gpu_peak_mb = (torch.cuda.max_memory_allocated(device) / 2**20
                       if device.type == "cuda" else 0.0)

        params_k = tech_out["params_best"]

        # Guard A — no-regresión: la salida de la técnica y params_init compiten con la
        # regla de selección (la salida entra primero y gana los empates). Si gana
        # params_init, descartamos el resultado. Evita que una sesión catastrófica
        # produzca peores params que los del contexto de partida (la cadena monótona
        # ya protege el best global, pero el reporte por sesión queda más limpio).
        guard = Selector.from_cfg(cfg, data)
        score_k = val_score(mlp, params_k, data["X_val"], data["y_val"], data["n_classes"])
        score_init = val_score(mlp, params_init, data["X_val"], data["y_val"], data["n_classes"])
        guard.offer(score_k, "technique")
        guard.offer(score_init, "init")
        guard_triggered = guard.choose()[0] == "init"
        if guard_triggered:
            print(f"  ⚠ Guard A activado: técnica produjo val ACC={score_k.acc:.4f} "
                  f"BACC={score_k.bacc:.4f} CE={score_k.ce:.4f}, peor que params_init "
                  f"(ACC={score_init.acc:.4f} BACC={score_init.bacc:.4f} CE={score_init.ce:.4f}) "
                  f"según {criterion}. Fallback a params_init.")
            params_k = params_init.clone()
        val_acc, val_ce, val_cost, _ = evaluate_full(mlp, params_k, data["X_val"], data["y_val"])

        test_acc, test_ce, test_cost, y_test_pred_k = evaluate_full(
            mlp, params_k, data["X_test"], data["y_test"])
        _, _, _, y_val_pred_k = evaluate_full(mlp, params_k, data["X_val"], data["y_val"])
        train_acc = evaluate_full(mlp, params_k, data["X_train"], data["y_train"])[0]
        test_cls = classification_metrics(data["y_test"], y_test_pred_k, data["n_classes"])
        val_cls = classification_metrics(data["y_val"], y_val_pred_k, data["n_classes"])
        train_loss = evaluate_train_loss(
            mlp, params_k, data["X_train"], data["y_train"], cfg["lambda_l2"]
        )
        l2_norm_sq = l2_norm_squared(params_k)
        stats_k = param_stats(mlp, params_k, cfg)

        val_test_acc_gap = val_acc - test_acc                       # diagnóstico, no selección

        print(f"  val: ACC={val_acc:.4f}  BACC={val_cls['balanced_acc']:.4f}  "
              f"CE={val_ce:.4f}  MSE={val_cost:.4f}  "
              f"train_loss={train_loss:.4f}  ||θ||²={l2_norm_sq:.2f}   │   "
              f"test: ACC={test_acc:.4f}  CE={test_ce:.4f}  MSE={test_cost:.4f}   │   "
              f"gap(val-test)={val_test_acc_gap:+.4f}"
              f"{'   [guard]' if guard_triggered else ''}")
        print(f"  costo: {cost['fitness_evals']} evaluaciones de fitness  "
              f"⏱ {session_time:.2f}s (selección {cost['selection_time']:.2f}s)"
              + (f"  GPU pico {gpu_peak_mb:.0f} MB" if device.type == "cuda" else ""))
        print(f"  {format_param_stats(stats_k)}")

        sessions.append(SessionResult(
            session=k, seed=session_seed,
            val_acc=float(val_acc), val_ce=float(val_ce), val_cost=float(val_cost),
            train_loss=float(train_loss),
            test_acc=float(test_acc), test_ce=float(test_ce), test_cost=float(test_cost),
            val_test_acc_gap=float(val_test_acc_gap),
            l2_norm_sq=float(l2_norm_sq),
            guard_triggered=bool(guard_triggered),
            per_layer=tech_out.get("per_layer", []),
            session_time=float(session_time),
            train_acc=float(train_acc),
            val_balanced_acc=val_cls["balanced_acc"],
            val_macro_f1=val_cls["macro_f1"],
            test_macro_f1=test_cls["macro_f1"],
            test_balanced_acc=test_cls["balanced_acc"],
            fitness_evals=int(cost["fitness_evals"]),
            fitness_calls=int(cost["fitness_calls"]),
            selection_evals=int(cost["selection_evals"]),
            selection_time=float(cost["selection_time"]),
            gpu_peak_mb=float(gpu_peak_mb),
            trace=cost["trace"],
            selection=tech_out.get("selection", {}),
            param_stats=stats_k,
        ))

        # Selección entre sesiones con la regla de selección. Un empate exacto
        # (p. ej. tras el fallback del Guard A) conserva la anterior.
        session_selector.offer(
            val_score(mlp, params_k, data["X_val"], data["y_val"], data["n_classes"]),
            params_k.clone(), {"session": k},
        )
        prev_idx = best_idx if k > 0 else None
        best_params, _, sess_info = session_selector.choose()
        best_idx = sess_info["session"]
        if best_idx != prev_idx:
            print(f"  ✨ nueva mejor sesión "
                  f"(val ACC={val_acc:.4f}, CE={val_ce:.4f}, train={train_loss:.4f})")

        # Cadena inter-sesión: SIEMPRE arrancamos desde el mejor global hasta ahora,
        # no desde el resultado inmediato. Una sesión peor no envenena la cadena.
        params_init = best_params.clone()

    total_time = time.time() - total_start
    # Costo de producir el modelo ganador: las sesiones están encadenadas, así que
    # cuenta todo lo corrido hasta la sesión ganadora inclusive.
    to_winner = sessions[: best_idx + 1]

    # Medición final única sobre test (ganador elegido por val)
    final_test_acc, final_test_ce, final_test_cost, y_test_pred = evaluate_full(
        mlp, best_params, data["X_test"], data["y_test"]
    )
    final_val_acc, final_val_ce, final_val_cost, y_val_pred = evaluate_full(
        mlp, best_params, data["X_val"], data["y_val"]
    )
    final_val_cls = classification_metrics(data["y_val"], y_val_pred, data["n_classes"])

    cm = confusion_matrix(data["y_test"], y_test_pred, data["n_classes"])

    final_train_loss = evaluate_train_loss(
        mlp, best_params, data["X_train"], data["y_train"], cfg["lambda_l2"]
    )
    final_train_acc = evaluate_full(mlp, best_params, data["X_train"], data["y_train"])[0]
    final_test_cls = classification_metrics(data["y_test"], y_test_pred, data["n_classes"])

    winner = {
        "session": int(best_idx),
        "selection_criterion": criterion,
        "selection_metric": session_selector.metric,
        "selection_se": session_selector.n_se,
        "val_acc": float(final_val_acc),
        "val_balanced_acc": float(final_val_cls["balanced_acc"]),
        "val_ce": float(final_val_ce),
        "val_cost": float(final_val_cost),
        "train_loss": float(final_train_loss),
        "train_acc": float(final_train_acc),
        "final_test_acc": float(final_test_acc),
        "final_test_ce": float(final_test_ce),
        "final_test_cost": float(final_test_cost),
        "final_test_macro_f1": final_test_cls["macro_f1"],
        "final_test_balanced_acc": final_test_cls["balanced_acc"],
        "fitness_evals_to_winner": int(sum(s.fitness_evals for s in to_winner)),
        "time_to_winner": float(sum(s.session_time for s in to_winner)),
        "test_confusion_matrix": cm,
        "param_stats": param_stats(mlp, best_params, cfg),
    }

    # Persistencia
    output_dir.mkdir(parents=True, exist_ok=True)

    # Curvas de aprendizaje a través de las sesiones (para análisis posterior y plots)
    learning_curve = {
        "session_idx":       [s.session for s in sessions],
        "val_acc":           [s.val_acc for s in sessions],
        "val_balanced_acc":  [s.val_balanced_acc for s in sessions],
        "val_ce":            [s.val_ce for s in sessions],
        "val_mse":           [s.val_cost for s in sessions],
        "test_acc":          [s.test_acc for s in sessions],
        "test_ce":           [s.test_ce for s in sessions],
        "test_mse":          [s.test_cost for s in sessions],
        "train_loss":        [s.train_loss for s in sessions],
        "val_test_acc_gap":  [s.val_test_acc_gap for s in sessions],
        "train_acc":         [s.train_acc for s in sessions],
        "test_macro_f1":     [s.test_macro_f1 for s in sessions],
        "fitness_evals_cum": list(np.cumsum([s.fitness_evals for s in sessions]).tolist()),
        "time_cum":          list(np.cumsum([s.session_time for s in sessions]).tolist()),
    }

    result_dict = {
        "config": cfg,
        "data_split": {
            "train": int(data["X_train"].shape[0]),
            "val":   int(data["X_val"].shape[0]),
            "test":  int(data["X_test"].shape[0]),
            "n_features": data["n_features"],
            "n_classes": data["n_classes"],
        },
        "sessions": [asdict(s) for s in sessions],
        "winner_by_val": winner,
        "learning_curve": learning_curve,
        "trace_columns": list(TRACE_COLUMNS),
        "total_time": float(total_time),
        "total_fitness_evals": int(sum(s.fitness_evals for s in sessions)),
        "gpu_peak_mb": float(max((s.gpu_peak_mb for s in sessions), default=0.0)),
        "provenance": collect_provenance(device),
    }

    with open(output_dir / "run.json", "w") as f:
        json.dump(result_dict, f, indent=2, default=_json_default)

    np.save(output_dir / "gbest.npy", best_params.detach().cpu().numpy())

    print("\n" + "═" * 70)
    print(f"🏆 Ganadora: sesión {best_idx}   (criterio: {criterion})")
    print(f"   val  ACC={final_val_acc:.4f}  BACC={final_val_cls['balanced_acc']:.4f}  "
          f"CE={final_val_ce:.4f}  MSE={final_val_cost:.4f}   (criterio de selección)")
    print(f"   test ACC={final_test_acc:.4f}  CE={final_test_ce:.4f}  MSE={final_test_cost:.4f}   (final, no sesgado)")
    print(f"⏱  Tiempo total: {total_time:.2f}s   "
          f"evaluaciones de fitness: {result_dict['total_fitness_evals']} "
          f"(hasta la ganadora: {winner['fitness_evals_to_winner']})")
    print(f"📁 Output: {output_dir}")
    print("═" * 70)

    return result_dict


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)

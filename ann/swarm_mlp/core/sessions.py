"""Infraestructura común de sesiones:
- Loop de sesiones con selección por VAL (nunca por test).
- Criterio de selección (el mismo que usan las técnicas, `core.metrics.selection_key`):
  mayor accuracy de validación y, con la misma accuracy, menor cross-entropy de
  validación. El MSE entre etiquetas enteras se sigue reportando, pero NO decide.
- Cadena inter-sesión: el `params_init` de la sesión k+1 es SIEMPRE el mejor global
  hasta ahora (no el de la sesión inmediata anterior). Una sesión peor no envenena
  la cadena.
- Test SOLO al final, UNA vez, sobre la sesión ganadora (medición no sesgada).
- Persistencia estructurada (JSON + npy).
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List
import json
import time
import numpy as np
import torch

from core.mlp import MLP
from core.metrics import SELECTION_CRITERION, accuracy, mse_int, confusion_matrix
from core.loss import cross_entropy_from_logits, l2_penalty


# Tipo del callable de una técnica: recibe contexto y retorna dict con resultados de la sesión.
TechniqueFn = Callable[..., Dict[str, Any]]


@dataclass
class SessionResult:
    session: int
    seed: int
    val_acc: float           # criterio de selección primario (mayor es mejor)
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


def evaluate_model(mlp: MLP, flat_params: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
    with torch.no_grad():
        y_pred = mlp.predict(X, flat_params)
    return mse_int(y, y_pred), accuracy(y, y_pred), y_pred


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

    L2 normalizado por (d_total · n_train) — ver core/loss.py:l2_penalty.
    """
    with torch.no_grad():
        logits = mlp.forward(X_train, flat_params)
        ce = cross_entropy_from_logits(logits, y_train, dim_classes=-1).mean().item()
        l2 = float(l2_penalty(
            flat_params, lambda_l2,
            d_total=mlp.d_total, n_train=X_train.shape[0],
        ).item())
    return ce + l2


def l2_norm_squared(params: torch.Tensor) -> float:
    """||params||² — reporte por sesión."""
    with torch.no_grad():
        return float((params ** 2).sum().item())


def init_params_from_swarms(
    mlp: MLP,
    bounds: tuple,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Inicializa los parámetros como concatenación de muestras random por capa.

    Matemáticamente equivalente a un único random global (la distribución es la misma),
    pero construido por capa para honrar la semántica "cada swarm aporta su segmento".
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    low, high = bounds
    segments = []
    for k in range(mlp.num_layers):
        dim_k = mlp.layer_specs[k].size
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

    params_init = init_params_from_swarms(mlp, cfg["bounds"], cfg["seed"], device, dtype)

    sessions: List[SessionResult] = []
    best_idx = 0
    # Clave de selección (1 - val_acc, val_ce): gana la menor, comparada como tupla.
    best_key = (float("inf"), float("inf"))
    best_params = params_init.clone()

    total_start = time.time()

    for k in range(cfg["n_sessions"]):
        session_seed = cfg["seed"] + k + 1
        print(f"\n━━━ Sesión {k+1}/{cfg['n_sessions']}  (seed={session_seed}) ━━━")

        t0 = time.time()
        tech_out = technique_fn(
            mlp=mlp,
            data=data,
            params_init=params_init,
            cfg=cfg,
            session_seed=session_seed,
        )
        session_time = time.time() - t0

        params_k = tech_out["params_best"]
        val_acc, val_ce, val_cost, _ = evaluate_full(mlp, params_k, data["X_val"], data["y_val"])
        key = (1.0 - val_acc, val_ce)

        # Guard A — no-regresión: si la salida de la técnica es PEOR (por el criterio
        # de selección) que el contexto inicial de la sesión, descartamos el resultado
        # y caemos a params_init. Esto evita que una sesión catastrófica produzca
        # peores params que los del contexto de partida (la cadena monótona ya protege
        # el best global, pero el reporte por sesión queda más limpio).
        val_acc_init, val_ce_init, val_cost_init, _ = evaluate_full(
            mlp, params_init, data["X_val"], data["y_val"]
        )
        key_init = (1.0 - val_acc_init, val_ce_init)
        guard_triggered = False
        if key_init < key:
            print(f"  ⚠ Guard A activado: técnica produjo val ACC={val_acc:.4f} CE={val_ce:.4f}, "
                  f"peor que params_init (ACC={val_acc_init:.4f} CE={val_ce_init:.4f}). "
                  f"Fallback a params_init.")
            params_k = params_init.clone()
            val_acc, val_ce, val_cost = val_acc_init, val_ce_init, val_cost_init
            key = key_init
            guard_triggered = True

        test_acc, test_ce, test_cost, _ = evaluate_full(mlp, params_k, data["X_test"], data["y_test"])
        train_loss = evaluate_train_loss(
            mlp, params_k, data["X_train"], data["y_train"], cfg["lambda_l2"]
        )
        l2_norm_sq = l2_norm_squared(params_k)

        val_test_acc_gap = val_acc - test_acc                       # diagnóstico, no selección

        print(f"  val: ACC={val_acc:.4f}  CE={val_ce:.4f}  MSE={val_cost:.4f}  "
              f"train_loss={train_loss:.4f}  ||θ||²={l2_norm_sq:.2f}   │   "
              f"test: ACC={test_acc:.4f}  CE={test_ce:.4f}  MSE={test_cost:.4f}   │   "
              f"gap(val-test)={val_test_acc_gap:+.4f}"
              f"{'   [guard]' if guard_triggered else ''}")

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
        ))

        # Selección entre sesiones: mayor val_acc; con la misma, menor val_ce.
        # Un empate exacto (p. ej. tras el fallback del Guard A) conserva la anterior.
        if key < best_key:
            best_key = key
            best_idx = k
            best_params = params_k.clone()
            print(f"  ✨ nueva mejor sesión "
                  f"(val ACC={val_acc:.4f}, CE={val_ce:.4f}, train={train_loss:.4f})")

        # Cadena inter-sesión: SIEMPRE arrancamos desde el mejor global hasta ahora,
        # no desde el resultado inmediato. Una sesión peor no envenena la cadena.
        params_init = best_params.clone()

    total_time = time.time() - total_start

    # Medición final única sobre test (ganador elegido por val)
    final_test_acc, final_test_ce, final_test_cost, y_test_pred = evaluate_full(
        mlp, best_params, data["X_test"], data["y_test"]
    )
    final_val_acc, final_val_ce, final_val_cost, _ = evaluate_full(
        mlp, best_params, data["X_val"], data["y_val"]
    )

    cm = confusion_matrix(data["y_test"], y_test_pred, data["n_classes"])

    final_train_loss = evaluate_train_loss(
        mlp, best_params, data["X_train"], data["y_train"], cfg["lambda_l2"]
    )

    winner = {
        "session": int(best_idx),
        "selection_criterion": SELECTION_CRITERION,
        "val_acc": float(final_val_acc),
        "val_ce": float(final_val_ce),
        "val_cost": float(final_val_cost),
        "train_loss": float(final_train_loss),
        "final_test_acc": float(final_test_acc),
        "final_test_ce": float(final_test_ce),
        "final_test_cost": float(final_test_cost),
        "test_confusion_matrix": cm,
    }

    # Persistencia
    output_dir.mkdir(parents=True, exist_ok=True)

    # Curvas de aprendizaje a través de las sesiones (para análisis posterior y plots)
    learning_curve = {
        "session_idx":       [s.session for s in sessions],
        "val_acc":           [s.val_acc for s in sessions],
        "val_ce":            [s.val_ce for s in sessions],
        "val_mse":           [s.val_cost for s in sessions],
        "test_acc":          [s.test_acc for s in sessions],
        "test_ce":           [s.test_ce for s in sessions],
        "test_mse":          [s.test_cost for s in sessions],
        "train_loss":        [s.train_loss for s in sessions],
        "val_test_acc_gap":  [s.val_test_acc_gap for s in sessions],
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
        "total_time": float(total_time),
    }

    with open(output_dir / "run.json", "w") as f:
        json.dump(result_dict, f, indent=2, default=_json_default)

    np.save(output_dir / "gbest.npy", best_params.detach().cpu().numpy())

    print("\n" + "═" * 70)
    print(f"🏆 Ganadora: sesión {best_idx}   (criterio: mayor val ACC, desempate menor val CE)")
    print(f"   val  ACC={final_val_acc:.4f}  CE={final_val_ce:.4f}  MSE={final_val_cost:.4f}   (criterio de selección)")
    print(f"   test ACC={final_test_acc:.4f}  CE={final_test_ce:.4f}  MSE={final_test_cost:.4f}   (final, no sesgado)")
    print(f"⏱  Tiempo total: {total_time:.2f}s")
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

"""Infraestructura común de sesiones:
- Loop de sesiones con selección por VAL (nunca por test).
- Criterio de selección: score combinado = (1 - val_acc) + (val_mse / mse_max).
  Menor score = mejor. Ambas métricas pesan igual; se normaliza MSE por su máximo
  posible mse_max = (n_classes - 1)² para que esté en [0, 1] como (1 - val_acc).
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
from core.metrics import accuracy, mse_int, confusion_matrix
from core.loss import cross_entropy_from_logits, l2_penalty


# Tipo del callable de una técnica: recibe contexto y retorna dict con resultados de la sesión.
TechniqueFn = Callable[..., Dict[str, Any]]


@dataclass
class SessionResult:
    session: int
    seed: int
    val_cost: float
    val_acc: float
    val_score: float         # criterio de selección primario: (1 - val_acc) + val_mse / mse_max
    train_loss: float        # tie-breaker unbiased: CE + L2 sobre train con params_best
    test_cost: float         # reporte; NO usado para selección
    test_acc: float          # reporte
    test_score: float        # reporte; mismo cálculo que val_score pero sobre test
    val_test_acc_gap: float  # diagnóstico: val_acc - test_acc (positivo = posible overfit a val)
    l2_norm_sq: float = 0.0  # ||params||² — tie-breaker terciario (regularización Occam)
    guard_triggered: bool = False  # True si el guard de no-regresión hizo fallback a params_init
    per_layer: list = field(default_factory=list)
    session_time: float = 0.0


def combined_score(val_acc: float, val_mse: float, mse_max: float) -> float:
    """Score combinado para selección: menor es mejor.

    score = (1 - val_acc) + (val_mse / mse_max)

    Ambos términos están en [0, 1]: (1 - acc) es el error rate y val_mse/mse_max
    normaliza el error ordinal. Suma pondera ambos por igual.
    """
    return (1.0 - val_acc) + (val_mse / mse_max)


def evaluate_model(mlp: MLP, flat_params: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
    with torch.no_grad():
        y_pred = mlp.predict(X, flat_params)
    return mse_int(y, y_pred), accuracy(y, y_pred), y_pred


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
    """||params||² — usado como tie-breaker terciario en la selección entre sesiones.
    Modelo con menor norma L2 es "más simple" → tiende a generalizar mejor (Occam).
    """
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

    # mse_max para normalizar el MSE en el score combinado.
    # MSE máximo posible por muestra = (n_classes - 1)² (predecir la clase más extrema).
    n_classes = data["n_classes"]
    mse_max = max(1.0, float((n_classes - 1) ** 2))

    sessions: List[SessionResult] = []
    best_idx = 0
    best_score = float("inf")
    best_train_loss = float("inf")  # tie-breaker primario cuando val_score empata
    best_l2_norm_sq = float("inf")  # tie-breaker secundario: ||params||² (Occam)
    best_val_acc = -1.0             # tie-breaker terciario (mayor mejor)
    best_val_cost = float("inf")    # tie-breaker cuaternario (menor mejor)
    best_params = params_init.clone()

    # Tolerancias para tratar empates con ruido de estocasticidad del QDPSO.
    val_score_tol = 1e-9            # float equality
    train_loss_rel_tol = 0.05       # 5% relativo: train_loss difiere por ruido del QDPSO

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
        val_cost, val_acc, _ = evaluate_model(mlp, params_k, data["X_val"], data["y_val"])
        val_score = combined_score(val_acc, val_cost, mse_max)

        # Guard A — no-regresión: si la salida de la técnica es PEOR (por val_score)
        # que el contexto inicial de la sesión, descartamos el resultado y caemos a
        # params_init. Esto evita que una sesión catastrófica produzca peores params
        # que los del contexto de partida (la cadena monótona ya protege el best
        # global, pero el reporte por sesión queda más limpio).
        val_cost_init, val_acc_init, _ = evaluate_model(
            mlp, params_init, data["X_val"], data["y_val"]
        )
        val_score_init = combined_score(val_acc_init, val_cost_init, mse_max)
        guard_triggered = False
        if val_score_init < val_score:
            print(f"  ⚠ Guard A activado: técnica produjo val_score={val_score:.4f} "
                  f"> params_init val_score={val_score_init:.4f}. Fallback a params_init.")
            params_k = params_init.clone()
            val_cost, val_acc = val_cost_init, val_acc_init
            val_score = val_score_init
            guard_triggered = True

        test_cost, test_acc, _ = evaluate_model(mlp, params_k, data["X_test"], data["y_test"])
        train_loss = evaluate_train_loss(
            mlp, params_k, data["X_train"], data["y_train"], cfg["lambda_l2"]
        )
        l2_norm_sq = l2_norm_squared(params_k)

        test_score = combined_score(test_acc, test_cost, mse_max)  # solo reporte
        val_test_acc_gap = val_acc - test_acc                       # diagnóstico, no selección

        print(f"  val: ACC={val_acc:.4f}  MSE={val_cost:.4f}  score={val_score:.4f}  "
              f"train_loss={train_loss:.4f}  ||θ||²={l2_norm_sq:.2f}   │   "
              f"test: ACC={test_acc:.4f}  MSE={test_cost:.4f}  score={test_score:.4f}   │   "
              f"gap(val-test)={val_test_acc_gap:+.4f}"
              f"{'   [guard]' if guard_triggered else ''}")

        sessions.append(SessionResult(
            session=k, seed=session_seed,
            val_cost=float(val_cost), val_acc=float(val_acc),
            val_score=float(val_score),
            train_loss=float(train_loss),
            test_cost=float(test_cost), test_acc=float(test_acc),
            test_score=float(test_score),
            val_test_acc_gap=float(val_test_acc_gap),
            l2_norm_sq=float(l2_norm_sq),
            guard_triggered=bool(guard_triggered),
            per_layer=tech_out.get("per_layer", []),
            session_time=float(session_time),
        ))

        # Selección con jerarquía robusta a ruido:
        #   1. menor val_score (estrictamente, con tolerancia de float)
        #   2. menor train_loss CON TOLERANCIA RELATIVA del 5% (evita que ruido del
        #      QDPSO decida — diferencias <5% son básicamente la misma calidad)
        #   3. menor ||θ||² (Occam: modelo más simple → mejor generalización)
        #   4. mayor val_acc
        #   5. menor val_mse
        val_score_diff = val_score - best_score
        if val_score_diff < -val_score_tol:
            is_better = True
        elif val_score_diff <= val_score_tol:
            # Tie en val_score → comparar train_loss con tolerancia relativa.
            train_threshold_low = best_train_loss * (1.0 - train_loss_rel_tol)
            train_threshold_high = best_train_loss * (1.0 + train_loss_rel_tol)
            if train_loss < train_threshold_low:
                is_better = True
            elif train_loss <= train_threshold_high:
                # Tie en train_loss (dentro de 5%) → comparar ||θ||² (Occam).
                if l2_norm_sq < best_l2_norm_sq:
                    is_better = True
                elif l2_norm_sq == best_l2_norm_sq:
                    # Tie en L2 también → val_acc, val_mse como fallback.
                    is_better = (val_acc > best_val_acc) or (
                        val_acc == best_val_acc and val_cost < best_val_cost
                    )
                else:
                    is_better = False
            else:
                is_better = False
        else:
            is_better = False

        if is_better:
            best_score = val_score
            best_train_loss = train_loss
            best_l2_norm_sq = l2_norm_sq
            best_val_acc = val_acc
            best_val_cost = val_cost
            best_idx = k
            best_params = params_k.clone()
            print(f"  ✨ nueva mejor sesión "
                  f"(score={val_score:.4f}, train={train_loss:.4f}, ||θ||²={l2_norm_sq:.2f}, ACC={val_acc:.4f})")

        # Cadena inter-sesión: SIEMPRE arrancamos desde el mejor global hasta ahora,
        # no desde el resultado inmediato. Una sesión peor no envenena la cadena.
        params_init = best_params.clone()

    total_time = time.time() - total_start

    # Medición final única sobre test (ganador elegido por val)
    final_test_cost, final_test_acc, y_test_pred = evaluate_model(
        mlp, best_params, data["X_test"], data["y_test"]
    )
    final_val_cost, final_val_acc, _ = evaluate_model(
        mlp, best_params, data["X_val"], data["y_val"]
    )

    cm = confusion_matrix(data["y_test"], y_test_pred, data["n_classes"])

    final_val_score = combined_score(final_val_acc, final_val_cost, mse_max)
    final_test_score = combined_score(final_test_acc, final_test_cost, mse_max)
    final_train_loss = evaluate_train_loss(
        mlp, best_params, data["X_train"], data["y_train"], cfg["lambda_l2"]
    )

    winner = {
        "session": int(best_idx),
        "val_score": float(final_val_score),
        "val_acc": float(final_val_acc),
        "val_cost": float(final_val_cost),
        "train_loss": float(final_train_loss),
        "final_test_score": float(final_test_score),
        "final_test_acc": float(final_test_acc),
        "final_test_cost": float(final_test_cost),
        "mse_max": float(mse_max),
        "test_confusion_matrix": cm,
    }

    # Persistencia
    output_dir.mkdir(parents=True, exist_ok=True)

    # Curvas de aprendizaje a través de las sesiones (para análisis posterior y plots)
    learning_curve = {
        "session_idx":       [s.session for s in sessions],
        "val_acc":           [s.val_acc for s in sessions],
        "val_mse":           [s.val_cost for s in sessions],
        "val_score":         [s.val_score for s in sessions],
        "test_acc":          [s.test_acc for s in sessions],
        "test_mse":          [s.test_cost for s in sessions],
        "test_score":        [s.test_score for s in sessions],
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
    print(f"🏆 Ganadora: sesión {best_idx}   (score combinado={best_score:.4f}, mse_max={mse_max})")
    print(f"   val  ACC={final_val_acc:.4f}  MSE={final_val_cost:.4f}  score={final_val_score:.4f}   (criterio de selección)")
    print(f"   test ACC={final_test_acc:.4f}  MSE={final_test_cost:.4f}  score={final_test_score:.4f}   (final, no sesgado)")
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

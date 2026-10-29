"""Selección de modelos por validación: regla de k errores estándar.

Es el criterio ÚNICO para elegir parámetros, dentro de las técnicas (T1, T2, T3) y
entre sesiones. Se configura con dos valores (`--selection-metric`, `--selection-se`):

1. Cada candidato se mide en validación: accuracy, balanced accuracy (media del
   recall por clase) y cross-entropy (CE), con un solo forward.
2. Métrica principal m ∈ {acc, bacc}. Sea m* la mejor m entre los candidatos y
   SE(m*) su error estándar (Agresti-Coull, ver `standard_error`).
3. Empatan todos los candidatos con m ≥ m* − k·SE(m*).
4. Entre los empatados gana el de MENOR CE de validación; si también empatan en CE,
   el que llegó primero.

Con `metric="acc"` y `k=0` es exactamente la regla histórica (más val_acc y, con la
misma, menos val_ce). Por qué así (detalle y referencias en la memoria del proyecto):
- La validación es chica (18 muestras en iris): diferencias menores que su error
  estándar son ruido, y quedarse con el máximo entre cientos de candidatos elige
  al que tuvo suerte (maldición del ganador). Regla de 1 SE: Breiman et al. (1984);
  Hastie, Tibshirani y Friedman, ESL §7.10.
- La CE es continua y es una regla de puntuación propia (Gneiting y Raftery, 2007):
  usa la confianza en todas las muestras, no sólo el acierto, así que es más
  estable para decidir entre empatados.
- La balanced accuracy no premia acertar sólo las clases grandes
  (Brodersen et al., 2010).

Memoria: un candidato se guarda sólo si puede llegar a ganar. Se descartan los que
quedan bajo el umbral más bajo posible (m_t − k·SE_max, con SE_max la cota de
p(1−p) ≤ 1/4) y los dominados (otro con m ≥ y CE ≤ que llegó antes). El resultado
es idéntico a guardar todos.
"""
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from core.budget import METER
from core.loss import cross_entropy_from_logits

#: Métricas principales de selección.
SELECTION_METRICS = ("acc", "bacc")

# Tolerancia numérica para comparar métricas que son fracciones de enteros.
_EPS = 1e-12


@dataclass(frozen=True)
class ValScore:
    """Métricas de validación de un candidato."""
    acc: float
    bacc: float
    ce: float
    recalls: tuple   # recall por clase presente en val (para el SE de bacc)

    def metric(self, name: str) -> float:
        return self.acc if name == "acc" else self.bacc


def val_score(model, flat_params: torch.Tensor, X: torch.Tensor, y: torch.Tensor,
              n_classes: int) -> ValScore:
    """Mide `flat_params` en (X, y) con un forward y lo registra en el contador de selección."""
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model.forward(X, flat_params)
        pred = logits.argmax(dim=-1)
        ce = cross_entropy_from_logits(logits, y, dim_classes=-1).mean()
        cm = torch.bincount(y * n_classes + pred, minlength=n_classes * n_classes)
        cm = cm.reshape(n_classes, n_classes)
        support = cm.sum(dim=1)
        hits = cm.diag()
        packed = torch.stack([hits.double(), support.double()]).cpu()   # una sola sincronización
        ce = float(ce.item())
    hits_l, support_l = packed[0].tolist(), packed[1].tolist()
    present = [(h, s) for h, s in zip(hits_l, support_l) if s > 0]
    recalls = tuple(h / s for h, s in present)
    n = sum(s for _, s in present)
    acc = sum(h for h, _ in present) / n
    bacc = sum(recalls) / len(recalls)
    METER.record_selection(acc, ce, time.perf_counter() - t0)
    return ValScore(acc=acc, bacc=bacc, ce=ce, recalls=recalls)


def class_supports(y: torch.Tensor, n_classes: int) -> List[int]:
    """Muestras por clase presente en `y` (en el orden de las clases)."""
    counts = torch.bincount(y, minlength=n_classes).tolist()
    return [int(c) for c in counts if c > 0]


def _ac_var(p: float, n: int) -> float:
    """Varianza de Agresti-Coull de una proporción p medida sobre n muestras.

    p̃ = (x + 2)/(n + 4), var = p̃(1 − p̃)/(n + 4). A diferencia de p(1−p)/n, no se
    anula cuando p = 0 o p = 1 (p. ej. val_acc = 1.0 en iris).
    """
    pt = (p * n + 2.0) / (n + 4.0)
    return pt * (1.0 - pt) / (n + 4.0)


def standard_error(metric: str, score: ValScore, supports: List[int]) -> float:
    """Error estándar de la métrica principal de `score`.

    - acc: proporción sobre n = Σ n_c muestras.
    - bacc: media de K recalls independientes, SE = √(Σ var_c)/K.
    """
    if metric == "acc":
        return math.sqrt(_ac_var(score.acc, sum(supports)))
    var = sum(_ac_var(r, n) for r, n in zip(score.recalls, supports))
    return math.sqrt(var) / len(supports)


def standard_error_max(metric: str, supports: List[int]) -> float:
    """Cota superior del error estándar (p̃(1 − p̃) ≤ 1/4), para descartar con seguridad."""
    if metric == "acc":
        return math.sqrt(0.25 / (sum(supports) + 4.0))
    return math.sqrt(sum(0.25 / (n + 4.0) for n in supports)) / len(supports)


@dataclass
class _Candidate:
    order: int
    score: ValScore
    payload: Any
    info: dict


class Selector:
    """Conjunto de candidatos y la regla de k errores estándar. Seguro entre hilos (T2)."""

    def __init__(self, metric: str, n_se: float, y_val: torch.Tensor, n_classes: int):
        if metric not in SELECTION_METRICS:
            raise ValueError(f"métrica de selección desconocida: {metric!r}")
        if n_se < 0:
            raise ValueError(f"selection_se debe ser >= 0 (se dio {n_se})")
        self.metric = metric
        self.n_se = float(n_se)
        self.supports = class_supports(y_val, n_classes)
        self._se_max = standard_error_max(metric, self.supports)
        self._pool: List[_Candidate] = []
        self._seen = 0
        self._best_m = -math.inf
        self._lock = threading.Lock()

    @classmethod
    def from_cfg(cls, cfg: dict, data: dict) -> "Selector":
        return cls(cfg.get("selection_metric", "acc"), cfg.get("selection_se", 0.0),
                   data["y_val"], data["n_classes"])

    @property
    def n_candidates(self) -> int:
        return self._seen

    @property
    def criterion(self) -> str:
        return criterion_name(self.metric, self.n_se)

    def offer(self, score: ValScore, payload: Any, info: Optional[dict] = None) -> None:
        """Agrega un candidato. `payload` es lo que devuelve `choose` si gana."""
        m = score.metric(self.metric)
        with self._lock:
            order = self._seen
            self._seen += 1
            self._best_m = max(self._best_m, m)
            floor = self._best_m - self.n_se * self._se_max - _EPS
            # Dominado por uno anterior: nunca puede ganar.
            for c in self._pool:
                if c.score.metric(self.metric) >= m - _EPS and c.score.ce <= score.ce:
                    self._prune(floor)
                    return
            # Saca los que el nuevo domina (m ≥ y CE estrictamente menor) y los que
            # ya no alcanzan el umbral más bajo posible.
            self._pool = [c for c in self._pool
                          if not (m >= c.score.metric(self.metric) - _EPS and score.ce < c.score.ce)]
            self._pool.append(_Candidate(order, score, payload, dict(info or {})))
            self._prune(floor)

    def _prune(self, floor: float) -> None:
        self._pool = [c for c in self._pool if c.score.metric(self.metric) >= floor]

    def choose(self):
        """Aplica la regla. Devuelve (payload, score, info) del ganador.

        `info` incluye el diagnóstico de la selección: m*, SE, umbral, cuántos
        candidatos empataron y cuántos se vieron.
        """
        with self._lock:
            if not self._pool:
                raise RuntimeError("Selector sin candidatos")
            ref = max(self._pool, key=lambda c: (c.score.metric(self.metric), -c.score.ce, -c.order))
            m_star = ref.score.metric(self.metric)
            se = standard_error(self.metric, ref.score, self.supports)
            threshold = m_star - self.n_se * se
            tied = [c for c in self._pool if c.score.metric(self.metric) >= threshold - _EPS]
            win = min(tied, key=lambda c: (c.score.ce, c.order))
            info = dict(win.info)
            info.update({
                "criterion": self.criterion,
                "best_metric": m_star,
                "se": se,
                "threshold": threshold,
                "n_tied_nondominated": len(tied),
                "n_candidates": self._seen,
            })
            return win.payload, win.score, info


def criterion_name(metric: str, n_se: float) -> str:
    """Nombre del criterio guardado en run.json. `val_acc>val_ce` es la regla histórica."""
    if n_se == 0:
        return f"val_{metric}>val_ce"
    return f"val_{metric}({n_se:g}SE)>val_ce"

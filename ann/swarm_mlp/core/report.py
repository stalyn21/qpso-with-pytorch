"""Datos y métricas para las tablas (`compare.py`) y las figuras (`visualize.py`).

Una sola definición de cada métrica, de cómo se identifica una configuración y de
cómo se comparan las técnicas, para que tablas y figuras digan lo mismo.

Unidad de dato: un FOLD de una corrida K-fold (`cv_summary.json` → `folds`). Una
corrida = (configuración, semilla); cada semilla aporta K folds.

Comparaciones pareadas: dos técnicas se comparan sobre las MISMAS (semilla, fold).
Por defecto se promedian los folds de cada semilla y se compara por semilla
(n = número de semillas), porque los folds de una semilla comparten la
inicialización y no son independientes. Los p-valores (Wilcoxon) se corrigen por
comparaciones múltiples con Holm.
"""
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.stats import describe, paired_comparison

TECHNIQUES_ORDER = ["t1", "t2", "t2-jacobi", "t3", "t4"]
DATASETS_ORDER = ["iris", "breast", "segment", "satimage", "digits"]


# --------------------------------------------------------------------------- métricas

@dataclass(frozen=True)
class Metric:
    key: str
    label: str            # nombre corto para ejes y columnas
    role: str             # reporte | diagnóstico | costo | selección
    higher_is_better: bool
    description: str      # para qué sirve (va a la leyenda de las tablas)
    getter: Callable[[Dict[str, Any]], Optional[float]]


def _f(name):
    return lambda fold: fold.get(name)


def _diff(a, b):
    def get(fold):
        x, y = fold.get(a), fold.get(b)
        return None if x is None or y is None else x - y
    return get


#: Registro de métricas por fold. El orden es el de las tablas.
METRICS: Dict[str, Metric] = {m.key: m for m in [
    Metric("test_bacc", "Balanced acc. (test)", "reporte", True,
           "Métrica principal: media del recall por clase en test; no premia acertar sólo las clases grandes.",
           _f("test_balanced_acc")),
    Metric("test_acc", "Accuracy (test)", "reporte", True,
           "Fracción de aciertos en test; para comparar con la literatura.",
           _f("test_acc")),
    Metric("test_f1", "Macro-F1 (test)", "reporte", True,
           "Media del F1 por clase en test; combina precisión y recall de cada clase.",
           _f("test_macro_f1")),
    Metric("test_ce", "Cross-entropy (test)", "reporte", False,
           "Calidad de las probabilidades (calibración); menor es mejor.",
           _f("test_ce")),
    Metric("val_bacc", "Balanced acc. (val)", "selección", True,
           "Métrica principal de selección (regla de 1 SE).",
           _f("val_balanced_acc")),
    Metric("val_acc", "Accuracy (val)", "selección", True,
           "Accuracy de validación; métrica de selección de la regla histórica.",
           _f("val_acc")),
    Metric("val_ce", "Cross-entropy (val)", "selección", False,
           "Desempate de la regla de selección.",
           _f("val_ce")),
    Metric("optimism_bacc", "Optimism val−test (bacc)", "diagnóstico", False,
           "Cuánto sobreestima la validación al test: sesgo de la selección (maldición del ganador).",
           _diff("val_balanced_acc", "test_balanced_acc")),
    Metric("optimism_acc", "Optimism val−test (acc)", "diagnóstico", False,
           "Igual que la anterior, con accuracy.",
           _diff("val_acc", "test_acc")),
    Metric("overfit_gap", "Gap train−test (acc)", "diagnóstico", False,
           "Sobreajuste: cuánto mejor le va en train que en test.",
           _diff("train_acc", "test_acc")),
    Metric("time", "Time per fold (s)", "costo", False,
           "Tiempo de pared de la corrida de un fold (GPU sincronizada).",
           _f("total_time")),
    Metric("evals", "Fitness evaluations", "costo", False,
           "Evaluaciones de la función de costo (una partícula sobre todo el train).",
           _f("fitness_evals")),
    Metric("gpu_mb", "Peak GPU memory (MB)", "costo", False,
           "Memoria pico de GPU.",
           _f("gpu_peak_mb")),
]}

#: Métrica principal del paper 2.
PRIMARY_METRIC = "test_bacc"


# ----------------------------------------------------------------- configuración

#: Campos de `config` que identifican un experimento (todo menos la semilla).
#: Los resúmenes viejos no traen los campos nuevos: se usa el valor que tenía
#: implícitamente el código de entonces.
CONFIG_FIELDS: Dict[str, Any] = {
    "technique": None,
    "dataset": None,
    "layer_sizes": None,
    "n_particles": None,
    "max_iter": None,
    "eval_budget": None,
    "n_rounds": 1,
    "n_sessions_per_fold": 1,
    "cv_folds": None,
    "optimizer": "qdpsoo",
    "g": None,
    "alpha": None,
    "lambda_l2": None,
    "l2_form": "mean",
    "reevaluate_pbest": False,
    "selection_metric": "acc",
    "selection_se": 0.0,
    "data_seed": 100,
}


#: Campos derivados: combinan varios campos en uno, para filtrar o contrastar.
#: `selection_rule` = métrica:k (p. ej. "bacc:1se"); contrastarlo compara reglas
#: que difieren a la vez en la métrica y en la tolerancia.
DERIVED_FIELDS = {"selection_rule": ("selection_metric", "selection_se")}


def _hashable(v):
    return tuple(v) if isinstance(v, list) else v


def config_of(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Campos de identidad de un `config`, con los valores por defecto de los resúmenes viejos."""
    out = {k: _hashable(cfg.get(k, default)) for k, default in CONFIG_FIELDS.items()}
    # El factor de contracción-expansión que aplica según el optimizador.
    if out["optimizer"] == "qpsoo":
        out["g"] = None
    else:
        out["alpha"] = None
    out["selection_rule"] = f"{out['selection_metric']}:{out['selection_se']:g}se"
    return out


def selection_label(metric: str, se: float) -> str:
    """Nombre corto de la regla de selección para leyendas."""
    if metric == "acc" and not se:
        return "max val acc (v1)"
    return f"val {metric}, {se:g} SE" + (" (v2)" if metric == "bacc" and se == 1 else "")


# ----------------------------------------------------------------------- carga

@dataclass
class FoldRow:
    config: Dict[str, Any]
    seed: int
    fold: int
    values: Dict[str, Optional[float]]
    run_path: Optional[Path]      # fold*/run.json (para trazas de convergencia)

    def get(self, metric: str) -> Optional[float]:
        return self.values.get(metric)


def load_folds(output_dir: Path) -> List[FoldRow]:
    """Una fila por fold de cada `cv_summary.json` bajo `output_dir`."""
    rows = []
    for path in sorted(Path(output_dir).rglob("cv_summary.json")):
        with open(path) as f:
            summary = json.load(f)
        cfg = summary["config"]
        ident = config_of(cfg)
        for fold in summary["folds"]:
            idx = int(fold["fold_idx"])
            run = path.parent / f"fold{idx}" / "run.json"
            rows.append(FoldRow(
                config=ident, seed=int(cfg["seed"]), fold=idx,
                values={k: m.getter(fold) for k, m in METRICS.items()},
                run_path=run if run.exists() else None,
            ))
    return rows


def parse_filters(items: Sequence[str]) -> Dict[str, str]:
    """`["selection_se=1", "dataset=iris"]` → dict. Se comparan como texto."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"filtro inválido {it!r}; se espera clave=valor")
        k, v = it.split("=", 1)
        if k not in CONFIG_FIELDS and k not in DERIVED_FIELDS:
            raise ValueError(f"campo desconocido {k!r}; válidos: "
                             f"{', '.join(list(CONFIG_FIELDS) + list(DERIVED_FIELDS))}")
        out[k] = v
    return out


def _as_text(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, tuple):
        return "-".join(_as_text(x) for x in v)
    return str(v)


def apply_filters(rows: Iterable[FoldRow], filters: Dict[str, str]) -> List[FoldRow]:
    def ok(r):
        for k, v in filters.items():
            got = r.config.get(k)
            if _as_text(got) != v and str(got) != v:
                try:
                    if float(got) == float(v):
                        continue
                except (TypeError, ValueError):
                    pass
                return False
        return True
    return [r for r in rows if ok(r)]


def config_key(cfg: Dict[str, Any], exclude: Sequence[str] = ()) -> Tuple:
    return tuple((k, cfg[k]) for k in CONFIG_FIELDS if k not in exclude)


def varying_fields(rows: Sequence[FoldRow], exclude: Sequence[str] = ("technique", "dataset", "layer_sizes")) -> Dict[str, list]:
    """Campos de configuración que toman más de un valor entre `rows`.

    Excluir un campo derivado excluye también sus componentes, y los derivados
    nunca se reportan (sus componentes ya lo hacen).
    """
    exclude = set(exclude)
    for d, parts in DERIVED_FIELDS.items():
        if d in exclude:
            exclude.update(parts)
    exclude.update(DERIVED_FIELDS)
    vals = defaultdict(set)
    for r in rows:
        for k, v in r.config.items():
            if k not in exclude:
                vals[k].add(v)
    return {k: sorted(v, key=lambda x: (x is None, str(x))) for k, v in vals.items() if len(v) > 1}


def group(rows: Iterable[FoldRow], by: Sequence[str]) -> Dict[Tuple, List[FoldRow]]:
    out = defaultdict(list)
    for r in rows:
        out[tuple(r.config[k] for k in by)].append(r)
    return out


def ordered(values: Iterable, order: Sequence) -> list:
    vals = list(values)
    return [v for v in order if v in vals] + sorted(v for v in vals if v not in order)


# ------------------------------------------------------------------- estadística

def values_of(rows: Sequence[FoldRow], metric: str) -> List[float]:
    return [r.get(metric) for r in rows if r.get(metric) is not None]


def seed_means(rows: Sequence[FoldRow], metric: str) -> Dict[int, float]:
    by = defaultdict(list)
    for r in rows:
        if r.get(metric) is not None:
            by[r.seed].append(r.get(metric))
    return {s: float(np.mean(v)) for s, v in by.items()}


def summarize(rows: Sequence[FoldRow], metric: str) -> Dict[str, Any]:
    """`describe` sobre los folds + desviación entre semillas (reproducibilidad)."""
    d = describe(values_of(rows, metric))
    sm = list(seed_means(rows, metric).values())
    d["n_seeds"] = len(sm)
    d["seed_std"] = float(np.std(sm, ddof=1)) if len(sm) > 1 else None
    return d


def paired(rows_a: Sequence[FoldRow], rows_b: Sequence[FoldRow], metric: str,
           unit: str = "seed") -> Optional[Dict[str, Any]]:
    """Compara b − a sobre las unidades comunes (`seed`: promedio de folds; `fold`: cada fold)."""
    if unit == "seed":
        a, b = seed_means(rows_a, metric), seed_means(rows_b, metric)
    else:
        a = {(r.seed, r.fold): r.get(metric) for r in rows_a if r.get(metric) is not None}
        b = {(r.seed, r.fold): r.get(metric) for r in rows_b if r.get(metric) is not None}
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    return paired_comparison([a[k] for k in common], [b[k] for k in common])


def holm(pvalues: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Corrección de Holm-Bonferroni (controla el error de familia). None se respeta."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    order = sorted(idx, key=lambda i: pvalues[i])
    m = len(order)
    adj = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj

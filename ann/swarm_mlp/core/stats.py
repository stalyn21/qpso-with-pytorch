"""Estadística descriptiva de un conjunto de corridas (semillas × splits).

Una sola definición para el runner, `compare.py` y las tablas del paper, de modo
que "media ± std" signifique lo mismo en todos lados:

- `std` es la desviación estándar MUESTRAL (n-1), no la poblacional.
- `ci95` es el intervalo de confianza del 95% de la media (t de Student).
- mediana, cuartiles, mínimo y máximo describen la estabilidad sin suponer
  normalidad; `min` es el peor caso observado.
"""
import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy import stats as _st


def describe(values: Sequence[float]) -> Dict[str, Any]:
    """Resumen de `values`. Con n < 2 la dispersión no está definida y queda en None."""
    x = np.asarray([v for v in values if v is not None], dtype=float)
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None, "median": None,
                "q1": None, "q3": None, "min": None, "max": None}
    mean = float(x.mean())
    std: Optional[float] = None
    ci95: Optional[float] = None
    if n > 1:
        std = float(x.std(ddof=1))
        ci95 = float(_st.t.ppf(0.975, n - 1) * std / math.sqrt(n))
    q1, median, q3 = (float(q) for q in np.percentile(x, [25, 50, 75]))
    return {"n": n, "mean": mean, "std": std, "ci95": ci95, "median": median,
            "q1": q1, "q3": q3, "min": float(x.min()), "max": float(x.max())}


def failure_rate(values: Sequence[float], threshold: float) -> Optional[float]:
    """Fracción de corridas con valor < `threshold` (p. ej. accuracy por debajo de un piso).

    Mide lo que la media esconde: cada cuánto el método entrega un modelo malo.
    """
    x = [v for v in values if v is not None]
    return (sum(v < threshold for v in x) / len(x)) if x else None


def paired_comparison(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Compara dos métodos corridos sobre las MISMAS (semilla, split), en el mismo orden.

    Devuelve la diferencia media `b - a`, su IC95 y el p-valor de Wilcoxon de
    rangos con signo (no supone normalidad). Con todas las diferencias en cero
    el test no está definido y `p_wilcoxon` queda en None.
    """
    if len(a) != len(b):
        raise ValueError(f"Las muestras pareadas deben tener el mismo largo ({len(a)} vs {len(b)})")
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    out = {"n": int(d.size), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
           "losses": int((d < 0).sum()), "p_wilcoxon": None}
    out.update({k: v for k, v in describe(d).items() if k in ("mean", "std", "ci95")})
    if d.size > 1 and np.any(d != 0):
        out["p_wilcoxon"] = float(_st.wilcoxon(d, zero_method="wilcox").pvalue)
    return out

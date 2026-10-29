"""Figuras del paper 2 a partir de los `cv_summary.json` (y los `run.json` por fold).

Las métricas, la identidad de una configuración y las comparaciones pareadas vienen
de `core/report.py`, las mismas que usa `compare.py --paper`: tablas y figuras dicen
lo mismo.

Figuras (`--figures`, por defecto todas las que los datos permitan):

  box          Distribución de la métrica principal (cada punto = un fold) por técnica.
  forest       Diferencias pareadas entre técnicas (Δ ± IC95 por semilla, Holm).
  pareto       Costo (tiempo por fold) contra la métrica principal.
  contrast     Dos valores de un campo (--contrast, p. ej. selection_rule) por técnica:
               métrica de test y optimismo val−test.
  convergence  Mejor valor de validación visto contra evaluaciones (o tiempo), desde
               las trazas de cada fold: mediana e IQR.
  radar        Resumen normalizado [0, 1]: balanced acc., estabilidad,
               reproducibilidad, velocidad y honestidad de la validación.
  sweep        Barrido de un campo (--sweep, p. ej. g) con líneas por técnica y,
               opcionalmente, estilo de línea por un segundo campo (--sweep-style).

Las corridas filtradas deben ser UNA configuración por (técnica, dataset), salvo el
campo de --contrast o --sweep; si no, el script dice qué campos varían.

Uso (desde ann/swarm_mlp):
    # Resultados finales con la regla v2 y el contraste de reglas
    python visualize.py --output-dir output/paper2-final-ds200 --filter selection_rule=bacc:1se
    python visualize.py --output-dir output/paper2-final-ds200 --contrast selection_rule \\
        --figures contrast --figs-dir figs/selection
    # Barridos de sensibilidad (seleccionados con la regla v1: métrica de val)
    python visualize.py --output-dir output/paper2-1s --figures sweep --sweep lambda_l2 \\
        --filter l2_form=sample n_particles=50 --sweep-metric val_acc
    python visualize.py --output-dir output/paper2-1s --figures sweep --sweep g \\
        --sweep-style max_iter --filter n_particles=75 lambda_l2=0 --sweep-metric val_acc
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches
import matplotlib.ticker

from core.report import (
    DATASETS_ORDER, METRICS, PRIMARY_METRIC, TECHNIQUES_ORDER, FoldRow, apply_filters,
    group, holm, load_folds, ordered, paired, parse_filters, seed_means, selection_label,
    summarize, values_of, varying_fields,
)

# ----------------------------------------------------------------------- estilo
#
# Paleta categórica validada (dataviz, scripts/validate_palette.js, --pairs all,
# superficie clara): las tres primeras ranuras de la paleta de referencia pasan
# todas las verificaciones. El aqua (T3) queda bajo 3:1 de contraste: por eso cada
# técnica lleva además su propio marcador y las figuras rotulan directo.
TECHNIQUE_COLORS = {"t1": "#2a78d6", "t2": "#eb6834", "t3": "#1baf7a",
                    "t2-jacobi": "#6250d6", "t4": "#e87ba4"}
TECHNIQUE_MARKERS = {"t1": "o", "t2": "s", "t3": "^", "t2-jacobi": "D", "t4": "v"}
TECHNIQUE_LABELS = {"t1": "T1 (sequential)", "t2": "T2 (parallel, async)",
                    "t3": "T3 (concurrent)", "t2-jacobi": "T2-jacobi", "t4": "T4 (single swarm)"}
DATASET_LABELS = {"iris": "Iris", "breast": "Breast Cancer", "segment": "Segment",
                  "satimage": "Satimage", "digits": "Digits"}

INK = "#0b0b0b"          # texto principal
INK_2 = "#52514e"        # texto secundario
INK_3 = "#8a8984"        # texto tenue
GRID = "#e4e3df"         # rejilla: un paso sobre la superficie
SURFACE = "#ffffff"

#: Estilos de línea para el segundo campo de un barrido (codificación secundaria).
LINESTYLES = ["-", "--", ":", "-."]


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK_3,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "legend.frameon": False, "figure.titlesize": 13, "figure.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round", "lines.markersize": 7,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def _pct(metric: str) -> float:
    """Escala de la métrica en el eje: puntos porcentuales para accuracies y optimismo."""
    return 1.0 if metric.endswith("_ce") or metric in ("time", "evals", "gpu_mb") else 100.0


def _axis_label(metric: str) -> str:
    m = METRICS[metric]
    return m.label + (" (%)" if _pct(metric) == 100 and not metric.startswith(("optimism", "overfit"))
                      else " (pts)" if _pct(metric) == 100 else "")


def _save(fig, path: Path, formats: Sequence[str]) -> List[Path]:
    out = []
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        p = path.with_suffix("." + ext)
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out


def _titles(fig, title: str, subtitle: str = "", gap_in: float = 0.0, bottom: float = 0.0):
    """Título y subtítulo arriba sin chocar con los paneles.

    Los márgenes se calculan en pulgadas (no dependen del alto de la figura):
    0.32" de título, 0.17" por línea de subtítulo y `gap_in` extra (p. ej. una
    leyenda arriba o los rótulos polares del radar).
    """
    h = fig.get_figheight()
    lines = subtitle.count("\n") + 1 if subtitle else 0
    header = 0.42 + 0.17 * lines + gap_in
    fig.tight_layout(rect=(0, bottom, 1, 1 - header / h))
    fig.suptitle(title, y=1 - 0.06 / h, va="top")
    if subtitle:
        fig.text(0.5, 1 - 0.36 / h, subtitle, ha="center", va="top", fontsize=8.5, color=INK_2)


def _grid(n: int, panel=(3.6, 3.2)):
    ncols = min(n, 5)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def _tech_legend(fig, techs, y=-0.02, extra=()):
    handles = [Line2D([], [], color=TECHNIQUE_COLORS[t], marker=TECHNIQUE_MARKERS[t], lw=2,
                      markersize=7, markeredgecolor=SURFACE, label=TECHNIQUE_LABELS[t])
               for t in techs] + list(extra)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=len(handles), frameon=False)


def _subtitle(rows: Sequence[FoldRow]) -> str:
    c = rows[0].config
    rule = selection_label(c["selection_metric"], c["selection_se"])
    return (f"P = {c['n_particles']}, I = {c['max_iter']} per layer, g = {c['g']}, "
            f"λ = {c['lambda_l2']:g} ({c['l2_form']}), selection: {rule}, data seed {c['data_seed']}")


# ------------------------------------------------------------------------ figuras

def fig_box(rows, metric, formats, out):
    """Distribución por fold: caja (IQR + mediana), puntos de cada fold y media."""
    by = group(rows, ("dataset", "technique"))
    dss = ordered({k[0] for k in by}, DATASETS_ORDER)
    techs = ordered({k[1] for k in by}, TECHNIQUES_ORDER)
    fig, axes = _grid(len(dss))
    sc = _pct(metric)
    rng = np.random.default_rng(0)
    for ax, ds in zip(axes, dss):
        for i, t in enumerate(techs):
            v = np.array(values_of(by.get((ds, t), []), metric)) * sc
            if not v.size:
                continue
            col = TECHNIQUE_COLORS[t]
            ax.boxplot([v], positions=[i], widths=0.5, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=col, alpha=0.12, edgecolor=col, linewidth=1.2),
                       medianprops=dict(color=INK, linewidth=1.6),
                       whiskerprops=dict(color=col, linewidth=1.2), capprops=dict(color=col, linewidth=1.2))
            ax.scatter(i + rng.uniform(-0.17, 0.17, v.size), v, s=12, color=col, alpha=0.55,
                       marker=TECHNIQUE_MARKERS[t], linewidths=0, zorder=3)
            ax.scatter([i], [v.mean()], s=46, color=SURFACE, edgecolor=INK, linewidths=1.4,
                       marker="D", zorder=4)
            ax.annotate(f"{v.mean():.1f}", (i, v.max()), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_xticks(range(len(techs)), [t.upper() for t in techs])
        ax.set_xlim(-0.6, len(techs) - 0.4)
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)
    axes[0].set_ylabel(_axis_label(metric))
    mean_key = Line2D([], [], color=SURFACE, marker="D", markeredgecolor=INK, markersize=7,
                      lw=0, label="mean")
    _tech_legend(fig, techs, extra=[mean_key])
    _titles(fig, f"{METRICS[metric].label} by technique (each point = one fold; label = mean)",
            _subtitle(rows))
    return _save(fig, out / f"box_{metric}", formats)


def _pairs_table(rows, metric, pairs, unit):
    by = group(rows, ("dataset", "technique"))
    res = []
    for ds in ordered({k[0] for k in by}, DATASETS_ORDER):
        for a, b in pairs:
            if (ds, a) in by and (ds, b) in by:
                c = paired(by[(ds, a)], by[(ds, b)], metric, unit)
                if c:
                    res.append(dict(dataset=ds, a=a, b=b, **c))
    for r, ph in zip(res, holm([r["p_wilcoxon"] for r in res])):
        r["p_holm"] = ph
    return res


def fig_forest(rows, metric, pairs, unit, alpha, formats, out):
    """Δ = b − a por dataset con IC95; relleno = significativo tras Holm."""
    res = _pairs_table(rows, metric, pairs, unit)
    if not res:
        print("⚠  forest: no hay pares de técnicas comparables.")
        return []
    sc = _pct(metric)
    dss = ordered({r["dataset"] for r in res}, DATASETS_ORDER)
    shapes = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(res) + 2.2))
    y, yt, yl = 0, [], []
    for ds in dss:
        for j, (a, b) in enumerate(pairs):
            r = next((x for x in res if x["dataset"] == ds and x["a"] == a and x["b"] == b), None)
            if r is None:
                continue
            m, ci = r["mean"] * sc, (r["ci95"] or 0) * sc
            sig = r["p_holm"] is not None and r["p_holm"] < alpha
            ax.errorbar(m, y, xerr=ci, fmt="none", ecolor=INK_2, elinewidth=1.4, capsize=3)
            ax.scatter([m], [y], s=52, marker=shapes[j % len(shapes)], zorder=3,
                       facecolor=INK if sig else SURFACE, edgecolor=INK, linewidths=1.4)
            p = r["p_holm"]
            ax.annotate(f"{m:+.1f}  (p$_{{Holm}}$ {'<0.001' if p is not None and p < 0.001 else f'{p:.3f}' if p is not None else '—'})",
                        (1.0, y), xycoords=("axes fraction", "data"), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK_2)
            yt.append(y)
            yl.append(f"{DATASET_LABELS.get(ds, ds)}: {b.upper()} − {a.upper()}")
            y -= 1
        y -= 0.6
    ax.axvline(0, color=INK_3, lw=1)
    ax.set_yticks(yt, yl)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Δ {METRICS[metric].label} ({'pts' if sc == 100 else ''})  ← a better · b better →")
    handles = [Line2D([], [], marker="o", lw=0, markerfacecolor=INK, markeredgecolor=INK,
                      label=f"significant (Holm, α = {alpha:g})"),
               Line2D([], [], marker="o", lw=0, markerfacecolor=SURFACE, markeredgecolor=INK,
                      label="not significant")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    _titles(fig, f"Paired differences in {METRICS[metric].label}",
            f"Δ = b − a per {'seed (mean of folds)' if unit == 'seed' else 'fold'}: mean ± 95% t-interval. "
            f"Fill = Wilcoxon signed-rank significant after Holm over the figure (rank test; "
            f"it can\ndisagree with the t-interval when one seed is an outlier). {_subtitle(rows)}",
            gap_in=0.0)
    return _save(fig, out / f"forest_{metric}", formats)


def fig_pareto(rows, metric, cost, formats, out):
    """Costo medio contra la métrica: media ± IC95 (vertical) y ± std (horizontal)."""
    by = group(rows, ("dataset", "technique"))
    dss = ordered({k[0] for k in by}, DATASETS_ORDER)
    techs = ordered({k[1] for k in by}, TECHNIQUES_ORDER)
    fig, axes = _grid(len(dss))
    sc = _pct(metric)
    for ax, ds in zip(axes, dss):
        pts = []
        for t in techs:
            g = by.get((ds, t))
            if not g:
                continue
            ym, xm = summarize(g, metric), summarize(g, cost)
            x, y = xm["mean"], ym["mean"] * sc
            ax.errorbar(x, y, yerr=(ym["ci95"] or 0) * sc, xerr=xm["std"] or 0, fmt="none",
                        ecolor=TECHNIQUE_COLORS[t], elinewidth=1.2, capsize=2.5, alpha=0.8)
            ax.scatter([x], [y], s=64, color=TECHNIQUE_COLORS[t], marker=TECHNIQUE_MARKERS[t],
                       edgecolor=SURFACE, linewidths=2, zorder=3)
            ax.annotate(t.upper(), (x, y), xytext=(7, 5), textcoords="offset points",
                        fontsize=8.5, color=INK_2)
            pts.append((x, y))
        # Frontera de Pareto (menor costo, mayor métrica).
        front, best = [], -np.inf
        for x, y in sorted(pts):
            if y > best:
                front.append((x, y))
                best = y
        if len(front) > 1:
            ax.plot(*zip(*front), color=INK_3, lw=1, ls="--", zorder=1)
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel(METRICS[cost].label)
        ax.margins(x=0.25, y=0.25)
    axes[0].set_ylabel(_axis_label(metric))
    _tech_legend(fig, techs, extra=[Line2D([], [], color=INK_3, lw=1, ls="--", label="Pareto front")])
    _titles(fig, f"Cost vs {METRICS[metric].label} (mean ± 95% CI; cost ± std)", _subtitle(rows))
    return _save(fig, out / f"pareto_{metric}_vs_{cost}", formats)


def fig_contrast(rows, field, metrics, unit, alpha, formats, out):
    """Pesa (dumbbell) entre los dos valores de `field`: hueco = primero, lleno = segundo."""
    levels = sorted({r.config[field] for r in rows}, key=str)
    if len(levels) != 2:
        print(f"⚠  contrast: {field} necesita 2 valores; hay {levels}.")
        return []
    lo, hi = levels
    by = group(rows, ("dataset", "technique", field))
    dss = ordered({k[0] for k in by}, DATASETS_ORDER)
    techs = ordered({k[1] for k in by}, TECHNIQUES_ORDER)
    # Holm sobre todas las pruebas de la figura.
    tests = []
    for mt in metrics:
        for ds in dss:
            for t in techs:
                if (ds, t, lo) in by and (ds, t, hi) in by:
                    c = paired(by[(ds, t, lo)], by[(ds, t, hi)], mt, unit)
                    if c:
                        tests.append(((mt, ds, t), c["p_wilcoxon"]))
    padj = dict(zip([k for k, _ in tests], holm([p for _, p in tests])))
    fig, axes = plt.subplots(len(metrics), len(dss), figsize=(3.3 * len(dss), 3.0 * len(metrics)),
                             squeeze=False)
    for i, mt in enumerate(metrics):
        sc = _pct(mt)
        for j, ds in enumerate(dss):
            ax = axes[i, j]
            for x, t in enumerate(techs):
                if (ds, t, lo) not in by or (ds, t, hi) not in by:
                    continue
                a = summarize(by[(ds, t, lo)], mt)["mean"] * sc
                b = summarize(by[(ds, t, hi)], mt)["mean"] * sc
                col = TECHNIQUE_COLORS[t]
                ax.plot([x, x], [a, b], color=col, lw=2, zorder=2)
                ax.scatter([x], [a], s=48, facecolor=SURFACE, edgecolor=col, linewidths=1.8,
                           marker=TECHNIQUE_MARKERS[t], zorder=3)
                ax.scatter([x], [b], s=48, color=col, edgecolor=SURFACE, linewidths=1.5,
                           marker=TECHNIQUE_MARKERS[t], zorder=4)
                p = padj.get((mt, ds, t))
                if p is not None and p < alpha:
                    ax.annotate("*", (x, max(a, b)), xytext=(0, 4), textcoords="offset points",
                                ha="center", fontsize=12, color=INK)
            ax.set_xticks(range(len(techs)), [t.upper() for t in techs])
            ax.set_xlim(-0.6, len(techs) - 0.4)
            ax.grid(axis="x", visible=False)
            if mt.startswith("optimism"):
                ax.axhline(0, color=INK_3, lw=1)
            if i == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds))
            if j == 0:
                ax.set_ylabel(_axis_label(mt))
            ax.margins(y=0.25)
    lab = lambda v: selection_label(*v.split(":")[0:1], float(v.split(":")[1][:-2])) \
        if field == "selection_rule" else f"{field} = {v}"
    extra = [Line2D([], [], marker="o", lw=0, markerfacecolor=SURFACE, markeredgecolor=INK_2, label=lab(lo)),
             Line2D([], [], marker="o", lw=0, markerfacecolor=INK_2, markeredgecolor=INK_2, label=lab(hi)),
             Line2D([], [], lw=0, marker="$*$", color=INK, markersize=9,
                    label=f"significant (Holm, α = {alpha:g})")]
    _tech_legend(fig, techs, extra=extra)
    sub = [r for r in rows if r.config[field] == hi]
    _titles(fig, f"Effect of {field}: {lab(lo)} → {lab(hi)}",
            f"Means over seeds × folds; * = paired Wilcoxon per {unit}, Holm over the figure. "
            + _subtitle(sub).split(", selection")[0])
    return _save(fig, out / f"contrast_{field}", formats)


def _trace(run_path: Path, x_col: int, y_col: int, best_is_max: bool):
    with open(run_path) as f:
        tr = json.load(f)["sessions"][0]["trace"]
    if not tr:
        return None
    a = np.asarray(tr, dtype=float)
    x, y = a[:, x_col], a[:, y_col]
    y = np.maximum.accumulate(y) if best_is_max else np.minimum.accumulate(y)
    return x, y


def fig_convergence(rows, x_axis, y_metric, formats, out, n_grid=120):
    """Mejor valor de validación visto hasta x (evaluaciones o segundos): mediana e IQR."""
    # Columnas de la traza (core/budget.py:TRACE_COLUMNS): evals, seconds, val_acc, val_ce.
    x_col = {"evals": 0, "time": 1}[x_axis]
    y_col, best_max, sc = {"acc": (2, True, 100.0), "ce": (3, False, 1.0)}[y_metric]
    by = group([r for r in rows if r.run_path], ("dataset", "technique"))
    if not by:
        print("⚠  convergence: no hay run.json por fold.")
        return []
    dss = ordered({k[0] for k in by}, DATASETS_ORDER)
    techs = ordered({k[1] for k in by}, TECHNIQUES_ORDER)
    fig, axes = _grid(len(dss))
    for ax, ds in zip(axes, dss):
        traces = {t: [tr for r in by.get((ds, t), []) if (tr := _trace(r.run_path, x_col, y_col, best_max))]
                  for t in techs}
        xmax = max((tr[0][-1] for ts in traces.values() for tr in ts), default=0)
        if xmax <= 0:
            continue
        grid = np.linspace(0, xmax, n_grid)
        for t, ts in traces.items():
            if not ts:
                continue
            # Valor vigente en cada punto de la grilla (escalón); NaN después del final.
            mat = np.full((len(ts), n_grid), np.nan)
            for k, (x, y) in enumerate(ts):
                idx = np.searchsorted(x, grid, side="right") - 1
                ok = (idx >= 0) & (grid <= x[-1])
                mat[k, ok] = y[idx[ok]] * sc
            valid = np.sum(~np.isnan(mat), axis=0) >= max(1, len(ts) // 2)
            mat[:, ~valid] = 0.0   # columnas sin datos suficientes: se enmascaran abajo
            med = np.where(valid, np.nanmedian(mat, axis=0), np.nan)
            q1, q3 = (np.where(valid, np.nanpercentile(mat, q, axis=0), np.nan) for q in (25, 75))
            col = TECHNIQUE_COLORS[t]
            ax.fill_between(grid, q1, q3, color=col, alpha=0.12, linewidth=0)
            ax.plot(grid, med, color=col, lw=2)
            last = np.where(~np.isnan(med))[0]
            if last.size:
                e = last[-1]
                ax.scatter([grid[e]], [med[e]], s=40, color=col, marker=TECHNIQUE_MARKERS[t],
                           edgecolor=SURFACE, linewidths=1.5, zorder=3)
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel("Fitness evaluations" if x_axis == "evals" else "Wall-clock time (s)")
        if x_axis == "evals":
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v / 1000:.0f}K" if v else "0"))
    axes[0].set_ylabel(f"Best validation {'accuracy (%)' if y_metric == 'acc' else 'cross-entropy'} so far")
    band = matplotlib.patches.Patch(color=INK_3, alpha=0.25, label="IQR across seeds × folds")
    _tech_legend(fig, techs, extra=[Line2D([], [], color=INK_2, lw=2, label="median"), band])
    _titles(fig, f"Convergence: best validation {y_metric} seen vs "
                 f"{'evaluations' if x_axis == 'evals' else 'time'}",
            _subtitle(rows) + ". Curve = best candidate checked so far, not the selected model.")
    return _save(fig, out / f"convergence_{y_metric}_vs_{x_axis}", formats)


RADAR_AXES = ["Balanced acc.", "Stability", "Reproducibility", "Speed", "Honest validation"]


def fig_radar(rows, metric, caps, formats, out):
    """Resumen normalizado por dataset (1 = ideal). Las cotas son parámetros (`--radar-caps`)."""
    std_cap, seed_cap, opt_cap = caps
    by = group(rows, ("dataset", "technique"))
    dss = ordered({k[0] for k in by}, DATASETS_ORDER)
    techs = ordered({k[1] for k in by}, TECHNIQUES_ORDER)
    ang = np.linspace(0, 2 * np.pi, len(RADAR_AXES), endpoint=False).tolist()
    ang += ang[:1]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.9 * len(dss), 4.2), subplot_kw=dict(polar=True),
                             squeeze=False)
    for ax, ds in zip(axes[0], dss):
        tmin = min(summarize(by[(ds, t)], "time")["mean"] for t in techs if (ds, t) in by)
        for t in techs:
            g = by.get((ds, t))
            if not g:
                continue
            s = summarize(g, metric)
            opt = summarize(g, "optimism_bacc" if metric.endswith("bacc") else "optimism_acc")["mean"] or 0
            vals = [s["mean"],
                    1 - min((s["std"] or 0) / std_cap, 1),
                    1 - min((s["seed_std"] or 0) / seed_cap, 1),
                    tmin / summarize(g, "time")["mean"],
                    1 - min(abs(opt) / opt_cap, 1)]
            vals += vals[:1]
            col = TECHNIQUE_COLORS[t]
            ax.plot(ang, vals, color=col, lw=2, marker=TECHNIQUE_MARKERS[t], markersize=6,
                    markeredgecolor=SURFACE)
            ax.fill(ang, vals, color=col, alpha=0.05)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(ang[:-1], RADAR_AXES, fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["", "0.5", "", "1"], fontsize=7, color=INK_3)
        ax.set_title(DATASET_LABELS.get(ds, ds), pad=30)
    _tech_legend(fig, techs, y=0.06)
    _titles(fig, "Summary per dataset (normalized, 1 = ideal)", f"Balanced acc. = test mean; Stability = 1 − std/{std_cap:g}; "
             f"Reproducibility = 1 − std of seed means/{seed_cap:g}; Speed = fastest time / time; "
             f"Honest validation = 1 − |val − test|/{opt_cap:g}", gap_in=0.15, bottom=0.08)
    return _save(fig, out / f"radar_{metric}", formats)


def fig_sweep(rows, field, style_field, metric, formats, out):
    """Métrica media ± IC95 (sobre semillas) contra los valores de `field`."""
    styles = [None]
    if style_field:
        styles = sorted({r.config[style_field] for r in rows}, key=lambda v: (v is None, v))
    by = group(rows, ("dataset", "technique", field) + ((style_field,) if style_field else ()))
    dss = ordered({k[0] for k in by}, DATASETS_ORDER)
    techs = ordered({k[1] for k in by}, TECHNIQUES_ORDER)
    xs = sorted({k[2] for k in by})
    numeric = all(isinstance(x, (int, float)) for x in xs)
    # λ incluye el 0: eje simbólico (log con 0 no existe).
    pos = (lambda v: xs.index(v)) if (not numeric or (field == "lambda_l2")) else (lambda v: v)
    fig, axes = _grid(len(dss))
    sc = _pct(metric)
    for ax, ds in zip(axes, dss):
        for t in techs:
            for si, st in enumerate(styles):
                pts = []
                for x in xs:
                    key = (ds, t, x) + ((st,) if style_field else ())
                    if key in by:
                        sm = list(seed_means(by[key], metric).values())
                        if sm:
                            m = np.mean(sm) * sc
                            ci = (1.96 * np.std(sm, ddof=1) / np.sqrt(len(sm)) * sc) if len(sm) > 1 else 0
                            pts.append((pos(x), m, ci))
                if not pts:
                    continue
                px, pm, pc = map(np.array, zip(*pts))
                col = TECHNIQUE_COLORS[t]
                ax.fill_between(px, pm - pc, pm + pc, color=col, alpha=0.10, linewidth=0)
                ax.plot(px, pm, color=col, ls=LINESTYLES[si % len(LINESTYLES)], lw=2,
                        marker=TECHNIQUE_MARKERS[t], markersize=6, markeredgecolor=SURFACE)
                b = int(np.argmax(pm))
                ax.scatter([px[b]], [pm[b]], s=110, facecolor="none", edgecolor=INK, linewidths=1.2, zorder=4)
        if pos(xs[0]) == 0 and not numeric or field == "lambda_l2":
            ax.set_xticks(range(len(xs)), [f"{x:g}" if isinstance(x, float) else str(x) for x in xs])
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel(field)
    axes[0].set_ylabel(_axis_label(metric))
    extra = [Line2D([], [], color=INK_2, ls=LINESTYLES[i % len(LINESTYLES)], lw=2,
                    label=f"{style_field} = {s}") for i, s in enumerate(styles)] if style_field else []
    extra.append(Line2D([], [], lw=0, marker="o", markerfacecolor="none", markeredgecolor=INK,
                        markersize=10, label="best mean"))
    _tech_legend(fig, techs, extra=extra)
    c = rows[0].config
    fixed = {k: c[k] for k in ("n_particles", "max_iter", "g", "lambda_l2", "l2_form", "selection_rule")
             if k not in (field, style_field)}
    _titles(fig, f"Sensitivity to {field}: {METRICS[metric].label} (mean ± 95% CI over seeds; "
                 f"ring = best mean)", ", ".join(f"{k} = {v}" for k, v in fixed.items()))
    return _save(fig, out / (f"sweep_{field}" + (f"_by_{style_field}" if style_field else "") + f"_{metric}"),
                 formats)


# ---------------------------------------------------------------------------- CLI

FIGURES = ["box", "forest", "pareto", "contrast", "convergence", "radar", "sweep"]


def main():
    p = argparse.ArgumentParser(description="Figuras del paper 2 (ver docstring).")
    p.add_argument("--output-dir", default="./output", help="Raíz con los cv_summary.json")
    p.add_argument("--figs-dir", default="./figs", help="Destino de las figuras")
    p.add_argument("--format", default="png", choices=["png", "pdf", "both"])
    p.add_argument("--figures", nargs="*", default=None, choices=FIGURES,
                   help="Figuras a generar (default: todas las aplicables)")
    p.add_argument("--filter", nargs="*", default=[], metavar="CAMPO=VALOR",
                   help="Filtros del config (core/report.py:CONFIG_FIELDS y selection_rule)")
    p.add_argument("--metric", default=PRIMARY_METRIC, help="Métrica principal (core/report.py:METRICS)")
    p.add_argument("--cost", default="time", choices=["time", "evals", "gpu_mb"], help="Eje de costo del Pareto")
    p.add_argument("--pairs", nargs="*", default=["t1-t3", "t2-t3", "t1-t2"],
                   help="Pares a-b de técnicas para el forest (Δ = b − a)")
    p.add_argument("--unit", default="seed", choices=["seed", "fold"], help="Unidad pareada")
    p.add_argument("--alpha", type=float, default=0.05, help="Nivel de significancia (Holm)")
    p.add_argument("--contrast", default=None, help="Campo con 2 valores para la figura contrast")
    p.add_argument("--contrast-metrics", nargs="*", default=["test_bacc", "optimism_bacc"])
    p.add_argument("--trace-x", default="evals", choices=["evals", "time"])
    p.add_argument("--trace-metric", default="acc", choices=["acc", "ce"])
    p.add_argument("--radar-caps", nargs=3, type=float, default=[0.15, 0.10, 0.10],
                   metavar=("STD", "SEED_STD", "OPTIMISM"), help="Cotas de normalización del radar")
    p.add_argument("--sweep", default=None, help="Campo barrido (p. ej. g, n_particles, lambda_l2)")
    p.add_argument("--sweep-style", default=None, help="Segundo campo, como estilo de línea (p. ej. max_iter)")
    p.add_argument("--sweep-metric", default="val_acc",
                   help="Métrica del barrido (default val_acc: los barridos eligen por validación)")
    args = p.parse_args()

    for m in [args.metric, args.sweep_metric] + args.contrast_metrics:
        if m not in METRICS:
            raise SystemExit(f"❌ Métrica desconocida {m!r}. Válidas: {', '.join(METRICS)}")
    setup_style()
    rows = apply_filters(load_folds(Path(args.output_dir)), parse_filters(args.filter))
    if not rows:
        raise SystemExit("❌ No hay folds que cumplan los filtros.")
    formats = ["png", "pdf"] if args.format == "both" else [args.format]
    out = Path(args.figs_dir)
    pairs = [tuple(x.split("-", 1)) for x in args.pairs]

    figs = args.figures or [f for f in FIGURES
                            if (f != "contrast" or args.contrast) and (f != "sweep" or args.sweep)]
    free = {"contrast": args.contrast, "sweep": args.sweep}
    written = []
    for f in figs:
        loose = [x for x in ((free["contrast"],) if f == "contrast" else
                             (args.sweep, args.sweep_style) if f == "sweep" else ()) if x]
        extra = varying_fields(rows, exclude=("technique", "dataset", "layer_sizes", *loose))
        if extra:
            print(f"⚠  {f}: las corridas mezclan configuraciones ("
                  + "; ".join(f"{k} ∈ {v}" for k, v in extra.items())
                  + "). Usa --filter" + (" o --contrast/--sweep" if not loose else "") + ". Se omite.")
            continue
        if f == "box":
            written += fig_box(rows, args.metric, formats, out)
        elif f == "forest":
            written += fig_forest(rows, args.metric, pairs, args.unit, args.alpha, formats, out)
        elif f == "pareto":
            written += fig_pareto(rows, args.metric, args.cost, formats, out)
        elif f == "contrast":
            if not args.contrast:
                print("⚠  contrast necesita --contrast CAMPO. Se omite.")
                continue
            written += fig_contrast(rows, args.contrast, args.contrast_metrics, args.unit, args.alpha,
                                    formats, out)
        elif f == "convergence":
            written += fig_convergence(rows, args.trace_x, args.trace_metric, formats, out)
        elif f == "radar":
            written += fig_radar(rows, args.metric, args.radar_caps, formats, out)
        elif f == "sweep":
            if not args.sweep:
                print("⚠  sweep necesita --sweep CAMPO. Se omite.")
                continue
            written += fig_sweep(rows, args.sweep, args.sweep_style, args.sweep_metric, formats, out)
    for w in written:
        print(f"📈 {w}")


if __name__ == "__main__":
    main()

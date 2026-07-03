"""Visualizaciones para el benchmark CV multi-seed.

Genera dos variantes de figuras a partir de los `cv_summary.json` bajo `output/`:

  **Globales** (grid 2x2 con los 4 datasets), en `figs/global/`:
    1. radar_by_dataset.{png,pdf}        - 5 métricas normalizadas [0,1]
    2. boxplot_by_dataset.{png,pdf}      - distribución test_acc
    3. pareto_by_dataset.{png,pdf}       - tiempo vs accuracy

  **Individuales** (una figura por dataset), en `figs/per_dataset/`:
    4. radar_<dataset>.{png,pdf}
    5. boxplot_<dataset>.{png,pdf}
    6. pareto_<dataset>.{png,pdf}

  **Heatmap** (no aplica per-dataset), en `figs/`:
    7. sensitivity_heatmap.{png,pdf}     - P × max_iter (si hay datos)

Uso:
    cd re-design
    python visualize.py                  # todo
    python visualize.py --radar          # solo radar (global + individuales)
    python visualize.py --boxplot        # solo boxplot
    python visualize.py --pareto         # solo pareto
    python visualize.py --sensitivity    # solo heatmap
    python visualize.py --no-individual  # omitir per-dataset
    python visualize.py --no-global      # omitir grid 2x2
    python visualize.py --figs-dir figs/ # otro destino raíz
    python visualize.py --format both    # PNG y PDF
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def setup_paper_style():
    """Configura matplotlib para producir figuras listas para paper.

    - Tipografía serif (DejaVu Serif disponible en todos los sistemas).
    - Tamaños consistentes entre figuras.
    - Líneas/markers más gruesos para legibilidad en print.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "patch.linewidth": 1.0,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# Paleta consistente por técnica (la misma en TODAS las figuras).
TECHNIQUE_COLORS = {
    "t1":        "#1f77b4",  # azul
    "t2":        "#ff7f0e",  # naranja
    "t2-jacobi": "#d62728",  # rojo
    "t3":        "#2ca02c",  # verde
    "t4":        "#9467bd",  # violeta
}
TECHNIQUE_LABELS = {
    "t1":        "T1 (BCD seq)",
    "t2":        "T2 (async)",
    "t2-jacobi": "T2-jacobi",
    "t3":        "T3 (lockstep)",
    "t4":        "T4 (single)",
}
TECHNIQUE_MARKERS = {
    "t1": "o", "t2": "s", "t2-jacobi": "D", "t3": "^", "t4": "v",
}
TECHNIQUES_ORDER = ["t1", "t2", "t2-jacobi", "t3", "t4"]

DATASETS_ORDER = ["iris", "wine", "circle", "breast"]
DATASET_LABELS = {
    "iris":   "Iris",
    "wine":   "Wine",
    "circle": "Circles",
    "breast": "Breast Cancer",
}

# Configuración "principal" (la del benchmark base). Sólo estas corridas aparecen
# en las figuras radar/boxplot/Pareto (sensitivity sí explora todas las combinaciones).
MAIN_CONFIG = {"n_particles": 100, "max_iter": 500}

METRIC_NAMES = ["Accuracy", "Stability", "Reproducibility", "Speed", "Calibration"]


# ============================================================================
# Carga y agregación
# ============================================================================

def load_cv_summaries(output_dir: Path) -> List[Dict[str, Any]]:
    """Carga todos los cv_summary.json del output_dir."""
    summaries = []
    for p in sorted(output_dir.rglob("cv_summary.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            data["__path__"] = str(p.parent.relative_to(output_dir))
            summaries.append(data)
        except Exception as e:
            print(f"⚠  No pude leer {p}: {e}")
    return summaries


def group_by_config(summaries, include_seed: bool = False):
    """Agrupa por (tech, ds, P, iter[, seed]). Devuelve dict key → lista de summaries."""
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        cfg = s["config"]
        if include_seed:
            key = (cfg["technique"], cfg["dataset"], cfg["n_particles"],
                   cfg["max_iter"], cfg["seed"])
        else:
            key = (cfg["technique"], cfg["dataset"], cfg["n_particles"], cfg["max_iter"])
        groups[key].append(s)
    return groups


def filter_main(summaries):
    """Devuelve sólo summaries con la config principal (P=100, max_iter=500)."""
    return [s for s in summaries
            if s["config"]["n_particles"] == MAIN_CONFIG["n_particles"]
            and s["config"]["max_iter"] == MAIN_CONFIG["max_iter"]]


def get_fold_test_accs(summary):
    """Extrae los test_acc por fold de un cv_summary."""
    return [f["test_acc"] for f in summary["folds"]]


def get_fold_val_accs(summary):
    return [f["val_acc"] for f in summary["folds"]]


def get_total_time(summary):
    """Tiempo total de la CV completa (suma de los K folds)."""
    return sum(f["total_time"] for f in summary["folds"])


# ============================================================================
# 1. Radar / Pentágono
# ============================================================================

def compute_radar_metrics(
    summaries: List[Dict[str, Any]],
    time_max_in_dataset: float,
    std_cap: float = 0.3,
    seed_std_cap: float = 0.2,
    gap_cap: float = 0.2,
) -> Dict[str, float]:
    """Computa las 5 métricas normalizadas a [0,1] (1 = ideal).

    - Accuracy:        test_acc_mean (directo)
    - Stability:       1 - test_acc_std/std_cap (capeado)
    - Reproducibility: 1 - seed_std/seed_std_cap (capeado)
    - Speed:           1 - time_mean/time_max_in_dataset
    - Calibration:     1 - |val_acc_mean - test_acc_mean|/gap_cap (capeado)
    """
    all_test, all_val, seed_means, times = [], [], [], []
    for s in summaries:
        accs = get_fold_test_accs(s)
        all_test.extend(accs)
        all_val.extend(get_fold_val_accs(s))
        if accs:
            seed_means.append(statistics.mean(accs))
        times.append(get_total_time(s))

    test_acc_mean = statistics.mean(all_test) if all_test else 0.0
    test_acc_std = statistics.pstdev(all_test) if len(all_test) > 1 else 0.0
    val_acc_mean = statistics.mean(all_val) if all_val else 0.0
    seed_std = statistics.pstdev(seed_means) if len(seed_means) > 1 else 0.0
    time_mean = statistics.mean(times) if times else 0.0

    accuracy = max(0.0, min(1.0, test_acc_mean))
    stability = max(0.0, 1.0 - min(test_acc_std / std_cap, 1.0))
    reproducibility = max(0.0, 1.0 - min(seed_std / seed_std_cap, 1.0))
    speed = (1.0 - min(time_mean / time_max_in_dataset, 1.0)
             if time_max_in_dataset > 0 else 0.0)
    speed = max(0.0, min(1.0, speed))
    gap = abs(val_acc_mean - test_acc_mean)
    calibration = max(0.0, 1.0 - min(gap / gap_cap, 1.0))

    return {
        "Accuracy":        accuracy,
        "Stability":       stability,
        "Reproducibility": reproducibility,
        "Speed":           speed,
        "Calibration":     calibration,
    }


def _compute_time_max_by_dataset(groups) -> Dict[str, float]:
    time_max_by_ds: Dict[str, float] = defaultdict(float)
    for (tech, ds, P, it), sums in groups.items():
        for s in sums:
            time_max_by_ds[ds] = max(time_max_by_ds[ds], get_total_time(s))
    return time_max_by_ds


def _draw_radar_subplot(ax, ds, groups, time_max_by_ds, angles, title_pad=24):
    """Dibuja el radar de un dataset en un axes polar dado."""
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_NAMES, fontsize=12, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                        fontsize=9, color="#555")
    ax.set_title(DATASET_LABELS[ds], fontsize=15, weight="bold", pad=title_pad)
    ax.grid(True, alpha=0.4, linewidth=0.7)
    ax.spines["polar"].set_linewidth(1.2)

    for tech in TECHNIQUES_ORDER:
        key = (tech, ds, MAIN_CONFIG["n_particles"], MAIN_CONFIG["max_iter"])
        if key not in groups:
            continue
        metrics = compute_radar_metrics(groups[key], time_max_by_ds[ds])
        values = [metrics[m] for m in METRIC_NAMES]
        values += values[:1]
        color = TECHNIQUE_COLORS[tech]
        ax.plot(angles, values, color=color, linewidth=2.5,
                marker=TECHNIQUE_MARKERS[tech], markersize=8,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.fill(angles, values, color=color, alpha=0.13)


def plot_radar(summaries: List[Dict[str, Any]], output_path: Path) -> bool:
    """Radar global: grid 2x2 con los 4 datasets."""
    main = filter_main(summaries)
    if not main:
        print(f"⚠  No hay corridas con P={MAIN_CONFIG['n_particles']} "
              f"max_iter={MAIN_CONFIG['max_iter']} (config principal).")
        return False

    groups = group_by_config(main, include_seed=False)
    time_max_by_ds = _compute_time_max_by_dataset(groups)

    angles = np.linspace(0, 2 * np.pi, len(METRIC_NAMES), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14),
                              subplot_kw={"projection": "polar"})
    axes = axes.flatten()

    for i, ds in enumerate(DATASETS_ORDER):
        _draw_radar_subplot(axes[i], ds, groups, time_max_by_ds, angles)

    handles = [Patch(facecolor=TECHNIQUE_COLORS[t], label=TECHNIQUE_LABELS[t],
                     edgecolor="black", linewidth=0.5)
               for t in TECHNIQUES_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11,
               bbox_to_anchor=(0.5, 0.01), frameon=True, framealpha=0.95,
               edgecolor="black", fancybox=False)
    fig.suptitle(
        "Multi-metric comparison by dataset (normalized [0,1], higher = better)",
        fontsize=16, weight="bold", y=0.995
    )
    fig.text(0.5, 0.045,
             "Accuracy = test_acc | Stability = 1−std (cap 0.3) | "
             "Reproducibility = 1−seed_std (cap 0.2) | "
             "Speed = 1−time/time_max | Calibration = 1−|val−test| (cap 0.2)",
             ha="center", fontsize=9, style="italic", color="#555")

    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_radar_single(summaries, output_path: Path, ds: str) -> bool:
    """Radar individual: un único dataset."""
    main = filter_main(summaries)
    if not main:
        return False

    groups = group_by_config(main, include_seed=False)
    time_max_by_ds = _compute_time_max_by_dataset(groups)
    if time_max_by_ds.get(ds, 0.0) == 0.0:
        return False

    angles = np.linspace(0, 2 * np.pi, len(METRIC_NAMES), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.5, 9.0))
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    _draw_radar_subplot(ax, ds, groups, time_max_by_ds, angles, title_pad=28)

    handles = [Patch(facecolor=TECHNIQUE_COLORS[t], label=TECHNIQUE_LABELS[t],
                     edgecolor="black", linewidth=0.5)
               for t in TECHNIQUES_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.01), frameon=True, framealpha=0.95,
               edgecolor="black", fancybox=False)
    fig.suptitle(
        f"Multi-metric comparison — {DATASET_LABELS[ds]} (normalized [0,1])",
        fontsize=14, weight="bold", y=0.995
    )
    fig.text(0.5, 0.06,
             "Accuracy = test_acc | Stability = 1−std (cap 0.3) | "
             "Reproducibility = 1−seed_std (cap 0.2)\n"
             "Speed = 1−time/time_max | Calibration = 1−|val−test| (cap 0.2)",
             ha="center", fontsize=8.5, style="italic", color="#555")

    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


# ============================================================================
# 2. Boxplot
# ============================================================================

def _draw_boxplot_subplot(ax, ds, groups) -> bool:
    """Dibuja el boxplot de un dataset en un axes dado. Devuelve True si hay datos."""
    data, labels, colors, means, stds = [], [], [], [], []
    for tech in TECHNIQUES_ORDER:
        key = (tech, ds, MAIN_CONFIG["n_particles"], MAIN_CONFIG["max_iter"])
        if key not in groups:
            continue
        all_accs = []
        for s in groups[key]:
            all_accs.extend(get_fold_test_accs(s))
        if not all_accs:
            continue
        data.append(all_accs)
        labels.append(TECHNIQUE_LABELS[tech])
        colors.append(TECHNIQUE_COLORS[tech])
        means.append(statistics.mean(all_accs))
        stds.append(statistics.pstdev(all_accs) if len(all_accs) > 1 else 0.0)

    if not data:
        ax.set_visible(False)
        return False

    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True,
        widths=0.55, showmeans=True,
        medianprops={"color": "black", "linewidth": 2.0},
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 8,
                   "markeredgewidth": 1.5},
        flierprops={"marker": "o", "markerfacecolor": "white",
                    "markeredgecolor": "black", "markersize": 6, "alpha": 0.7},
        whiskerprops={"linewidth": 1.2}, capprops={"linewidth": 1.2},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.3)

    for j, (m, sd) in enumerate(zip(means, stds)):
        ax.text(j + 1, 0.035, f"{m:.3f}±{sd:.3f}",
                ha="center", fontsize=8.5, color=colors[j], weight="bold")

    ax.set_title(DATASET_LABELS[ds], fontsize=15, weight="bold", pad=8)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_ylim(0.0, 1.08)

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1.0)
    ax.axhline(0.33, color="gray", linestyle="--", alpha=0.4, linewidth=1.0)
    ax.text(0.02, 0.51, "random (binary)", transform=ax.get_yaxis_transform(),
            fontsize=7, color="gray", style="italic")
    ax.text(0.02, 0.34, "random (3-class)", transform=ax.get_yaxis_transform(),
            fontsize=7, color="gray", style="italic")

    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=10)
    return True


def plot_boxplot(summaries: List[Dict[str, Any]], output_path: Path) -> bool:
    """Boxplot global: grid 2x2 con los 4 datasets."""
    main = filter_main(summaries)
    if not main:
        return False

    groups = group_by_config(main, include_seed=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    any_data = False
    for i, ds in enumerate(DATASETS_ORDER):
        if _draw_boxplot_subplot(axes[i], ds, groups):
            any_data = True

    if not any_data:
        plt.close(fig)
        return False

    fig.suptitle(
        "Test accuracy distribution by technique × dataset "
        "(n = N_seeds × 5 folds per cell)",
        fontsize=16, weight="bold", y=0.995
    )
    fig.text(0.5, 0.005,
             "Box: Q1-Q3 with median (black line). Diamond: mean (annotated as mean±std). "
             "Circles: outliers. Dashed lines: random baselines.",
             ha="center", fontsize=9, style="italic", color="#555")

    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_boxplot_single(summaries, output_path: Path, ds: str) -> bool:
    """Boxplot individual: un único dataset."""
    main = filter_main(summaries)
    if not main:
        return False

    groups = group_by_config(main, include_seed=False)

    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    if not _draw_boxplot_subplot(ax, ds, groups):
        plt.close(fig)
        return False

    fig.suptitle(
        f"Test accuracy distribution — {DATASET_LABELS[ds]} "
        "(n = N_seeds × 5 folds)",
        fontsize=14, weight="bold", y=0.995
    )
    fig.text(0.5, 0.005,
             "Box: Q1-Q3 with median (black line). Diamond: mean (annotated as mean±std). "
             "Circles: outliers. Dashed lines: random baselines.",
             ha="center", fontsize=9, style="italic", color="#555")

    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


# ============================================================================
# 3. Pareto: tiempo vs accuracy
# ============================================================================

def _draw_pareto_subplot(ax, ds, groups, show_legend=False, show_frontier_label=False) -> bool:
    """Dibuja el Pareto de un dataset. Devuelve True si hay datos."""
    tech_means: Dict[str, Tuple[float, float]] = {}
    has_data = False

    for tech in TECHNIQUES_ORDER:
        key = (tech, ds)
        if key not in groups:
            continue
        seed_times = []
        seed_accs = []
        for s in groups[key]:
            t = get_total_time(s)
            accs = get_fold_test_accs(s)
            if not accs:
                continue
            acc_mean = statistics.mean(accs)
            acc_std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
            seed_times.append(t)
            seed_accs.append(acc_mean)
            ax.errorbar(t, acc_mean, yerr=acc_std, fmt=TECHNIQUE_MARKERS[tech],
                        color=TECHNIQUE_COLORS[tech], markersize=6, alpha=0.35,
                        capsize=2, elinewidth=0.7)
            has_data = True

        if seed_times:
            t_mean = statistics.mean(seed_times)
            a_mean = statistics.mean(seed_accs)
            tech_means[tech] = (t_mean, a_mean)
            ax.scatter([t_mean], [a_mean], s=220, marker=TECHNIQUE_MARKERS[tech],
                       color=TECHNIQUE_COLORS[tech], edgecolor="black",
                       linewidth=1.8, zorder=10, label=TECHNIQUE_LABELS[tech])
            ax.annotate(
                TECHNIQUE_LABELS[tech].split(" ")[0],
                xy=(t_mean, a_mean), xytext=(8, 8),
                textcoords="offset points",
                fontsize=10, weight="bold",
                color=TECHNIQUE_COLORS[tech],
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=TECHNIQUE_COLORS[tech], linewidth=1.0,
                          alpha=0.85),
                zorder=11,
            )

    if not has_data:
        ax.set_visible(False)
        return False

    # Pareto frontier sobre los centroides.
    if len(tech_means) >= 2:
        pts = sorted(tech_means.items(), key=lambda kv: kv[1][0])
        pareto_pts: List[Tuple[str, float, float]] = []
        best_acc = -1.0
        for tech_name, (t, a) in pts:
            if a > best_acc:
                pareto_pts.append((tech_name, t, a))
                best_acc = a
        if len(pareto_pts) >= 2:
            xs = [t for _, t, _ in pareto_pts]
            ys = [a for _, _, a in pareto_pts]
            ax.plot(xs, ys, "--", color="#444", alpha=0.7, linewidth=1.8,
                    zorder=5, label="Pareto frontier")
            if show_frontier_label:
                mid_x = (xs[0] * xs[-1]) ** 0.5
                mid_y = (ys[0] + ys[-1]) / 2 - 0.08
                ax.annotate(
                    "Pareto frontier",
                    xy=(mid_x, mid_y), fontsize=10, style="italic",
                    color="#333", ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0f0f0",
                              edgecolor="#888", linewidth=0.8, alpha=0.9),
                )

    ax.set_title(DATASET_LABELS[ds], fontsize=15, weight="bold", pad=8)
    ax.set_xlabel("Total CV wall-time (s, log scale)", fontsize=12)
    ax.set_ylabel("Test Accuracy (mean per seed)", fontsize=12)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, alpha=0.3, which="both", linewidth=0.5)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(fontsize=9, loc="lower left", framealpha=0.92,
                  edgecolor="black", fancybox=False)

    ax.annotate("← faster", xy=(0.02, 0.98), xycoords="axes fraction",
                fontsize=8.5, style="italic", color="#666", weight="bold",
                va="top")
    ax.annotate("↑ better", xy=(0.98, 0.05), xycoords="axes fraction",
                fontsize=8.5, style="italic", color="#666", weight="bold",
                ha="right", va="bottom")
    return True


def plot_pareto(summaries: List[Dict[str, Any]], output_path: Path) -> bool:
    """Pareto global: grid 2x2 con los 4 datasets."""
    main = filter_main(summaries)
    if not main:
        return False

    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for s in main:
        cfg = s["config"]
        groups[(cfg["technique"], cfg["dataset"])].append(s)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for i, ds in enumerate(DATASETS_ORDER):
        _draw_pareto_subplot(
            axes[i], ds, groups,
            show_legend=(i == 0),
            show_frontier_label=(i == 0),
        )

    fig.suptitle(
        "Pareto: wall-time vs test accuracy by dataset",
        fontsize=16, weight="bold", y=0.995
    )
    fig.text(0.5, 0.005,
             "Small markers: individual seeds (error bars = intra-seed fold std). "
             "Large markers with black edge: per-technique centroid. "
             "Dashed line: Pareto frontier (non-dominated points).",
             ha="center", fontsize=9, style="italic", color="#555")

    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_pareto_single(summaries, output_path: Path, ds: str) -> bool:
    """Pareto individual: un único dataset."""
    main = filter_main(summaries)
    if not main:
        return False

    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for s in main:
        cfg = s["config"]
        groups[(cfg["technique"], cfg["dataset"])].append(s)

    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    if not _draw_pareto_subplot(ax, ds, groups,
                                 show_legend=True, show_frontier_label=True):
        plt.close(fig)
        return False

    fig.suptitle(
        f"Pareto: wall-time vs test accuracy — {DATASET_LABELS[ds]}",
        fontsize=14, weight="bold", y=0.995
    )
    fig.text(0.5, 0.005,
             "Small markers: individual seeds (error bars = intra-seed fold std). "
             "Large markers with black edge: per-technique centroid. "
             "Dashed line: Pareto frontier.",
             ha="center", fontsize=9, style="italic", color="#555")

    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


# ============================================================================
# 4. Sensitivity heatmap (P × max_iter)
# ============================================================================

def plot_sensitivity(summaries: List[Dict[str, Any]], output_path: Path) -> bool:
    groups = group_by_config(summaries, include_seed=False)

    parts = sorted({k[2] for k in groups.keys()})
    iters = sorted({k[3] for k in groups.keys()})

    if len(parts) < 2 and len(iters) < 2:
        print("⚠  Necesito ≥2 valores distintos en P o max_iter para el heatmap.")
        print(f"   Encontrado: P={parts}, max_iter={iters}.")
        return False

    n_techs = len(TECHNIQUES_ORDER)
    n_ds = len(DATASETS_ORDER)
    fig, axes = plt.subplots(n_techs, n_ds,
                              figsize=(3.2 * n_ds, 2.8 * n_techs),
                              squeeze=False)

    last_im = None
    for i, tech in enumerate(TECHNIQUES_ORDER):
        for j, ds in enumerate(DATASETS_ORDER):
            ax = axes[i, j]
            data = np.full((len(iters), len(parts)), np.nan)
            for ii, it in enumerate(iters):
                for jj, P in enumerate(parts):
                    key = (tech, ds, P, it)
                    if key in groups:
                        all_accs = []
                        for s in groups[key]:
                            all_accs.extend(get_fold_test_accs(s))
                        if all_accs:
                            data[ii, jj] = statistics.mean(all_accs)

            im = ax.imshow(data, cmap="RdYlGn", vmin=0.4, vmax=1.0,
                           aspect="auto", origin="lower")
            last_im = im
            ax.set_xticks(range(len(parts)))
            ax.set_xticklabels(parts, fontsize=8)
            ax.set_yticks(range(len(iters)))
            ax.set_yticklabels(iters, fontsize=8)

            if i == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=11, weight="bold")
            if j == 0:
                ax.set_ylabel(f"{TECHNIQUE_LABELS[tech]}\nmax_iter",
                              fontsize=9, weight="bold")
            if i == n_techs - 1:
                ax.set_xlabel("n_particles", fontsize=9)

            for ii in range(len(iters)):
                for jj in range(len(parts)):
                    val = data[ii, jj]
                    if not np.isnan(val):
                        text_color = "white" if val < 0.6 else "black"
                        ax.text(jj, ii, f"{val:.3f}", ha="center", va="center",
                                color=text_color, fontsize=7)

    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(last_im, cax=cbar_ax, label="Test accuracy mean")

    fig.suptitle(
        "Sensitivity analysis: P × max_iter (filas = técnicas, columnas = datasets)",
        fontsize=14, weight="bold", y=0.995
    )
    fig.tight_layout(rect=(0, 0, 0.9, 0.96))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualizaciones para el benchmark CV multi-seed"
    )
    parser.add_argument("--output-dir", default="./output",
                        help="Directorio raíz con cv_summary.json (default: ./output)")
    parser.add_argument("--figs-dir", default="./figs",
                        help="Donde guardar las figuras (default: ./figs)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "both"],
                        help="Formato (default: png; 'both' genera PNG y PDF)")
    parser.add_argument("--radar", action="store_true", help="Sólo radar")
    parser.add_argument("--boxplot", action="store_true", help="Sólo boxplot")
    parser.add_argument("--pareto", action="store_true", help="Sólo Pareto")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Sólo heatmap P × iter")
    parser.add_argument("--no-global", action="store_true",
                        help="Omitir versiones globales (grid 2x2)")
    parser.add_argument("--no-individual", action="store_true",
                        help="Omitir versiones individuales (una por dataset)")
    args = parser.parse_args()

    no_specific = not any([args.radar, args.boxplot, args.pareto, args.sensitivity])
    do_all = no_specific

    figs_dir = Path(args.figs_dir).resolve()
    global_dir = figs_dir / "global"
    per_ds_dir = figs_dir / "per_dataset"
    figs_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_global:
        global_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_individual:
        per_ds_dir.mkdir(parents=True, exist_ok=True)

    extensions = ["png", "pdf"] if args.format == "both" else [args.format]

    setup_paper_style()

    output_dir = Path(args.output_dir).resolve()
    print(f"📂 Cargando corridas desde: {output_dir}")
    summaries = load_cv_summaries(output_dir)
    print(f"📊 {len(summaries)} cv_summary.json encontrados.")
    print(f"🎨 Estilo: paper (serif, dpi=300)\n")

    if not summaries:
        print("❌ No hay datos para visualizar.")
        return

    n_ok = 0

    def _save_global(plot_fn, name, label):
        nonlocal n_ok
        if args.no_global:
            return
        for ext in extensions:
            path = global_dir / f"{name}.{ext}"
            if plot_fn(summaries, path):
                print(f"✅ {label:<14} {path.relative_to(figs_dir)}")
                n_ok += 1

    def _save_per_dataset(plot_single_fn, prefix, label):
        nonlocal n_ok
        if args.no_individual:
            return
        for ds in DATASETS_ORDER:
            for ext in extensions:
                path = per_ds_dir / f"{prefix}_{ds}.{ext}"
                if plot_single_fn(summaries, path, ds):
                    print(f"   ↳ {label} {ds:<7}  {path.relative_to(figs_dir)}")
                    n_ok += 1

    if do_all or args.radar:
        _save_global(plot_radar, "radar_by_dataset", "Radar (global):")
        _save_per_dataset(plot_radar_single, "radar", "radar")

    if do_all or args.boxplot:
        _save_global(plot_boxplot, "boxplot_by_dataset", "Boxplot (global):")
        _save_per_dataset(plot_boxplot_single, "boxplot", "boxplot")

    if do_all or args.pareto:
        _save_global(plot_pareto, "pareto_by_dataset", "Pareto (global):")
        _save_per_dataset(plot_pareto_single, "pareto", "pareto")

    if do_all or args.sensitivity:
        # El sensitivity no aplica per-dataset (ya tiene filas/columnas internas).
        for ext in extensions:
            path = figs_dir / f"sensitivity_heatmap.{ext}"
            if plot_sensitivity(summaries, path):
                print(f"✅ {'Sensitivity:':<14} {path.relative_to(figs_dir)}")
                n_ok += 1

    print(f"\n📈 {n_ok} figura(s) generada(s) en {figs_dir}")


if __name__ == "__main__":
    main()

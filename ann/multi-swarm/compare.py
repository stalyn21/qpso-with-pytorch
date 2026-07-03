"""Comparador de corridas: lee todos los run.json y cv_summary.json bajo `output/`
y produce un resumen tabular (markdown + CSV) cruzando técnica × dataset × config.

Detecta dos tipos de corrida:
  - **Single-split** (`run.json` directo): un split 70/15/15 + N sesiones.
  - **K-fold CV** (`cv_summary.json` en `cv<K>/`): K folds, agregado mean ± std.

Los runs single-split se reportan en la tabla principal; los CV en una tabla aparte
con columnas mean / std / min / max.

Uso:
    cd re-design
    python compare.py                          # imprime ambas tablas y guarda CSV/MD
    python compare.py --filter-dataset iris    # solo iris
    python compare.py --output-dir ./output    # otro directorio raíz
    python compare.py --sort-by test_acc       # ordenar por columna (single-split)
    python compare.py --plot                   # generar learning_curve.png en cada run dir
    python compare.py --plot --plot-comparison # además, plot agregado por dataset
"""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CV_DIR_RE = re.compile(r"^cv\d+$")


# Columnas del resumen y cómo extraerlas de run.json
COLUMNS = [
    ("technique",      lambda r: r["config"]["technique"]),
    ("dataset",        lambda r: r["config"]["dataset"]),
    ("hidden",         lambda r: "x".join(map(str, r["config"]["layer_sizes"][1:-1]))),
    ("particles",      lambda r: r["config"]["n_particles"]),
    ("max_iter",       lambda r: r["config"]["max_iter"]),
    ("sessions",       lambda r: r["config"]["n_sessions"]),
    ("seed",           lambda r: r["config"]["seed"]),
    ("winner_session", lambda r: r["winner_by_val"]["session"]),
    ("val_acc",        lambda r: r["winner_by_val"]["val_acc"]),
    ("val_mse",        lambda r: r["winner_by_val"]["val_cost"]),
    ("val_score",      lambda r: r["winner_by_val"]["val_score"]),
    ("train_loss",     lambda r: r["winner_by_val"].get("train_loss", float("nan"))),
    ("test_acc",       lambda r: r["winner_by_val"]["final_test_acc"]),
    ("test_mse",       lambda r: r["winner_by_val"]["final_test_cost"]),
    ("test_score",     lambda r: r["winner_by_val"].get("final_test_score", float("nan"))),
    ("gap_winner",     lambda r: _winner_gap(r)),
    ("gap_mean",       lambda r: _mean_gap(r)),
    ("total_time",     lambda r: r["total_time"]),
]


def _winner_gap(r):
    """val_acc - test_acc del ganador (positivo => val sobreestima test)."""
    w = r["winner_by_val"]
    return w["val_acc"] - w["final_test_acc"]


def _mean_gap(r):
    """Promedio de val_acc - test_acc a lo largo de las sesiones."""
    lc = r.get("learning_curve")
    if not lc or "val_test_acc_gap" not in lc or not lc["val_test_acc_gap"]:
        return float("nan")
    g = lc["val_test_acc_gap"]
    return sum(g) / len(g)


def _is_inside_cv(rel_parts) -> bool:
    """True si el path contiene un componente cv\\d+ (corrida dentro de CV)."""
    return any(CV_DIR_RE.match(part) for part in rel_parts)


def load_runs(output_dir: Path) -> List[Dict[str, Any]]:
    """Recorre output_dir buscando run.json de single-split (excluye los de CV folds)."""
    runs = []
    for p in output_dir.rglob("run.json"):
        rel_parts = p.relative_to(output_dir).parts
        if _is_inside_cv(rel_parts):
            continue  # los runs por fold de CV se agregan vía cv_summary.json
        try:
            with open(p) as f:
                data = json.load(f)
            data["__path__"] = str(p.parent.relative_to(output_dir))
            runs.append(data)
        except Exception as e:
            print(f"⚠  No pude leer {p}: {e}")
    return runs


def load_cv_summaries(output_dir: Path) -> List[Dict[str, Any]]:
    """Recorre output_dir buscando todos los cv_summary.json."""
    summaries = []
    for p in output_dir.rglob("cv_summary.json"):
        try:
            with open(p) as f:
                data = json.load(f)
            data["__path__"] = str(p.parent.relative_to(output_dir))
            summaries.append(data)
        except Exception as e:
            print(f"⚠  No pude leer {p}: {e}")
    return summaries


def to_rows(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in runs:
        row = {}
        for name, getter in COLUMNS:
            try:
                row[name] = getter(r)
            except KeyError:
                row[name] = None
        rows.append(row)
    return rows


# Columnas para corridas K-fold CV (agregado mean ± std sobre folds).
CV_COLUMNS = [
    ("technique",       lambda c: c["config"]["technique"]),
    ("dataset",         lambda c: c["config"]["dataset"]),
    ("hidden",          lambda c: "x".join(map(str, c["config"]["layer_sizes"][1:-1]))),
    ("particles",       lambda c: c["config"]["n_particles"]),
    ("max_iter",        lambda c: c["config"]["max_iter"]),
    ("folds",           lambda c: c["config"]["cv_folds"]),
    ("sess/fold",       lambda c: c["config"]["n_sessions_per_fold"]),
    ("seed",            lambda c: c["config"]["seed"]),
    ("test_acc_mean",   lambda c: c["aggregated"]["test_acc"]["mean"]),
    ("test_acc_std",    lambda c: c["aggregated"]["test_acc"]["std"]),
    ("test_acc_min",    lambda c: c["aggregated"]["test_acc"]["min"]),
    ("test_acc_max",    lambda c: c["aggregated"]["test_acc"]["max"]),
    ("val_acc_mean",    lambda c: c["aggregated"]["val_acc"]["mean"]),
    ("val_acc_std",     lambda c: c["aggregated"]["val_acc"]["std"]),
    ("train_loss_mean", lambda c: c["aggregated"]["train_loss"]["mean"]),
    ("time_total",      lambda c: c["aggregated"]["total_time"]["mean"] * c["config"]["cv_folds"]),
    ("time_per_fold",   lambda c: c["aggregated"]["total_time"]["mean"]),
]


def to_cv_rows(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for s in summaries:
        row = {}
        for name, getter in CV_COLUMNS:
            try:
                row[name] = getter(s)
            except KeyError:
                row[name] = None
        rows.append(row)
    return rows


# ---------- Agregación multi-seed (CV × N seeds) ----------
#
# Cuando el usuario corre la misma config con varios seeds (ej. seed in {42, 123, 7}),
# cada seed produce un cv_summary.json. Aquí agrupamos por config (sin seed) y
# reportamos:
#   - test_acc_mean ± test_acc_std: grand mean/std sobre N seeds × K folds = N*K puntos.
#   - seed_mean: promedio de los means-per-seed (mismo valor que test_acc_mean por
#     definición — lo dejamos como "centro estimado").
#   - seed_std: std de los means-per-seed → variabilidad ENTRE seeds (diagnóstico).
#     Si seed_std es alta, el método es sensible a la inicialización del QDPSO;
#     si seed_std es baja, los resultados son robustos a través de seeds.
#   - n_seeds, n_runs (= n_seeds × n_folds): tamaño de muestra para defender el reporte.

MULTI_SEED_COLUMNS = [
    ("technique",     lambda r: r["technique"]),
    ("dataset",       lambda r: r["dataset"]),
    ("hidden",        lambda r: r["hidden"]),
    ("particles",     lambda r: r["particles"]),
    ("max_iter",      lambda r: r["max_iter"]),
    ("rounds",        lambda r: r["rounds"]),
    ("n_seeds",       lambda r: r["n_seeds"]),
    ("n_runs",        lambda r: r["n_runs"]),
    ("test_acc_mean", lambda r: r["test_acc_mean"]),
    ("test_acc_std",  lambda r: r["test_acc_std"]),
    ("test_acc_min",  lambda r: r["test_acc_min"]),
    ("test_acc_max",  lambda r: r["test_acc_max"]),
    ("seed_std",      lambda r: r["seed_std"]),
    ("val_acc_mean",  lambda r: r["val_acc_mean"]),
    ("train_loss_mean", lambda r: r["train_loss_mean"]),
    ("time_total",    lambda r: r["time_total"]),
]


def _config_key(cv_summary: Dict[str, Any]) -> Tuple:
    """Clave de agregación: todo lo que define el experimento EXCEPTO el seed.

    Dos cv_summaries comparten la misma configuración si comparten estos campos.
    """
    cfg = cv_summary["config"]
    hidden = tuple(cfg["layer_sizes"][1:-1])
    return (
        cfg["technique"],
        cfg["dataset"],
        hidden,
        cfg["n_particles"],
        cfg["max_iter"],
        cfg.get("n_rounds", 1),
        cfg["cv_folds"],
    )


def aggregate_multi_seed(cv_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Agrupa cv_summaries por configuración (sin seed) y devuelve filas agregadas.

    Sólo retorna grupos con n_seeds >= 2 (un solo seed ya está en la tabla CV).
    """
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for s in cv_summaries:
        groups[_config_key(s)].append(s)

    rows = []
    for key, summaries in groups.items():
        if len(summaries) < 2:
            continue  # sólo agrupar cuando hay 2+ seeds para esta config

        tech, ds, hidden, particles, max_iter, n_rounds, cv_folds = key

        # Recolectar TODOS los puntos individuales (n_seeds × n_folds).
        all_test_accs: List[float] = []
        all_val_accs: List[float] = []
        all_train_losses: List[float] = []
        seed_means: List[float] = []
        seeds_used: List[int] = []
        total_time = 0.0

        for s in summaries:
            fold_test_accs = [f["test_acc"] for f in s["folds"]]
            fold_val_accs  = [f["val_acc"]  for f in s["folds"]]
            fold_train     = [f["train_loss"] for f in s["folds"]]
            fold_times     = [f["total_time"] for f in s["folds"]]

            all_test_accs.extend(fold_test_accs)
            all_val_accs.extend(fold_val_accs)
            all_train_losses.extend(fold_train)
            seed_means.append(statistics.mean(fold_test_accs))
            seeds_used.append(s["config"]["seed"])
            total_time += sum(fold_times)

        n_runs = len(all_test_accs)
        rows.append({
            "technique": tech,
            "dataset": ds,
            "hidden": "x".join(map(str, hidden)),
            "particles": particles,
            "max_iter": max_iter,
            "rounds": n_rounds,
            "n_seeds": len(summaries),
            "n_runs": n_runs,
            "test_acc_mean": statistics.mean(all_test_accs),
            "test_acc_std":  statistics.pstdev(all_test_accs) if n_runs > 1 else 0.0,
            "test_acc_min":  min(all_test_accs),
            "test_acc_max":  max(all_test_accs),
            # std de los means-per-seed: si es alta, el método es sensible a la
            # inicialización del QDPSO entre corridas (no determinismo importa).
            "seed_std": statistics.pstdev(seed_means) if len(seed_means) > 1 else 0.0,
            "val_acc_mean":   statistics.mean(all_val_accs),
            "train_loss_mean": statistics.mean(all_train_losses),
            "time_total": total_time,
            "seeds_used": sorted(seeds_used),
        })

    return rows


def to_multi_seed_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """No-op: aggregate_multi_seed ya devuelve filas con las columnas correctas.
    Este wrapper existe para mantener simetría con to_rows / to_cv_rows.
    """
    return rows


def fmt_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 1e-4 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)


def _sort_rows(rows, columns, sort_by):
    if sort_by and rows and sort_by in rows[0]:
        descending = sort_by in ("val_acc", "test_acc", "test_acc_mean", "val_acc_mean")
        return sorted(
            rows,
            key=lambda r: (r[sort_by] if r[sort_by] is not None else float("inf")),
            reverse=descending,
        )
    return rows


def print_table(
    rows: List[Dict[str, Any]],
    columns=None,
    sort_by: Optional[str] = None,
) -> str:
    if not rows:
        return "(sin runs)"
    cols = columns if columns is not None else COLUMNS
    rows = _sort_rows(rows, cols, sort_by)

    headers = [name for name, _ in cols]
    str_rows = [[fmt_value(r[h]) for h in headers] for r in rows]
    widths = [max(len(h), max((len(row[i]) for row in str_rows), default=0))
              for i, h in enumerate(headers)]

    lines = []
    lines.append(" │ ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("─┼─".join("─" * w for w in widths))
    for row in str_rows:
        lines.append(" │ ".join(row[i].ljust(widths[i]) for i in range(len(headers))))

    return "\n".join(lines)


def to_markdown(
    rows: List[Dict[str, Any]],
    columns=None,
    sort_by: Optional[str] = None,
) -> str:
    if not rows:
        return "_(sin runs)_\n"
    cols = columns if columns is not None else COLUMNS
    rows = _sort_rows(rows, cols, sort_by)

    headers = [name for name, _ in cols]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt_value(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def to_csv(rows: List[Dict[str, Any]], path: Path, columns=None) -> None:
    if not rows:
        path.write_text("")
        return
    cols = columns if columns is not None else COLUMNS
    headers = [name for name, _ in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow({h: row[h] for h in headers})


def plot_learning_curve(run: Dict[str, Any], save_path: Path) -> bool:
    """Genera PNG con val_acc y test_acc por sesión, marca ganadora y muestra gap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠  matplotlib no instalado; --plot omitido")
        return False

    lc = run.get("learning_curve")
    if not lc or not lc.get("val_acc"):
        return False

    sessions = lc.get("session_idx", list(range(len(lc["val_acc"]))))
    val_acc = lc["val_acc"]
    test_acc = lc["test_acc"]
    gaps = lc.get("val_test_acc_gap", [v - t for v, t in zip(val_acc, test_acc)])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(sessions, val_acc, marker="o", label="val ACC", color="#1f77b4")
    ax1.plot(sessions, test_acc, marker="s", label="test ACC", color="#ff7f0e")

    winner_idx = run["winner_by_val"]["session"]
    ax1.axvline(winner_idx, color="red", linestyle=":", alpha=0.6,
                label=f"ganadora (s{winner_idx})")

    cfg = run["config"]
    hidden = "-".join(map(str, cfg["layer_sizes"][1:-1]))
    ax1.set_title(
        f"{cfg['technique']} / {cfg['dataset']}  "
        f"hidden={hidden}  P={cfg['n_particles']}  iters={cfg['max_iter']}  "
        f"seed={cfg['seed']}"
    )
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Subplot inferior: gap val-test (diagnóstico de overfit a val)
    ax2.bar(sessions, gaps, color=["#d62728" if g > 0 else "#2ca02c" for g in gaps], alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("gap\n(val − test)")
    ax2.set_xlabel("Sesión")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return True


def plot_comparison_by_dataset(runs: List[Dict[str, Any]], output_path: Path) -> bool:
    """Para cada dataset, plot superpuesto de test_acc por sesión de las técnicas que corrieron."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    by_ds: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        ds = r["config"]["dataset"]
        by_ds.setdefault(ds, []).append(r)

    if not by_ds:
        return False

    n = len(by_ds)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n), squeeze=False)

    colors = {"t1": "#1f77b4", "t2": "#ff7f0e", "t3": "#2ca02c", "t4": "#d62728"}
    markers = {"t1": "o", "t2": "s", "t3": "^", "t4": "D"}

    for i, (ds, ds_runs) in enumerate(sorted(by_ds.items())):
        ax = axes[i, 0]
        for r in ds_runs:
            lc = r.get("learning_curve") or {}
            test_acc = lc.get("test_acc")
            if not test_acc:
                continue
            sessions = lc.get("session_idx", list(range(len(test_acc))))
            tech = r["config"]["technique"]
            cfg = r["config"]
            label = (f"{tech.upper()}  P={cfg['n_particles']} iters={cfg['max_iter']} "
                     f"sess={cfg['n_sessions']}")
            ax.plot(sessions, test_acc, marker=markers.get(tech, "o"),
                    color=colors.get(tech, "gray"), label=label, alpha=0.85)
        ax.set_title(f"Dataset: {ds}  —  test ACC por sesión")
        ax.set_xlabel("Sesión")
        ax.set_ylabel("Test ACC")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description="Comparador de corridas re-design")
    p.add_argument("--output-dir", default="./output", help="Directorio raíz donde están las corridas")
    p.add_argument("--filter-technique", default=None, help="Filtrar por técnica (t1/t2/t2-jacobi/t3/t4)")
    p.add_argument("--filter-dataset", default=None, help="Filtrar por dataset (iris/wine/breast/circle)")
    p.add_argument("--sort-by", default="val_score", help="Columna para ordenar (default: val_score)")
    p.add_argument("--cv-sort-by", default="test_acc_mean",
                   help="Columna para ordenar la tabla CV (default: test_acc_mean)")
    p.add_argument("--multi-sort-by", default="test_acc_mean",
                   help="Columna para ordenar la tabla multi-seed (default: test_acc_mean)")
    p.add_argument("--csv", default="compare.csv", help="Archivo CSV single-split")
    p.add_argument("--cv-csv", default="compare_cv.csv", help="Archivo CSV CV (default: compare_cv.csv)")
    p.add_argument("--multi-csv", default="compare_multi_seed.csv",
                   help="Archivo CSV multi-seed (sólo si hay configs con N≥2 seeds)")
    p.add_argument("--md", default="compare.md", help="Archivo markdown de salida (incluye todas las tablas)")
    p.add_argument("--no-files", action="store_true", help="No escribir archivos, solo imprimir")
    p.add_argument("--plot", action="store_true",
                   help="Genera learning_curve.png en cada directorio de corrida (single-split y por fold de CV)")
    p.add_argument("--plot-comparison", action="store_true",
                   help="Genera comparison_by_dataset.png con todas las técnicas superpuestas")
    args = p.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        print(f"❌ No existe el directorio {output_dir}")
        return

    runs = load_runs(output_dir)
    cv_summaries = load_cv_summaries(output_dir)

    if not runs and not cv_summaries:
        print(f"⚠  No se encontraron run.json ni cv_summary.json bajo {output_dir}")
        return

    # Filtros
    def _match(cfg):
        if args.filter_technique and cfg["technique"] != args.filter_technique:
            return False
        if args.filter_dataset and cfg["dataset"] != args.filter_dataset:
            return False
        return True

    filtered_runs = [r for r in runs if _match(r["config"])]
    filtered_cvs = [s for s in cv_summaries if _match(s["config"])]

    md_sections = []

    # ---------- Tabla single-split ----------
    if filtered_runs:
        rows = to_rows(filtered_runs)
        print(f"📊 {len(rows)} corrida(s) single-split en {output_dir}\n")
        print(print_table(rows, sort_by=args.sort_by))
        md_sections.append("## Single-split (70/15/15)\n\n" +
                           to_markdown(rows, sort_by=args.sort_by))
    else:
        print(f"📊 0 corridas single-split.\n")

    # ---------- Tabla K-fold CV ----------
    if filtered_cvs:
        cv_rows = to_cv_rows(filtered_cvs)
        print(f"\n📊 {len(cv_rows)} corrida(s) K-fold CV en {output_dir}\n")
        print(print_table(cv_rows, columns=CV_COLUMNS, sort_by=args.cv_sort_by))
        md_sections.append("## K-fold CV\n\n" +
                           to_markdown(cv_rows, columns=CV_COLUMNS, sort_by=args.cv_sort_by))
    else:
        if cv_summaries:  # había CVs pero filtros los descartaron
            print("\n📊 0 corridas K-fold CV (después de filtros).")

    # ---------- Tabla multi-seed (CV agregada por config × N seeds) ----------
    multi_rows = aggregate_multi_seed(filtered_cvs) if filtered_cvs else []
    if multi_rows:
        print(f"\n📊 {len(multi_rows)} configuración(es) con múltiples seeds (CV × N seeds)\n")
        print(print_table(multi_rows, columns=MULTI_SEED_COLUMNS, sort_by=args.multi_sort_by))
        print("\n  Leyenda: test_acc_mean ± test_acc_std reportado sobre n_runs = n_seeds × n_folds puntos.")
        print("           seed_std = variabilidad ENTRE seeds (alto → método sensible a inicialización).")
        md_sections.append("## CV multi-seed (mean ± std sobre N seeds × K folds)\n\n" +
                           to_markdown(multi_rows, columns=MULTI_SEED_COLUMNS, sort_by=args.multi_sort_by))

    # ---------- Persistencia ----------
    if not args.no_files:
        csv_path = Path(args.csv)
        cv_csv_path = Path(args.cv_csv)
        multi_csv_path = Path(args.multi_csv)
        md_path = Path(args.md)
        if filtered_runs:
            to_csv(to_rows(filtered_runs), csv_path)
            print(f"\n📁 CSV single-split: {csv_path.resolve()}")
        if filtered_cvs:
            to_csv(to_cv_rows(filtered_cvs), cv_csv_path, columns=CV_COLUMNS)
            print(f"📁 CSV CV:           {cv_csv_path.resolve()}")
        if multi_rows:
            to_csv(multi_rows, multi_csv_path, columns=MULTI_SEED_COLUMNS)
            print(f"📁 CSV multi-seed:   {multi_csv_path.resolve()}")
        if md_sections:
            md_path.write_text("\n".join(md_sections))
            print(f"📁 MD:               {md_path.resolve()}")

    # ---------- Plots ----------
    if args.plot:
        n_ok = 0
        # Single-split runs
        for r in filtered_runs:
            run_dir = output_dir / Path(r["__path__"])
            png = run_dir / "learning_curve.png"
            if plot_learning_curve(r, png):
                n_ok += 1
        # Por-fold runs dentro de CV (cargados por separado para plot)
        if filtered_cvs:
            for cv in filtered_cvs:
                cv_dir = output_dir / Path(cv["__path__"])
                for fold_run_path in cv_dir.glob("fold*/run.json"):
                    try:
                        with open(fold_run_path) as f:
                            fold_run = json.load(f)
                        png = fold_run_path.parent / "learning_curve.png"
                        if plot_learning_curve(fold_run, png):
                            n_ok += 1
                    except Exception:
                        pass
        if n_ok:
            print(f"\n📈 {n_ok} learning_curve.png generadas.")
        else:
            print("\n⚠  Ninguna corrida tenía datos de learning_curve.")

    if args.plot_comparison:
        comp_path = Path("./comparison_by_dataset.png").resolve()
        if plot_comparison_by_dataset(filtered_runs, comp_path):
            print(f"📈 Comparativo por dataset: {comp_path}")
        else:
            print("⚠  No se pudo generar plot comparativo (matplotlib o datos faltantes).")


if __name__ == "__main__":
    main()

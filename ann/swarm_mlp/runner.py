"""CLI para correr una técnica × un dataset × una configuración.

Soporta tres modos:
  - Single-split (default): un split 70/15/15 fijo + N sesiones con cadena monótona.
  - Splits repetidos (`--data-seed S1 S2 ...` con 2+ semillas): PROTOCOLO DEL PAPER.
    Una corrida completa (N sesiones) por semilla de split, reporte agregado con
    media ± std sobre los splits. Cada split es 70/15/15 estratificado independiente.
  - K-fold CV (`--cv-folds K` con K>1): K folds estratificados, una corrida por fold,
    reporte agregado con media ± std. Modo extra (folds ≈ 68/12/20, no 70/15/15).

Todos los modos son sin data leakage (scaler/reductor fit sólo en train).
"""
import argparse
import json
import statistics
from pathlib import Path
import sys
import numpy as np
import torch

# Permitir ejecutar desde cualquier directorio.
# _HERE  → resuelve core/, swarm/ y gradient/ (locales al paquete)
# _ANN   → resuelve tensor_qpso/ (compartido a nivel de ann/, única fuente de verdad)
_HERE = Path(__file__).parent.resolve()
_ANN = _HERE.parent
for _p in (str(_HERE), str(_ANN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.data import prepare_dataset, prepare_dataset_kfold, BENCHMARK_DATASETS, PAPER_DATASETS
from core.mlp import MLP
from core.sessions import run_sessions
from swarm.multi_swarm.t1_sequential import run_session_t1
from swarm.multi_swarm.t2_async import run_session_t2_async
from swarm.multi_swarm.t2_jacobi import run_session_t2_jacobi
from swarm.multi_swarm.t3_concurrency import run_session_t3
from swarm.one_swarm.t4_single_swarm import run_session_t4
from tensor_qpso import get_device


TECHNIQUES = {
    "t1": run_session_t1,
    "t2": run_session_t2_async,
    "t2-jacobi": run_session_t2_jacobi,
    "t3": run_session_t3,
    "t4": run_session_t4,
}


def parse_args():
    p = argparse.ArgumentParser(description="Multi-swarm MLP via QDPSO (PyTorch)")
    p.add_argument("--technique", required=True, choices=sorted(TECHNIQUES.keys()))
    p.add_argument("--dataset", required=True,
                   choices=sorted(set(BENCHMARK_DATASETS) | set(PAPER_DATASETS)),
                   help="Suite del paper: " + ", ".join(PAPER_DATASETS)
                        + ". Extra (fuera del paper): wine, circle.")
    p.add_argument("--n-sessions", type=int, default=15,
                   help="Sesiones internas con cadena monótona inter-sesión y selección por val. "
                        "En modo CV (--cv-folds K), se ejecutan POR FOLD (recomendado N=1 para CV).")
    p.add_argument("--cv-folds", type=int, default=1,
                   help="K folds de cross-validation estratificada. Default 1 = split fijo 70/15/15. "
                        "Con K>1: cada fold ~ n_total/K muestras de test, sin data leakage. Reporta media ± std. "
                        "Recomendado K=5 para datasets pequeños.")
    p.add_argument("--n-rounds", type=int, default=1,
                   help="(T1, T2-jacobi) Sub-rondas de block coordinate descent dentro de cada sesión. "
                        "Default 1. Sólo afecta a T1/T2-jacobi — T2 async, T3 y T4 lo ignoran.")
    p.add_argument("--n-particles", type=int, default=50)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--warm-particles", type=int, default=10,
                   help="(TODAS las técnicas) Cantidad de partículas inicializadas como cluster alrededor "
                        "de params_init/current (1 seed exacto + K-1 con ruido). Default 10. "
                        "Reduce dramáticamente la varianza por cold start.")
    p.add_argument("--warm-noise", type=float, default=0.05,
                   help="(TODAS las técnicas) Escala del ruido para el cluster warm-start, relativa al rango de bounds. "
                        "Default 0.05.")
    p.add_argument("--hidden-sizes", type=int, nargs="*", default=None,
                   help="Tamaños de capas ocultas. Default: [3 * n_features]")
    p.add_argument("--lambda-l2", type=float, default=10.0,
                   help="Coeficiente del L2 normalizado (l2 = (lambda/2)·||W||²/(d_total·n_train)). "
                        "Default 10.0 con la nueva normalización; valores razonables ∈ [1.0, 100.0]. "
                        "OJO: el lambda viejo (1e-3, sin normalización) NO es comparable a este.")
    p.add_argument("--g", type=float, default=0.96)
    p.add_argument("--bounds", type=float, nargs=2, default=[-1.0, 1.0])
    p.add_argument("--seed", type=int, default=42, help="Semilla base (se deriva una distinta por sesión)")
    p.add_argument("--data-seed", type=int, nargs="+", default=[100],
                   help="Semilla(s) del split de datos. Un valor (default 100) = split fijo. "
                        "Varios valores = splits 70/15/15 repetidos (protocolo del paper): "
                        "una corrida completa por semilla + agregado media ± std. "
                        "Excluyente con --cv-folds > 1.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--boundary-strategy", default="clamp",
                   choices=["none", "clamp", "reflect", "wrap", "random"])
    p.add_argument("--tol", type=float, default=1e-12)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--output-dir", default=str(_HERE / "output"))
    return p.parse_args()


def _build_cfg_and_run(
    *, args, device, dtype, data, output_dir, data_seed,
):
    """Construye el MLP y la cfg para un split dado y ejecuta run_sessions."""
    hidden = args.hidden_sizes if args.hidden_sizes else [3 * data["n_features"]]
    layer_sizes = [data["n_features"]] + list(hidden) + [data["n_classes"]]
    mlp = MLP(layer_sizes=layer_sizes, device=device, dtype=dtype)
    print(f"🧠 MLP layers: {layer_sizes}   params={mlp.d_total}   swarms={mlp.num_layers}")

    cfg = {
        "technique": args.technique,
        "dataset": args.dataset,
        "layer_sizes": layer_sizes,
        "n_sessions": args.n_sessions,
        "n_rounds": args.n_rounds,
        "n_particles": args.n_particles,
        "max_iter": args.max_iter,
        "lambda_l2": args.lambda_l2,
        "g": args.g,
        "bounds": tuple(args.bounds),
        "seed": args.seed,
        "data_seed": data_seed,
        "device": args.device,
        "device_resolved": str(device),
        "dtype": args.dtype,
        "boundary_strategy": args.boundary_strategy,
        "tol": args.tol,
        "patience": args.patience,
        "warm_particles": args.warm_particles,
        "warm_noise": args.warm_noise,
        "cv_folds": args.cv_folds,
        "fold_idx": data.get("fold_idx", None),
    }

    return run_sessions(
        technique_fn=TECHNIQUES[args.technique],
        mlp=mlp,
        data=data,
        cfg=cfg,
        output_dir=output_dir,
    )


def _stat_summary(values):
    """Devuelve (mean, std, min, max) de una lista numérica."""
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(statistics.mean(values)),
        "std":  float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min":  float(min(values)),
        "max":  float(max(values)),
        "n":    int(len(values)),
    }


def main():
    args = parse_args()

    device = get_device(args.device)
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]

    # Pre-warm cuBLAS para evitar el UserWarning del primer matmul desde un thread.
    if device.type == "cuda":
        _warm = torch.zeros(2, 2, device=device, dtype=dtype) @ torch.zeros(2, 2, device=device, dtype=dtype)
        del _warm
        torch.cuda.synchronize()

    data_seeds = list(args.data_seed)
    if args.cv_folds > 1 and len(data_seeds) > 1:
        raise ValueError(
            "--cv-folds > 1 y múltiples --data-seed son modos excluyentes: "
            "use splits repetidos (varios --data-seed) O cross-validation (--cv-folds K)."
        )

    if args.cv_folds > 1:
        mode = f"CV {args.cv_folds} folds"
    elif len(data_seeds) > 1:
        mode = f"splits repetidos 70/15/15 × {len(data_seeds)} data_seeds (protocolo del paper)"
    else:
        mode = "single-split 70/15/15"

    print(f"⚙  Device: {device}   dtype: {dtype}")
    print(f"⚙  Semillas: base={args.seed}  data={data_seeds}")
    print(f"⚙  Modo: {mode}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hidden_for_tag = args.hidden_sizes if args.hidden_sizes else None
    # No conocemos n_features hasta cargar; el tag se completa abajo.

    base_dir = Path(args.output_dir) / args.technique / args.dataset

    # ---------- Ruta single-split / splits repetidos ----------
    if args.cv_folds == 1:
        # -- Un solo data_seed: comportamiento clásico, mismo output dir. --
        if len(data_seeds) == 1:
            ds = data_seeds[0]
            data = prepare_dataset(args.dataset, device=device, dtype=dtype, data_seed=ds)
            print(f"📊 Dataset {args.dataset}: {data['n_features']} features, {data['n_classes']} clases")
            print(f"📊 Splits: train={data['X_train'].shape[0]}  "
                  f"val={data['X_val'].shape[0]}  test={data['X_test'].shape[0]}")

            hidden = hidden_for_tag if hidden_for_tag else [3 * data["n_features"]]
            hidden_tag = "-".join(map(str, hidden))
            rounds_tag = f"_rounds{args.n_rounds}" if args.technique in ("t1", "t2-jacobi") and args.n_rounds > 1 else ""
            output_dir = base_dir / (
                f"hid{hidden_tag}_iters{args.max_iter}_part{args.n_particles}{rounds_tag}_seed{args.seed}"
            )

            _build_cfg_and_run(
                args=args, device=device, dtype=dtype, data=data,
                output_dir=output_dir, data_seed=ds,
            )
            return

        # -- Varios data_seed: splits 70/15/15 repetidos (protocolo del paper). --
        # Una corrida completa (N sesiones + ganadora por val) por semilla de split;
        # agregado media ± std sobre los splits en splits_summary.json.
        probe = prepare_dataset(args.dataset, device=device, dtype=dtype, data_seed=data_seeds[0])
        n_features = probe["n_features"]
        n_classes = probe["n_classes"]
        hidden = hidden_for_tag if hidden_for_tag else [3 * n_features]
        hidden_tag = "-".join(map(str, hidden))
        rounds_tag = f"_rounds{args.n_rounds}" if args.technique in ("t1", "t2-jacobi") and args.n_rounds > 1 else ""
        splits_root = base_dir / f"splits{len(data_seeds)}" / (
            f"hid{hidden_tag}_iters{args.max_iter}_part{args.n_particles}{rounds_tag}_seed{args.seed}"
        )
        splits_root.mkdir(parents=True, exist_ok=True)

        print(f"📊 Dataset {args.dataset}: {n_features} features, {n_classes} clases")
        print(f"📊 Splits repetidos: {len(data_seeds)} data_seeds {data_seeds} (output: {splits_root})")

        split_summaries = []
        for i, ds in enumerate(data_seeds):
            print(f"\n{'═' * 70}")
            print(f"📁 SPLIT {i + 1}/{len(data_seeds)}  (data_seed={ds})")
            print('═' * 70)

            data_split = (probe if i == 0 else
                          prepare_dataset(args.dataset, device=device, dtype=dtype, data_seed=ds))
            print(f"📊 Splits dseed {ds}: "
                  f"train={data_split['X_train'].shape[0]}  "
                  f"val={data_split['X_val'].shape[0]}  "
                  f"test={data_split['X_test'].shape[0]}")

            split_dir = splits_root / f"dseed{ds}"
            result = _build_cfg_and_run(
                args=args, device=device, dtype=dtype,
                data=data_split, output_dir=split_dir, data_seed=ds,
            )

            winner = result["winner_by_val"]
            split_summaries.append({
                "data_seed": ds,
                "test_acc":  winner["final_test_acc"],
                "test_cost": winner["final_test_cost"],
                "test_score": winner["final_test_score"],
                "val_acc":   winner["val_acc"],
                "val_cost":  winner["val_cost"],
                "val_score": winner["val_score"],
                "train_loss": winner["train_loss"],
                "session":   winner["session"],
                "total_time": result["total_time"],
                "n_train":  int(data_split["X_train"].shape[0]),
                "n_val":    int(data_split["X_val"].shape[0]),
                "n_test":   int(data_split["X_test"].shape[0]),
            })

        # ---------- Agregado sobre splits ----------
        test_accs    = [s["test_acc"]   for s in split_summaries]
        test_costs   = [s["test_cost"]  for s in split_summaries]
        val_accs     = [s["val_acc"]    for s in split_summaries]
        train_losses = [s["train_loss"] for s in split_summaries]
        times        = [s["total_time"] for s in split_summaries]

        splits_summary = {
            "config": {
                "technique": args.technique,
                "dataset": args.dataset,
                "data_seeds": data_seeds,
                "n_sessions_per_split": args.n_sessions,
                "n_rounds": args.n_rounds,
                "n_particles": args.n_particles,
                "max_iter": args.max_iter,
                "lambda_l2": args.lambda_l2,
                "seed": args.seed,
                "layer_sizes": [n_features] + list(hidden) + [n_classes],
                "device": str(device),
            },
            "splits": split_summaries,
            "aggregated": {
                "test_acc":   _stat_summary(test_accs),
                "test_cost":  _stat_summary(test_costs),
                "val_acc":    _stat_summary(val_accs),
                "train_loss": _stat_summary(train_losses),
                "total_time": _stat_summary(times),
            },
        }

        summary_path = splits_root / "splits_summary.json"
        with open(summary_path, "w") as f:
            json.dump(splits_summary, f, indent=2)

        agg = splits_summary["aggregated"]
        print(f"\n{'═' * 70}")
        print(f"🏁 SPLITS REPETIDOS ×{len(data_seeds)} RESUMEN  ({args.technique} / {args.dataset})")
        print('═' * 70)
        print(f"  test_acc:   {agg['test_acc']['mean']:.4f} ± {agg['test_acc']['std']:.4f}   "
              f"(min={agg['test_acc']['min']:.4f}, max={agg['test_acc']['max']:.4f})")
        print(f"  val_acc:    {agg['val_acc']['mean']:.4f} ± {agg['val_acc']['std']:.4f}")
        print(f"  train_loss: {agg['train_loss']['mean']:.4f} ± {agg['train_loss']['std']:.4f}")
        print(f"  ⏱  total:   {sum(times):.2f}s   (mean/split: {agg['total_time']['mean']:.2f}s)")
        print(f"  📁 {summary_path}")
        print('═' * 70)
        return

    # ---------- Ruta K-fold CV ----------
    if args.cv_folds < 2:
        raise ValueError(f"--cv-folds debe ser 1 (single-split) o >=2 (CV). Recibido: {args.cv_folds}")

    cv_data_seed = data_seeds[0]

    # Carga preliminar para conocer n_features y armar el tag del output dir.
    probe = prepare_dataset_kfold(
        args.dataset, k=args.cv_folds, fold_idx=0,
        device=device, dtype=dtype, data_seed=cv_data_seed,
    )
    n_features = probe["n_features"]
    n_classes = probe["n_classes"]
    hidden = hidden_for_tag if hidden_for_tag else [3 * n_features]
    hidden_tag = "-".join(map(str, hidden))
    rounds_tag = f"_rounds{args.n_rounds}" if args.technique in ("t1", "t2-jacobi") and args.n_rounds > 1 else ""
    cv_root = base_dir / f"cv{args.cv_folds}" / (
        f"hid{hidden_tag}_iters{args.max_iter}_part{args.n_particles}{rounds_tag}_seed{args.seed}"
    )
    cv_root.mkdir(parents=True, exist_ok=True)

    print(f"📊 Dataset {args.dataset}: {n_features} features, {n_classes} clases")
    print(f"📊 K-fold CV: {args.cv_folds} folds (output: {cv_root})")

    fold_summaries = []
    for fold_idx in range(args.cv_folds):
        print(f"\n{'═' * 70}")
        print(f"📁 FOLD {fold_idx + 1}/{args.cv_folds}")
        print('═' * 70)

        data_fold = prepare_dataset_kfold(
            args.dataset, k=args.cv_folds, fold_idx=fold_idx,
            device=device, dtype=dtype, data_seed=cv_data_seed,
        )
        print(f"📊 Splits fold {fold_idx}: "
              f"train={data_fold['X_train'].shape[0]}  "
              f"val={data_fold['X_val'].shape[0]}  "
              f"test={data_fold['X_test'].shape[0]}")

        fold_dir = cv_root / f"fold{fold_idx}"
        result = _build_cfg_and_run(
            args=args, device=device, dtype=dtype,
            data=data_fold, output_dir=fold_dir, data_seed=cv_data_seed,
        )

        winner = result["winner_by_val"]
        fold_summaries.append({
            "fold_idx": fold_idx,
            "test_acc":  winner["final_test_acc"],
            "test_cost": winner["final_test_cost"],
            "test_score": winner["final_test_score"],
            "val_acc":   winner["val_acc"],
            "val_cost":  winner["val_cost"],
            "val_score": winner["val_score"],
            "train_loss": winner["train_loss"],
            "session":   winner["session"],
            "total_time": result["total_time"],
            "n_train":  int(data_fold["X_train"].shape[0]),
            "n_val":    int(data_fold["X_val"].shape[0]),
            "n_test":   int(data_fold["X_test"].shape[0]),
        })

    # ---------- Agregado CV ----------
    test_accs   = [f["test_acc"]   for f in fold_summaries]
    test_costs  = [f["test_cost"]  for f in fold_summaries]
    val_accs    = [f["val_acc"]    for f in fold_summaries]
    train_losses = [f["train_loss"] for f in fold_summaries]
    times       = [f["total_time"] for f in fold_summaries]

    cv_summary = {
        "config": {
            "technique": args.technique,
            "dataset": args.dataset,
            "cv_folds": args.cv_folds,
            "n_sessions_per_fold": args.n_sessions,
            "n_rounds": args.n_rounds,
            "n_particles": args.n_particles,
            "max_iter": args.max_iter,
            "lambda_l2": args.lambda_l2,
            "seed": args.seed,
            "data_seed": cv_data_seed,
            "layer_sizes": [n_features] + list(hidden) + [n_classes],
            "device": str(device),
        },
        "folds": fold_summaries,
        "aggregated": {
            "test_acc":   _stat_summary(test_accs),
            "test_cost":  _stat_summary(test_costs),
            "val_acc":    _stat_summary(val_accs),
            "train_loss": _stat_summary(train_losses),
            "total_time": _stat_summary(times),
        },
    }

    summary_path = cv_root / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(cv_summary, f, indent=2)

    agg = cv_summary["aggregated"]
    print(f"\n{'═' * 70}")
    print(f"🏁 CV {args.cv_folds}-FOLD RESUMEN  ({args.technique} / {args.dataset})")
    print('═' * 70)
    print(f"  test_acc:   {agg['test_acc']['mean']:.4f} ± {agg['test_acc']['std']:.4f}   "
          f"(min={agg['test_acc']['min']:.4f}, max={agg['test_acc']['max']:.4f})")
    print(f"  val_acc:    {agg['val_acc']['mean']:.4f} ± {agg['val_acc']['std']:.4f}")
    print(f"  train_loss: {agg['train_loss']['mean']:.4f} ± {agg['train_loss']['std']:.4f}")
    print(f"  ⏱  total:   {sum(times):.2f}s   (mean/fold: {agg['total_time']['mean']:.2f}s)")
    print(f"  📁 {summary_path}")
    print('═' * 70)


if __name__ == "__main__":
    main()

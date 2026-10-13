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
from pathlib import Path
import sys
import numpy as np
import torch

# Permitir ejecutar desde cualquier directorio.
# _HERE → resuelve core/, swarm/ y gradient/ (locales al paquete).
# El optimizador (QPSO/QDPSO) viene del paquete instalado `qpso-torch`.
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core.budget import evals_per_session, iters_for_budget
from core.data import prepare_dataset, prepare_dataset_kfold, BENCHMARK_DATASETS, PAPER_DATASETS
from core.mlp import MLP
from core.sessions import run_sessions
from core.stats import describe
from swarm.multi_swarm.t1_sequential import run_session_t1
from swarm.multi_swarm.t2_async import run_session_t2_async
from swarm.multi_swarm.t2_jacobi import run_session_t2_jacobi
from swarm.multi_swarm.t3_concurrency import run_session_t3
from swarm.one_swarm.t4_single_swarm import run_session_t4
from swarm.optimizer_factory import (
    DEFAULT_ALPHA, DEFAULT_G, OPTIMIZERS, optimizer_tag, resolve_alpha,
)
from qpso_torch import get_device


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
                        + ". Extra (fuera del paper): " + ", ".join(BENCHMARK_DATASETS) + ".")
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
    p.add_argument("--eval-budget", type=int, default=None,
                   help="Presupuesto de evaluaciones de la función de costo POR SESIÓN "
                        "(una evaluación = una partícula sobre todo el conjunto de fitness). "
                        "Si se da, reemplaza a --max-iter: cada técnica recibe las iteraciones "
                        "que caben en el presupuesto, de modo que todas se comparan a igual costo. "
                        "A igual --max-iter las técnicas multi-swarm gastan varias veces más "
                        "evaluaciones que T4 (un enjambre por capa; T2/T3 evalúan dos veces por paso).")
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
    p.add_argument("--optimizer", default="qdpsoo", choices=list(OPTIMIZERS),
                   help="Optimizador swarm (eje del paper). "
                        "qdpsoo = factor CE ESTÁTICO `g` (QDPSOTensorOptimized). "
                        "qpsoo = factor CE CALCULADO, alpha con decaimiento lineal "
                        "(QPSOTensorOptimized). Default qdpsoo.")
    p.add_argument("--g", type=float, default=DEFAULT_G,
                   help=f"(qdpsoo) Factor de contracción-expansión estático. "
                        f"Default {DEFAULT_G} = default del paper. "
                        "OJO: el default histórico era 0.96 y los resultados preliminares "
                        "1-seed se corrieron con ese valor — no son comparables. "
                        "Candidato a re-tuneo: el análisis de sensibilidad barre "
                        "g ∈ {1.13, 1.20, 1.30, 1.40, 1.50}.")
    p.add_argument("--alpha", type=float, nargs="+", default=list(DEFAULT_ALPHA),
                   metavar=("MAX", "MIN"),
                   help="(qpsoo) Factor CE calculado: par (max, min) con decaimiento "
                        f"lineal alpha(t) = max - (max-min)·t/T. Default {DEFAULT_ALPHA[0]} "
                        f"{DEFAULT_ALPHA[1]}. Un solo valor = alpha constante (sin decay). "
                        "Ignorado con --optimizer qdpsoo.")
    p.add_argument("--reevaluate-pbest", action=argparse.BooleanOptionalAction, default=True,
                   help="(T2, T3) Re-evaluar los pbest contra la cost function fresca de cada "
                        "paso (el contexto de las otras capas cambió). Default: activado. "
                        "Cuesta una evaluación extra del swarm por paso. "
                        "--no-reevaluate-pbest reproduce el comportamiento previo (valores "
                        "obsoletos) para la ablación. T1, T2-jacobi y T4 lo ignoran.")
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

    budget_kwargs = dict(
        n_particles=args.n_particles, n_layers=mlp.num_layers,
        warm_particles=args.warm_particles, n_rounds=args.n_rounds,
        reevaluate_pbest=args.reevaluate_pbest,
    )
    max_iter = args.max_iter
    if args.eval_budget is not None:
        max_iter = iters_for_budget(args.technique, args.eval_budget, **budget_kwargs)
    planned_evals = evals_per_session(args.technique, max_iter, **budget_kwargs)
    print(f"💰 max_iter={max_iter}  →  {planned_evals} evaluaciones de fitness por sesión"
          + (f"  (presupuesto {args.eval_budget})" if args.eval_budget is not None else ""))

    cfg = {
        "technique": args.technique,
        "dataset": args.dataset,
        "layer_sizes": layer_sizes,
        "n_sessions": args.n_sessions,
        "n_rounds": args.n_rounds,
        "n_particles": args.n_particles,
        "max_iter": max_iter,
        "eval_budget": args.eval_budget,
        "planned_evals_per_session": planned_evals,
        "lambda_l2": args.lambda_l2,
        "optimizer": args.optimizer,
        "g": args.g,
        "alpha": resolve_alpha(args.alpha),
        "reevaluate_pbest": args.reevaluate_pbest,
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


def _run_tag(args, hidden) -> str:
    """Nombre del directorio de una corrida.

    El optimizador va SIEMPRE en el tag (no sólo cuando difiere del default) para que
    las celdas del factorial tengan paths inequívocos: sin esto una corrida qpsoo
    sobrescribiría a su equivalente qdpsoo.
    """
    hidden_tag = "-".join(map(str, hidden))
    rounds_tag = (f"_rounds{args.n_rounds}"
                  if args.technique in ("t1", "t2-jacobi") and args.n_rounds > 1 else "")
    opt_tag = optimizer_tag({"optimizer": args.optimizer})
    # Sólo se marca la ablación (no-default) y sólo donde aplica, para no cambiar
    # los paths del resto de las celdas.
    reeval_tag = ("_noreeval"
                  if args.technique in ("t2", "t3") and not args.reevaluate_pbest else "")
    # Hiperparámetros barridos en el análisis de sensibilidad: sin esto, dos valores
    # de g (o de alpha, o de lambda) escribirían en el mismo directorio.
    hp_tag = ""
    if args.optimizer == "qdpsoo" and args.g != DEFAULT_G:
        hp_tag += f"_g{args.g:g}"
    if args.optimizer == "qpsoo" and tuple(args.alpha) != tuple(DEFAULT_ALPHA):
        hp_tag += "_a" + "-".join(f"{a:g}" for a in args.alpha)
    if args.lambda_l2 != 10.0:
        hp_tag += f"_l2-{args.lambda_l2:g}"
    # Con presupuesto de evaluaciones, `max_iter` depende de la técnica y de la red:
    # lo que identifica a la corrida es el presupuesto.
    budget_tag = (f"evals{args.eval_budget}" if args.eval_budget is not None
                  else f"iters{args.max_iter}")
    return (f"hid{hidden_tag}_{budget_tag}_part{args.n_particles}"
            f"{rounds_tag}_{opt_tag}{hp_tag}{reeval_tag}_seed{args.seed}")


def _run_summary(result):
    """Campos de una corrida que entran al resumen de splits repetidos / K-fold."""
    winner = result["winner_by_val"]
    return {
        "test_acc":  winner["final_test_acc"],
        "test_ce":   winner["final_test_ce"],
        "test_cost": winner["final_test_cost"],
        "test_macro_f1": winner["final_test_macro_f1"],
        "test_balanced_acc": winner["final_test_balanced_acc"],
        "val_acc":   winner["val_acc"],
        "val_ce":    winner["val_ce"],
        "val_cost":  winner["val_cost"],
        "train_loss": winner["train_loss"],
        "train_acc": winner["train_acc"],
        "session":   winner["session"],
        "total_time": result["total_time"],
        "time_to_winner": winner["time_to_winner"],
        "fitness_evals": result["total_fitness_evals"],
        "fitness_evals_to_winner": winner["fitness_evals_to_winner"],
        "gpu_peak_mb": result["gpu_peak_mb"],
    }


#: Métricas que se agregan sobre splits o folds con `core.stats.describe`
#: (media, std MUESTRAL, IC95, mediana, cuartiles, min, max).
_AGGREGATED = (
    "test_acc", "test_ce", "test_cost", "test_macro_f1", "test_balanced_acc",
    "val_acc", "train_loss", "train_acc", "total_time", "time_to_winner",
    "fitness_evals", "fitness_evals_to_winner",
)


def _aggregate(summaries):
    return {key: describe([r[key] for r in summaries]) for key in _AGGREGATED}


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
            output_dir = base_dir / _run_tag(args, hidden)

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
        splits_root = base_dir / f"splits{len(data_seeds)}" / _run_tag(args, hidden)
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

            split_summaries.append({
                "data_seed": ds,
                **_run_summary(result),
                "n_train":  int(data_split["X_train"].shape[0]),
                "n_val":    int(data_split["X_val"].shape[0]),
                "n_test":   int(data_split["X_test"].shape[0]),
            })

        # ---------- Agregado sobre splits ----------
        times = [r["total_time"] for r in split_summaries]

        splits_summary = {
            "config": {
                "technique": args.technique,
                "dataset": args.dataset,
                "data_seeds": data_seeds,
                "n_sessions_per_split": args.n_sessions,
                "n_rounds": args.n_rounds,
                "n_particles": args.n_particles,
                "max_iter": args.max_iter,
                "eval_budget": args.eval_budget,
                "lambda_l2": args.lambda_l2,
                "optimizer": args.optimizer,
                "g": args.g,
                "alpha": resolve_alpha(args.alpha),
                "reevaluate_pbest": args.reevaluate_pbest,
                "seed": args.seed,
                "layer_sizes": [n_features] + list(hidden) + [n_classes],
                "device": str(device),
            },
            "splits": split_summaries,
            "aggregated": _aggregate(split_summaries),
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
        print(f"  IC95 test_acc: ±{agg['test_acc']['ci95']:.4f}   "
              f"evaluaciones de fitness (media): {agg['fitness_evals']['mean']:.0f}")
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
    cv_root = base_dir / f"cv{args.cv_folds}" / _run_tag(args, hidden)
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

        fold_summaries.append({
            "fold_idx": fold_idx,
            **_run_summary(result),
            "n_train":  int(data_fold["X_train"].shape[0]),
            "n_val":    int(data_fold["X_val"].shape[0]),
            "n_test":   int(data_fold["X_test"].shape[0]),
        })

    # ---------- Agregado CV ----------
    times = [r["total_time"] for r in fold_summaries]

    cv_summary = {
        "config": {
            "technique": args.technique,
            "dataset": args.dataset,
            "cv_folds": args.cv_folds,
            "n_sessions_per_fold": args.n_sessions,
            "n_rounds": args.n_rounds,
            "n_particles": args.n_particles,
            "max_iter": args.max_iter,
            "eval_budget": args.eval_budget,
            "lambda_l2": args.lambda_l2,
            "optimizer": args.optimizer,
            "g": args.g,
            "alpha": resolve_alpha(args.alpha),
            "reevaluate_pbest": args.reevaluate_pbest,
            "seed": args.seed,
            "data_seed": cv_data_seed,
            "layer_sizes": [n_features] + list(hidden) + [n_classes],
            "device": str(device),
        },
        "folds": fold_summaries,
        "aggregated": _aggregate(fold_summaries),
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
    print(f"  IC95 test_acc: ±{agg['test_acc']['ci95']:.4f}   "
          f"evaluaciones de fitness (media): {agg['fitness_evals']['mean']:.0f}")
    print(f"  ⏱  total:   {sum(times):.2f}s   (mean/fold: {agg['total_time']['mean']:.2f}s)")
    print(f"  📁 {summary_path}")
    print('═' * 70)


if __name__ == "__main__":
    main()

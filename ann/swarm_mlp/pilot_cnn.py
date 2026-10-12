"""PILOTO — ¿se puede entrenar una CNN pequeña en Fashion-MNIST con las técnicas del paper?

No forma parte del runner ni del factorial: es un experimento de viabilidad. Reusa
SIN cambios las técnicas (T1..T4), el protocolo de sesiones y la factory del
optimizador; lo único nuevo es `core.cnn.TinyCNN`, que expone la interfaz del MLP.

Datos: la misma submuestra canónica de Fashion-MNIST que usan las variantes
`fmnist-*` (1000 imágenes por clase del split de train, seed 100), pero con los
píxeles crudos 28x28 en [0, 1]. Split estratificado 70/15/15.

Baseline: Adam sobre la MISMA arquitectura y los mismos splits.

Uso:
    cd ann/swarm_mlp
    python pilot_cnn.py --techniques t4 t3 --optimizer qdpsoo
    python pilot_cnn.py --adam-only
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core.cnn import TinyCNN
from core.data import split_70_15_15
from core.features import (
    DEFAULT_CACHE_DIR, FMNIST_PER_CLASS, FMNIST_SUBSAMPLE_SEED,
)
from core.loss import cross_entropy_from_logits
from core.sessions import run_sessions
from swarm.multi_swarm.t1_sequential import run_session_t1
from swarm.multi_swarm.t2_async import run_session_t2_async
from swarm.multi_swarm.t2_jacobi import run_session_t2_jacobi
from swarm.multi_swarm.t3_concurrency import run_session_t3
from swarm.one_swarm.t4_single_swarm import run_session_t4
from swarm.optimizer_factory import DEFAULT_ALPHA, DEFAULT_G, OPTIMIZERS
from qpso_torch import get_device

TECHNIQUES = {
    "t1": run_session_t1,
    "t2": run_session_t2_async,
    "t2-jacobi": run_session_t2_jacobi,
    "t3": run_session_t3,
    "t4": run_session_t4,
}


def load_fmnist_images(device, dtype, data_seed: int = 100, per_class: int = FMNIST_PER_CLASS):
    """Submuestra canónica de Fashion-MNIST como imágenes (N, 1, 28, 28) en [0, 1]."""
    from torchvision.datasets import FashionMNIST

    tv_root = DEFAULT_CACHE_DIR / "torchvision"
    tv_root.mkdir(parents=True, exist_ok=True)
    ds = FashionMNIST(root=str(tv_root), train=True, download=True)
    images = ds.data.numpy()
    labels = ds.targets.numpy()

    # Misma selección que core.features.load_fmnist_raw.
    rng = np.random.default_rng(FMNIST_SUBSAMPLE_SEED)
    idx = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(labels == c), size=per_class, replace=False)
        for c in range(10)
    ]))
    X = images[idx].astype(np.float32)[:, None, :, :] / 255.0
    y = labels[idx].astype(np.int64)

    X_train, y_train, X_val, y_val, X_test, y_test = split_70_15_15(X, y, data_seed=data_seed)

    def f(a):
        return torch.as_tensor(a, dtype=dtype, device=device)

    def l(a):
        return torch.as_tensor(a, dtype=torch.long, device=device)

    return {
        "X_train": f(X_train), "y_train": l(y_train),
        "X_val": f(X_val), "y_val": l(y_val),
        "X_test": f(X_test), "y_test": l(y_test),
        "n_features": int(X.shape[-1] * X.shape[-2]),
        "n_classes": 10,
    }


def run_adam(cnn: TinyCNN, data, *, lr: float, epochs: int, batch_size: int, seed: int):
    """Adam sobre la misma arquitectura. Selección por accuracy de validación."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    segments = []
    for spec in cnn.layer_specs:
        fan_in = spec.n_in * max(1, spec.ksize * spec.ksize)
        bound = 1.0 / math.sqrt(fan_in)
        segments.append((torch.rand(spec.size, generator=gen) * 2 - 1) * bound)
    flat = torch.nn.Parameter(torch.cat(segments).to(device=cnn.device, dtype=cnn.dtype))
    opt = torch.optim.Adam([flat], lr=lr)

    X, y = data["X_train"], data["y_train"]
    n = X.shape[0]
    best = {"val_acc": -1.0, "params": None, "epoch": 0}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, generator=gen).to(X.device)
        for start in range(0, n, batch_size):
            b = perm[start : start + batch_size]
            loss = cross_entropy_from_logits(cnn.forward(X[b], flat), y[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val_acc = (cnn.predict(data["X_val"], flat) == data["y_val"]).float().mean().item()
        if val_acc >= best["val_acc"]:
            best = {"val_acc": val_acc, "params": flat.detach().clone(), "epoch": epoch}
    elapsed = time.time() - t0
    with torch.no_grad():
        test_acc = (cnn.predict(data["X_test"], best["params"]) == data["y_test"]).float().mean().item()
        train_acc = (cnn.predict(X, best["params"]) == y).float().mean().item()
    return {"lr": lr, "epochs": epochs, "best_epoch": best["epoch"], "train_acc": train_acc,
            "val_acc": best["val_acc"], "test_acc": test_acc, "time": elapsed}


def parse_args():
    p = argparse.ArgumentParser(description="Piloto CNN + Fashion-MNIST con swarms")
    p.add_argument("--techniques", nargs="*", default=["t4", "t3"], choices=sorted(TECHNIQUES))
    p.add_argument("--optimizer", default="qdpsoo", choices=list(OPTIMIZERS))
    p.add_argument("--g", type=float, default=DEFAULT_G)
    p.add_argument("--alpha", type=float, nargs=2, default=list(DEFAULT_ALPHA))
    p.add_argument("--channels", type=int, nargs="+", default=[8, 16])
    p.add_argument("--head", default="gap", choices=["gap", "flatten"])
    p.add_argument("--n-sessions", type=int, default=3)
    p.add_argument("--n-rounds", type=int, default=1)
    p.add_argument("--n-particles", type=int, default=50)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--lambda-l2", type=float, default=10.0)
    p.add_argument("--bounds", type=float, nargs=2, default=[-1.0, 1.0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-seed", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--fitness-samples", type=int, default=0,
                   help="Imágenes de train que usa el fitness de los swarms (0 = todas). "
                        "El split ya viene mezclado y estratificado, así que se toman las primeras N. "
                        "Adam siempre entrena con todo el train.")
    p.add_argument("--adam-only", action="store_true")
    p.add_argument("--no-adam", action="store_true")
    p.add_argument("--adam-epochs", type=int, default=30)
    p.add_argument("--adam-lrs", type=float, nargs="+", default=[1e-3, 1e-2])
    p.add_argument("--output-dir", default=str(_HERE / "output" / "pilot_cnn"))
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    dtype = torch.float32
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_fmnist_images(device, dtype, data_seed=args.data_seed)
    cnn = TinyCNN(channels=args.channels, head=args.head, device=device, dtype=dtype)
    arch = f"conv{'-'.join(map(str, args.channels))}_{args.head}"
    print(f"⚙  Device: {device}")
    print(f"📊 Fashion-MNIST (imágenes): train={data['X_train'].shape[0]} "
          f"val={data['X_val'].shape[0]} test={data['X_test'].shape[0]}")
    print(f"🧠 CNN {arch}: params={cnn.d_total}  bloques={[(s.kind, s.size) for s in cnn.layer_specs]}")

    summary = []

    if not args.no_adam:
        for lr in args.adam_lrs:
            r = run_adam(cnn, data, lr=lr, epochs=args.adam_epochs, batch_size=128, seed=args.seed)
            print(f"🎓 Adam lr={lr:g}: test_acc={r['test_acc']:.4f} val_acc={r['val_acc']:.4f} "
                  f"train_acc={r['train_acc']:.4f} (mejor época {r['best_epoch']}/{r['epochs']}) "
                  f"⏱ {r['time']:.1f}s")
            summary.append({"method": f"adam_lr{lr:g}", **r})

    if not args.adam_only:
        swarm_data = dict(data)
        if args.fitness_samples > 0:
            swarm_data["X_train"] = data["X_train"][: args.fitness_samples]
            swarm_data["y_train"] = data["y_train"][: args.fitness_samples]
            print(f"🐝 Fitness de los swarms sobre {swarm_data['X_train'].shape[0]} imágenes de train")
        for tech in args.techniques:
            cfg = {
                "technique": tech, "dataset": "fmnist-images", "layer_sizes": cnn.layer_sizes,
                "arch": arch, "n_sessions": args.n_sessions, "n_rounds": args.n_rounds,
                "n_particles": args.n_particles, "max_iter": args.max_iter,
                "lambda_l2": args.lambda_l2, "optimizer": args.optimizer, "g": args.g,
                "alpha": tuple(args.alpha), "reevaluate_pbest": True,
                "bounds": tuple(args.bounds), "seed": args.seed, "data_seed": args.data_seed,
                "device": str(device), "device_resolved": str(device), "dtype": "float32",
                "boundary_strategy": "clamp", "tol": 1e-12, "patience": args.patience,
                "warm_particles": 10, "warm_noise": 0.05,
            }
            out_dir = (Path(args.output_dir) / tech /
                       f"{arch}_iters{args.max_iter}_part{args.n_particles}_{args.optimizer}_seed{args.seed}")
            print(f"\n{'═' * 70}\n▶ {tech.upper()} / {args.optimizer}\n{'═' * 70}")
            cfg["fitness_samples"] = int(swarm_data["X_train"].shape[0])
            res = run_sessions(TECHNIQUES[tech], mlp=cnn, data=swarm_data, cfg=cfg, output_dir=out_dir)
            w = res["winner_by_val"]
            summary.append({
                "method": f"{tech}_{args.optimizer}", "test_acc": w["final_test_acc"],
                "val_acc": w["val_acc"], "train_loss": w["train_loss"],
                "session_test_accs": [s["test_acc"] for s in res["sessions"]],
                "session_times": [s["session_time"] for s in res["sessions"]],
                "time": res["total_time"],
            })

    print(f"\n{'═' * 70}\nRESUMEN  ({arch}, {cnn.d_total} params)\n{'═' * 70}")
    for r in summary:
        extra = ""
        if "session_test_accs" in r:
            extra = ("  por sesión: " + " ".join(f"{a:.3f}" for a in r["session_test_accs"])
                     + f"  ({np.mean(r['session_times']):.1f}s/sesión)")
        print(f"  {r['method']:16s} test_acc={r['test_acc']:.4f}  ⏱ {r['time']:.1f}s{extra}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "adam" if args.adam_only else f"{'-'.join(args.techniques)}_{args.optimizer}"
    with open(out / f"summary_{arch}_{tag}_seed{args.seed}.json", "w") as f:
        json.dump({"args": vars(args), "d_total": cnn.d_total, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()

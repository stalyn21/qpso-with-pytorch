"""
Utilidades compartidas para los runs de resubmisión del paper 1.

Parametriza los main_*.py por CLI sin cambiar la semántica del entrenamiento:
- dataset / épocas / partículas / folds configurables
- semillas reproducibles (random, numpy, torch)
- selección del mejor modelo por VALIDACIÓN (corrige la fuga por test del original;
  --select-by test reproduce el comportamiento legacy)
- nombres de salida únicos por seed/tag para runs multi-semilla
- dump JSON de resultados por fold para los tests estadísticos
"""

import argparse
import json
import os
import random

import numpy as np
import torch

BENCHMARK_DATASETS = ['circle', 'iris', 'wine', 'breast_cancer']


def parse_args(description, pso_variant=False, adam_arch=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--dataset', default='iris', choices=BENCHMARK_DATASETS + ['mcw'])
    p.add_argument('--epochs', type=int, default=None,
                   help='default: 100 benchmark / 1000 mcw (igual que el paper)')
    p.add_argument('--particles', type=int, default=20)
    p.add_argument('--g', type=float, default=1.13,
                   help='coeficiente de contracción-expansión (QDPSO/QDPSOo)')
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--seed', type=int, default=0,
                   help='semilla del algoritmo (init de enjambre/pesos)')
    p.add_argument('--split-seed', type=int, default=100,
                   help='semilla del split train/test y del KFold (paper: 100)')
    p.add_argument('--reduction', default=None, choices=['isomap', 'pca', 'mds'],
                   help='solo mcw')
    p.add_argument('--components', type=int, default=None, help='solo mcw')
    p.add_argument('--select-by', default='val', choices=['val', 'test'],
                   help='criterio del best model entre folds (val corrige la fuga)')
    p.add_argument('--tag', default='', help='sufijo extra para los archivos de salida')
    if pso_variant:
        p.add_argument('--variant', default='PSO_bound', choices=['PSO', 'PSO_bound'])
    if adam_arch:
        p.add_argument('--legacy-arch', action='store_true',
                       help='usa la arquitectura original de Adam [2in, 1.5in, in] '
                            '(por defecto se usa la misma de los metaheurísticos)')
    args = p.parse_args()
    if args.epochs is None:
        args.epochs = 1000 if args.dataset == 'mcw' else 100
    return args


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(args):
    if args.dataset == 'mcw':
        from data.weather_image import load_and_preprocess_mcw
        return load_and_preprocess_mcw(reduction_method=args.reduction,
                                       n_components=args.components,
                                       random_state=args.split_seed)
    from data.benchmarck import load_and_preprocess_data
    return load_and_preprocess_data(args.dataset, split_seed=args.split_seed)


def run_suffix(args, input_shape):
    """Identificador único del run; se usa como config['features_reduction'] para
    que logs, modelos y figuras no se pisen entre seeds/configs."""
    red = f"{args.reduction}_{input_shape}" if args.reduction else f"{input_shape}"
    tag = f"_{args.tag}" if args.tag else ''
    return f"{red}_s{args.seed}{tag}"


def ensure_dirs(optimizer_name):
    for d in (f'./output/{optimizer_name}', f'./models/{optimizer_name}',
              f'./metrics/graphics/loss/{optimizer_name}',
              f'./metrics/graphics/roc/{optimizer_name}', './results'):
        os.makedirs(d, exist_ok=True)


def select_score(val_acc, test_acc, select_by):
    return val_acc if select_by == 'val' else test_acc


def dump_results(optimizer_name, args, config, train_results, val_results,
                 test_results, time_results, extra=None):
    payload = {
        'optimizer': optimizer_name,
        'args': vars(args),
        'config': {k: v for k, v in config.items()
                   if isinstance(v, (int, float, str, list, bool))},
        'train_acc_per_fold': train_results,
        'val_acc_per_fold': val_results,
        'test_acc_per_fold': test_results,
        'train_time_per_fold_s': time_results,
    }
    if extra:
        payload.update(extra)
    # config['features_reduction'] ya contiene el sufijo único del run
    path = f"./results/{args.dataset}_{optimizer_name}_{config['features_reduction']}.json"
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path

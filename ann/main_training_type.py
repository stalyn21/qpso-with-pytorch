"""
Benchmark: Comparativa de Estrategias de Entrenamiento

Este script compara diferentes estrategias de entrenamiento para redes
neuronales usando optimizadores QPSO y QDPSO.

Estrategias comparadas:
    - Forward: Entrenamiento estandar (todos los pesos a la vez)
    - Weighted: Forward con pesos por capa (prioriza capas de salida)
    - Layerwise: Entrenamiento capa por capa (output -> input)

Configuracion:
    - Optimizadores: QPSO y QDPSO
    - Datasets: Iris, Wine, Breast Cancer (sklearn)
    - Cross-validation: 4 folds

Requisitos:
    conda activate pytorch_qpso_gpu
    python ann/main_training_type.py
"""

import torch
import numpy as np
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Agregar path para imports (portable: funciona sin importar el nombre del folder)
sys.path.insert(0, os.path.dirname(__file__))

from models import QPSOCompatibleANN
from optimizers.training_strategies import (
    create_training_strategy,
    StrategyConfig,
    TrainingStrategy,
    TrainingResult
)
from utils import (
    MulticlassMetrics,
    plot_confusion_matrix,
    plot_training_history,
    plot_loss_curves,
    plot_accuracy_curves,
    plot_complete_training_summary,
)

# Sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURACION
# =============================================================================

# Configuracion general
GENERAL_CONFIG = {
    'n_particles': 30,
    'max_iters': 100,
    'patience': 30,
    'alpha': (1.0, 0.5),  # QPSO
    'g': 0.96,            # QDPSO
    'weight_bound': 1.0,
    'random_state': 42,
}

# Configuracion por estrategia
STRATEGY_CONFIGS = {
    'forward': {
        # Sin configuracion adicional
    },
    'weighted': {
        'layer_decay': 0.7,       # Decaimiento entre capas
        'regularization': 0.01,   # Factor de regularizacion
    },
    'layerwise': {
        'iters_per_layer': 30,    # Iteraciones por capa
        'fine_tune_iters': 30,    # Fine-tuning final
    }
}

# Estrategias a comparar
STRATEGIES = ['forward', 'weighted', 'layerwise']

# Optimizadores a usar
OPTIMIZERS = ['QPSO', 'QDPSO']

# Datasets
DATASETS = ['iris', 'wine', 'breast_cancer']

# Split
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Directorio de salida
IMG_OUTPUT_DIR = './img/metric/TrainingType'


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def print_header(title: str, char: str = "="):
    """Imprime encabezado formateado."""
    width = 70
    print("\n" + char * width)
    print(f" {title}")
    print(char * width)


def print_subheader(title: str):
    """Imprime subencabezado."""
    print(f"\n--- {title} ---")


def ensure_output_dir():
    """Crea directorio de salida."""
    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)


def get_architecture(input_dim: int, output_dim: int) -> List[int]:
    """Genera arquitectura de red."""
    return [input_dim * 3, input_dim * 2]


def load_and_split_dataset(
    name: str,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int
) -> tuple:
    """Carga y divide dataset."""
    loaders = {
        'iris': datasets.load_iris,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer,
    }

    data = loaders[name]()
    X, y = data.data, data.target

    # Normalizar
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split train / val
    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def print_system_info():
    """Imprime informacion del sistema."""
    print_header("INFORMACION DEL SISTEMA")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return device


def get_image_filename(
    dataset: str,
    optimizer: str,
    strategy: str,
    plot_type: str
) -> str:
    """Genera nombre de archivo para graficas."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{optimizer}_{strategy}_{dataset}_{plot_type}_{timestamp}.png"


# =============================================================================
# BENCHMARK DE ESTRATEGIA
# =============================================================================

def run_strategy_benchmark(
    dataset_name: str,
    strategy_name: str,
    optimizer_name: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Ejecuta benchmark para una estrategia especifica.

    Args:
        dataset_name: Nombre del dataset
        strategy_name: Nombre de la estrategia
        optimizer_name: 'QPSO' o 'QDPSO'
        verbose: Imprimir progreso

    Returns:
        Diccionario con resultados
    """
    use_qdpso = (optimizer_name == 'QDPSO')

    print_subheader(f"{optimizer_name} + {strategy_name.upper()} en {dataset_name}")

    # 1. Cargar datos
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_dataset(
        dataset_name,
        TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
        GENERAL_CONFIG['random_state']
    )

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    hidden_layers = get_architecture(input_dim, output_dim)

    if verbose:
        print(f"  Dataset: {dataset_name}")
        print(f"  Arquitectura: {input_dim} -> {hidden_layers} -> {output_dim}")
        print(f"  Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)}")

    # 2. Crear modelo
    model = QPSOCompatibleANN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        activation='tanh',
        output_activation='softmax'
    )

    n_params = model.num_params
    if verbose:
        print(f"  Parametros: {n_params:,}")

    # 3. Crear configuracion de estrategia
    strategy_specific = STRATEGY_CONFIGS.get(strategy_name, {})

    config = StrategyConfig(
        n_particles=GENERAL_CONFIG['n_particles'],
        max_iters=GENERAL_CONFIG['max_iters'],
        alpha=GENERAL_CONFIG['alpha'],
        g=GENERAL_CONFIG['g'],
        weight_bound=GENERAL_CONFIG['weight_bound'],
        patience=GENERAL_CONFIG['patience'],
        seed=GENERAL_CONFIG['random_state'],
        **strategy_specific
    )

    # 4. Crear estrategia
    strategy = create_training_strategy(
        model=model,
        strategy=strategy_name,
        config=config,
        use_qdpso=use_qdpso
    )

    # 5. Establecer datos
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    strategy.set_data(X_train_t, y_train_t, X_val_t, y_val_t)

    # 6. Entrenar
    start_time = time.time()
    result = strategy.train(verbose=verbose)
    training_time = time.time() - start_time

    # 7. Evaluar en test
    with torch.no_grad():
        model.to(strategy._device)
        X_test_t = X_test_t.to(strategy._device)
        y_test_t = y_test_t.to(strategy._device)

        test_output = model(X_test_t)
        test_loss = torch.nn.CrossEntropyLoss()(test_output, y_test_t).item()
        test_preds = test_output.argmax(dim=1)
        test_acc = (test_preds == y_test_t).float().mean().item()

    # 8. Metricas detalladas
    metrics_calc = MulticlassMetrics()
    detailed_metrics = metrics_calc.calculate_all_metrics(
        y_test,
        test_preds.cpu().numpy()
    )

    if verbose:
        print(f"\n  Resultados:")
        print(f"    Train Acc: {result.best_accuracy:.4f}")
        print(f"    Test Acc:  {test_acc:.4f}")
        print(f"    Test Loss: {test_loss:.6f}")
        print(f"    F1 (macro): {detailed_metrics['f1_score']['macro']:.4f}")
        print(f"    Tiempo: {training_time:.2f}s")

    return {
        'dataset': dataset_name,
        'strategy': strategy_name,
        'optimizer': optimizer_name,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_layers': hidden_layers,
        'n_params': n_params,
        'train_accuracy': result.best_accuracy,
        'train_loss': result.best_loss,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'iterations': result.iterations,
        'elapsed_time': result.elapsed_time,
        'total_time': training_time,
        'detailed_metrics': detailed_metrics,
        'history': result.history,
        'convergence_reason': result.convergence_reason,
        'model': model,
        'test_preds': test_preds.cpu().numpy(),
        'y_test': y_test,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Ejecuta benchmark completo."""

    # Info del sistema
    device = print_system_info()

    # Crear directorio de salida
    ensure_output_dir()

    # Configuracion
    print_header("CONFIGURACION DEL BENCHMARK")
    print(f"\nGeneral:")
    for key, value in GENERAL_CONFIG.items():
        print(f"  {key}: {value}")

    print(f"\nEstrategias: {STRATEGIES}")
    print(f"Optimizadores: {OPTIMIZERS}")
    print(f"Datasets: {DATASETS}")

    # Ejecutar benchmarks
    all_results = {}
    total_start = time.time()

    for dataset in DATASETS:
        print_header(f"BENCHMARK: {dataset.upper()}")

        for optimizer in OPTIMIZERS:
            for strategy in STRATEGIES:
                key = f"{dataset}_{optimizer}_{strategy}"

                try:
                    result = run_strategy_benchmark(
                        dataset_name=dataset,
                        strategy_name=strategy,
                        optimizer_name=optimizer,
                        verbose=True
                    )
                    all_results[key] = result

                    # Generar graficas
                    # Matriz de confusion
                    cm_path = os.path.join(
                        IMG_OUTPUT_DIR,
                        get_image_filename(dataset, optimizer, strategy, 'confusion_matrix')
                    )
                    class_names = [f"C{i}" for i in range(result['output_dim'])]
                    plot_confusion_matrix(
                        y_true=result['y_test'],
                        y_pred=result['test_preds'],
                        class_names=class_names,
                        save_path=cm_path
                    )

                    # Curvas de loss
                    if result['history']['train_loss']:
                        loss_path = os.path.join(
                            IMG_OUTPUT_DIR,
                            get_image_filename(dataset, optimizer, strategy, 'loss_curves')
                        )
                        plot_loss_curves(
                            history=result['history'],
                            title=f"{optimizer} {strategy.upper()} - {dataset}",
                            save_path=loss_path
                        )

                except Exception as e:
                    print(f"\n  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[key] = {'error': str(e)}

    total_time = time.time() - total_start

    # ==========================================================================
    # RESUMEN FINAL
    # ==========================================================================

    print_header("RESUMEN FINAL - COMPARATIVA DE ESTRATEGIAS", "=")

    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dispositivo: {device}")
    print(f"Tiempo total: {total_time:.2f}s")

    # Tabla de resultados por dataset
    for dataset in DATASETS:
        print(f"\n{'='*80}")
        print(f" Dataset: {dataset.upper()}")
        print(f"{'='*80}")

        print(f"\n{'-'*80}")
        print(f"{'Optimizer':<10} {'Strategy':<12} {'Train Acc':<12} {'Test Acc':<12} {'F1':<10} {'Time':<10}")
        print(f"{'-'*80}")

        for optimizer in OPTIMIZERS:
            for strategy in STRATEGIES:
                key = f"{dataset}_{optimizer}_{strategy}"
                result = all_results.get(key, {})

                if 'error' in result:
                    print(f"{optimizer:<10} {strategy:<12} ERROR: {result['error'][:30]}")
                elif result:
                    print(f"{optimizer:<10} {strategy:<12} "
                          f"{result['train_accuracy']:<12.4f} "
                          f"{result['test_accuracy']:<12.4f} "
                          f"{result['detailed_metrics']['f1_score']['macro']:<10.4f} "
                          f"{result['total_time']:<10.2f}s")

        print(f"{'-'*80}")

    # Mejor combinacion por dataset
    print("\n" + "="*80)
    print(" MEJORES COMBINACIONES")
    print("="*80)

    for dataset in DATASETS:
        best_key = None
        best_acc = 0

        for optimizer in OPTIMIZERS:
            for strategy in STRATEGIES:
                key = f"{dataset}_{optimizer}_{strategy}"
                result = all_results.get(key, {})

                if 'error' not in result and result.get('test_accuracy', 0) > best_acc:
                    best_acc = result['test_accuracy']
                    best_key = key

        if best_key:
            result = all_results[best_key]
            print(f"\n  {dataset.upper()}:")
            print(f"    Mejor: {result['optimizer']} + {result['strategy'].upper()}")
            print(f"    Test Acc: {result['test_accuracy']:.4f}")
            print(f"    F1: {result['detailed_metrics']['f1_score']['macro']:.4f}")

    # Analisis por estrategia
    print("\n" + "="*80)
    print(" ANALISIS POR ESTRATEGIA")
    print("="*80)

    for strategy in STRATEGIES:
        accs = []
        times = []

        for dataset in DATASETS:
            for optimizer in OPTIMIZERS:
                key = f"{dataset}_{optimizer}_{strategy}"
                result = all_results.get(key, {})

                if 'error' not in result and result:
                    accs.append(result['test_accuracy'])
                    times.append(result['total_time'])

        if accs:
            print(f"\n  {strategy.upper()}:")
            print(f"    Accuracy promedio: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
            print(f"    Tiempo promedio: {np.mean(times):.2f}s")

    print("\n" + "="*80)
    print(" BENCHMARK COMPLETADO")
    print("="*80)

    return all_results


if __name__ == "__main__":
    results = main()

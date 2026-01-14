"""
Benchmark: Entrenamiento de Redes Neuronales con QPSO

Este script ejecuta un benchmark comparativo en los datasets:
- Iris (4 features, 3 clases)
- Wine (13 features, 3 clases)
- Breast Cancer (30 features, 2 clases)

Configuracion:
- Activacion capas ocultas: tanh
- Activacion salida: softmax
- Arquitectura: input -> input*3 -> input*2 -> output
- Split: 70% train, 20% test, 10% validacion
- Cross-validation: 4 folds

Requisitos:
    conda activate pytorch_qpso_gpu
    python ann/main_qpso.py
"""

import torch
import numpy as np
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Agregar path para imports (portable: funciona sin importar el nombre del folder)
sys.path.insert(0, os.path.dirname(__file__))

from models import QPSOCompatibleANN
from optimizers import QPSONNOptimizer
from optimizers.qpso_nn import NNOptimizationConfig
from trainers import Trainer, TrainingConfig
from utils import (
    MulticlassMetrics,
    plot_confusion_matrix,
    plot_training_history,
    plot_loss_curves,
    plot_accuracy_curves,
    plot_complete_training_summary,
    plot_cv_summary
)

# Sklearn para datasets y split
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURACION DEL BENCHMARK
# =============================================================================

BENCHMARK_CONFIG = {
    'activation': 'tanh',           # Activacion capas ocultas
    'output_activation': 'softmax', # Activacion capa salida
    'n_particles': 50,              # Numero de particulas QPSO
    'max_iters': 150,               # Iteraciones maximas
    'alpha': (1.0, 0.5),            # Alpha con decay
    'n_folds': 4,                   # Folds para CV
    'train_size': 0.70,             # 70% entrenamiento
    'test_size': 0.20,              # 20% test
    'val_size': 0.10,               # 10% validacion
    'random_state': 42,             # Semilla para reproducibilidad
    'patience': 50,                 # Early stopping patience
}

DATASETS = ['iris', 'wine', 'breast_cancer']

# Directorio de salida para graficas
IMG_OUTPUT_DIR = './img/metric/QPSO'


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_image_filename(
    dataset: str,
    plot_type: str,
    config: Dict[str, Any]
) -> str:
    """
    Genera nombre de archivo descriptivo para graficas.

    Args:
        dataset: Nombre del dataset
        plot_type: Tipo de grafica ('confusion_matrix', 'training_history')
        config: Configuracion del benchmark

    Returns:
        Nombre de archivo con formato descriptivo
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    alpha_str = f"alpha_{config['alpha'][0]}-{config['alpha'][1]}"
    particles = f"p{config['n_particles']}"
    iters = f"i{config['max_iters']}"

    filename = f"QPSO_{dataset}_{plot_type}_{alpha_str}_{particles}_{iters}_{timestamp}.png"
    return filename


def ensure_output_dir():
    """Crea el directorio de salida si no existe."""
    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
    print(f"Directorio de graficas: {IMG_OUTPUT_DIR}")

def print_header(title: str, char: str = "="):
    """Imprime un encabezado formateado."""
    width = 70
    print("\n" + char * width)
    print(f" {title}")
    print(char * width)


def print_subheader(title: str):
    """Imprime un subencabezado."""
    print(f"\n--- {title} ---")


def get_architecture(input_dim: int, output_dim: int) -> List[int]:
    """
    Genera arquitectura segun especificacion: input*3, input*2.

    Args:
        input_dim: Dimension de entrada
        output_dim: Dimension de salida

    Returns:
        Lista con dimensiones de capas ocultas [input*3, input*2]
    """
    return [input_dim * 3, input_dim * 2]


def load_and_split_dataset(
    name: str,
    train_size: float = 0.70,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga un dataset y lo divide en train/test/val.

    Args:
        name: Nombre del dataset ('iris', 'wine', 'breast_cancer')
        train_size: Proporcion de entrenamiento (default: 0.70)
        test_size: Proporcion de test (default: 0.20)
        val_size: Proporcion de validacion (default: 0.10)
        random_state: Semilla aleatoria

    Returns:
        Tupla (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Cargar dataset
    loaders = {
        'iris': datasets.load_iris,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer,
    }

    if name not in loaders:
        raise ValueError(f"Dataset no soportado: {name}")

    data = loaders[name]()
    X, y = data.data, data.target

    # Normalizar features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Primer split: separar test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Segundo split: separar train y val del resto
    # val_size relativo al conjunto restante
    val_ratio = val_size / (train_size + val_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def print_system_info():
    """Imprime informacion del sistema."""
    print_header("INFORMACION DEL SISTEMA")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo a usar: {device}")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return device


# =============================================================================
# FUNCION DE BENCHMARK
# =============================================================================

def run_benchmark(
    dataset_name: str,
    config: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Ejecuta benchmark en un dataset especifico.

    Args:
        dataset_name: Nombre del dataset
        config: Configuracion del benchmark
        verbose: Imprimir progreso

    Returns:
        Diccionario con resultados del benchmark
    """
    print_header(f"BENCHMARK: {dataset_name.upper()}")

    # ==========================================================================
    # 1. Cargar y preparar datos
    # ==========================================================================

    print_subheader("1. Cargando datos")

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_dataset(
        dataset_name,
        train_size=config['train_size'],
        test_size=config['test_size'],
        val_size=config['val_size'],
        random_state=config['random_state']
    )

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    print(f"Dataset: {dataset_name}")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Train samples: {len(y_train)} ({config['train_size']*100:.0f}%)")
    print(f"  Val samples: {len(y_val)} ({config['val_size']*100:.0f}%)")
    print(f"  Test samples: {len(y_test)} ({config['test_size']*100:.0f}%)")
    print(f"  Class distribution (train): {np.bincount(y_train)}")

    # ==========================================================================
    # 2. Definir arquitectura
    # ==========================================================================

    print_subheader("2. Arquitectura de la red")

    hidden_layers = get_architecture(input_dim, output_dim)

    print(f"  Entrada: {input_dim}")
    print(f"  Oculta 1: {hidden_layers[0]} (input * 3)")
    print(f"  Oculta 2: {hidden_layers[1]} (input * 2)")
    print(f"  Salida: {output_dim}")
    print(f"  Activacion ocultas: {config['activation']}")
    print(f"  Activacion salida: {config['output_activation']}")

    # Calcular parametros
    n_params = (
        (input_dim + 1) * hidden_layers[0] +
        (hidden_layers[0] + 1) * hidden_layers[1] +
        (hidden_layers[1] + 1) * output_dim
    )
    print(f"  Total parametros: {n_params:,}")

    # ==========================================================================
    # 3. Crear modelo
    # ==========================================================================

    print_subheader("3. Creando modelo")

    model = QPSOCompatibleANN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        activation=config['activation'],
        output_activation=config['output_activation']
    )

    print(model)

    # ==========================================================================
    # 4. Configurar optimizador
    # ==========================================================================

    print_subheader("4. Configurando optimizador QPSO")

    opt_config = NNOptimizationConfig(
        n_particles=config['n_particles'],
        max_iters=config['max_iters'],
        alpha=config['alpha'],
        patience=config['patience'],
        seed=config['random_state'],
        track_history=True
    )

    optimizer = QPSONNOptimizer(model, config=opt_config)

    print(f"  Particulas: {config['n_particles']}")
    print(f"  Iteraciones max: {config['max_iters']}")
    print(f"  Alpha: {config['alpha']}")
    print(f"  Patience: {config['patience']}")

    # ==========================================================================
    # 5. Entrenar
    # ==========================================================================

    print_subheader("5. Entrenando modelo")

    # Convertir a tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    start_time = time.time()

    result = optimizer.fit(
        X_train_t, y_train_t,
        X_val=X_val_t, y_val=y_val_t,
        verbose=verbose
    )

    training_time = time.time() - start_time

    # ==========================================================================
    # 6. Evaluar
    # ==========================================================================

    print_subheader("6. Evaluando modelo")

    train_metrics = optimizer.evaluate(X_train_t, y_train_t)
    val_metrics = optimizer.evaluate(X_val_t, y_val_t)
    test_metrics = optimizer.evaluate(X_test_t, y_test_t)

    print(f"  Train - Loss: {train_metrics['loss']:.6f}, Acc: {train_metrics['accuracy']:.4f}")
    print(f"  Val   - Loss: {val_metrics['loss']:.6f}, Acc: {val_metrics['accuracy']:.4f}")
    print(f"  Test  - Loss: {test_metrics['loss']:.6f}, Acc: {test_metrics['accuracy']:.4f}")

    # ==========================================================================
    # 7. Metricas detalladas en test
    # ==========================================================================

    print_subheader("7. Metricas detalladas (Test)")

    predictions = optimizer.predict(X_test_t)

    metrics_calc = MulticlassMetrics()
    detailed_metrics = metrics_calc.calculate_all_metrics(
        y_test,
        predictions.cpu().numpy()
    )

    print(f"  Accuracy: {detailed_metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {detailed_metrics['precision']['macro']:.4f}")
    print(f"  Recall (macro): {detailed_metrics['recall']['macro']:.4f}")
    print(f"  F1-Score (macro): {detailed_metrics['f1_score']['macro']:.4f}")
    print(f"  Cohen's Kappa: {detailed_metrics['cohen_kappa']:.4f}")

    print(f"\n  Matriz de Confusion:")
    print(detailed_metrics['confusion_matrix'])

    # ==========================================================================
    # 8. Cross-validation
    # ==========================================================================

    print_subheader(f"8. Cross-Validation ({config['n_folds']} folds)")

    # Combinar train y val para CV
    X_cv = np.vstack([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val])

    trainer_config = TrainingConfig(
        hidden_layers=hidden_layers,
        activation=config['activation'],
        n_particles=config['n_particles'],
        max_iters=config['max_iters'],
        alpha=config['alpha'],
        n_folds=config['n_folds'],
        patience=config['patience'],
        random_state=config['random_state'],
        verbose=False,
        save_best_model=False
    )

    trainer = Trainer(input_dim, output_dim, trainer_config)

    cv_start = time.time()
    cv_result = trainer.fit_cv(X_cv, y_cv, X_test=X_test, y_test=y_test)
    cv_time = time.time() - cv_start

    print(f"  Resultados por fold:")
    fold_accs = []
    for fold in cv_result.fold_results:
        fold_accs.append(fold['val_acc'])
        print(f"    Fold {fold['fold']}: train={fold['train_acc']:.4f}, val={fold['val_acc']:.4f}")

    print(f"\n  Media +/- Std: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print(f"  Test Accuracy (mejor modelo): {cv_result.test_accuracy:.4f}")
    print(f"  Tiempo CV: {cv_time:.2f}s")

    # ==========================================================================
    # 9. Resumen
    # ==========================================================================

    print_subheader("9. Resumen")

    print(f"  Dataset: {dataset_name}")
    print(f"  Arquitectura: {input_dim} -> {hidden_layers} -> {output_dim}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  CV Accuracy: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print(f"  Tiempo total: {training_time + cv_time:.2f}s")

    # ==========================================================================
    # 10. Generar graficas
    # ==========================================================================

    print_subheader("10. Generando graficas")

    # Obtener nombres de clases
    class_names = [f"Clase_{i}" for i in range(output_dim)]
    if dataset_name == 'iris':
        class_names = ['setosa', 'versicolor', 'virginica']
    elif dataset_name == 'wine':
        class_names = ['Clase_0', 'Clase_1', 'Clase_2']
    elif dataset_name == 'breast_cancer':
        class_names = ['maligno', 'benigno']

    # Obtener historial del optimizador
    history = optimizer.get_history()

    # Grafica 1: Matriz de confusion
    cm_filename = get_image_filename(dataset_name, 'confusion_matrix', config)
    cm_path = os.path.join(IMG_OUTPUT_DIR, cm_filename)
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=predictions.cpu().numpy(),
        class_names=class_names,
        save_path=cm_path
    )

    # Grafica 2: Curvas de Loss (train/val)
    if history and 'train_loss' in history:
        loss_filename = get_image_filename(dataset_name, 'loss_curves', config)
        loss_path = os.path.join(IMG_OUTPUT_DIR, loss_filename)
        plot_loss_curves(
            history=history,
            title=f"QPSO - Curvas de Loss ({dataset_name})",
            save_path=loss_path
        )

    # Grafica 3: Curvas de Accuracy (train/val)
    if history and 'train_acc' in history:
        acc_filename = get_image_filename(dataset_name, 'accuracy_curves', config)
        acc_path = os.path.join(IMG_OUTPUT_DIR, acc_filename)
        plot_accuracy_curves(
            history=history,
            title=f"QPSO - Curvas de Accuracy ({dataset_name})",
            save_path=acc_path
        )

    # Grafica 4: Resumen completo de entrenamiento (train/val/test)
    summary_filename = get_image_filename(dataset_name, 'training_summary', config)
    summary_path = os.path.join(IMG_OUTPUT_DIR, summary_filename)
    plot_complete_training_summary(
        history=history if history else {},
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        title=f"QPSO - Resumen Entrenamiento ({dataset_name})",
        save_path=summary_path
    )

    # Grafica 5: Resumen de Cross-Validation
    if cv_result.fold_results:
        cv_filename = get_image_filename(dataset_name, 'cv_summary', config)
        cv_path = os.path.join(IMG_OUTPUT_DIR, cv_filename)
        plot_cv_summary(
            fold_results=cv_result.fold_results,
            title=f"QPSO - Cross-Validation ({dataset_name}, {config['n_folds']} folds)",
            save_path=cv_path
        )

    # Compilar resultados
    results = {
        'dataset': dataset_name,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_layers': hidden_layers,
        'n_params': n_params,
        'train_samples': len(y_train),
        'val_samples': len(y_val),
        'test_samples': len(y_test),
        'train_accuracy': train_metrics['accuracy'],
        'val_accuracy': val_metrics['accuracy'],
        'test_accuracy': test_metrics['accuracy'],
        'train_loss': train_metrics['loss'],
        'val_loss': val_metrics['loss'],
        'test_loss': test_metrics['loss'],
        'cv_mean': np.mean(fold_accs),
        'cv_std': np.std(fold_accs),
        'cv_folds': fold_accs,
        'detailed_metrics': detailed_metrics,
        'training_time': training_time,
        'cv_time': cv_time,
        'total_time': training_time + cv_time,
        'iterations': result.iterations,
        'convergence_reason': result.convergence_reason
    }

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Ejecuta el benchmark completo."""

    # Informacion del sistema
    device = print_system_info()

    # Crear directorio de graficas
    ensure_output_dir()

    # Configuracion
    print_header("CONFIGURACION DEL BENCHMARK")
    for key, value in BENCHMARK_CONFIG.items():
        print(f"  {key}: {value}")

    # Ejecutar benchmark para cada dataset
    all_results = {}
    total_start = time.time()

    for dataset in DATASETS:
        try:
            results = run_benchmark(
                dataset,
                BENCHMARK_CONFIG,
                verbose=True
            )
            all_results[dataset] = results
        except Exception as e:
            print(f"\nError en {dataset}: {e}")
            all_results[dataset] = {'error': str(e)}

    total_time = time.time() - total_start

    # ==========================================================================
    # RESUMEN FINAL
    # ==========================================================================

    print_header("RESUMEN FINAL DEL BENCHMARK", "=")

    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dispositivo: {device}")
    print(f"Tiempo total: {total_time:.2f}s\n")

    # Tabla de resultados
    print("-" * 90)
    print(f"{'Dataset':<15} {'Arch':<20} {'Params':<10} {'Test Acc':<12} {'CV Acc':<18} {'Time':<10}")
    print("-" * 90)

    for dataset, results in all_results.items():
        if 'error' in results:
            print(f"{dataset:<15} ERROR: {results['error']}")
        else:
            arch = f"{results['input_dim']}->{results['hidden_layers']}->{results['output_dim']}"
            cv_acc = f"{results['cv_mean']:.4f} +/- {results['cv_std']:.4f}"
            print(f"{dataset:<15} {arch:<20} {results['n_params']:<10,} {results['test_accuracy']:<12.4f} {cv_acc:<18} {results['total_time']:<10.2f}s")

    print("-" * 90)

    # Mejor resultado
    best_dataset = max(
        [d for d in all_results if 'error' not in all_results[d]],
        key=lambda x: all_results[x]['test_accuracy']
    )

    print(f"\nMejor resultado: {best_dataset} con {all_results[best_dataset]['test_accuracy']:.4f} accuracy")

    # Detalles por dataset
    print("\nDetalles por dataset:")
    for dataset, results in all_results.items():
        if 'error' not in results:
            print(f"\n  {dataset.upper()}:")
            print(f"    Train: {results['train_accuracy']:.4f}")
            print(f"    Val:   {results['val_accuracy']:.4f}")
            print(f"    Test:  {results['test_accuracy']:.4f}")
            print(f"    CV:    {results['cv_mean']:.4f} +/- {results['cv_std']:.4f}")
            print(f"    F1:    {results['detailed_metrics']['f1_score']['macro']:.4f}")
            print(f"    Kappa: {results['detailed_metrics']['cohen_kappa']:.4f}")

    print("\n" + "=" * 90)
    print(" BENCHMARK COMPLETADO")
    print("=" * 90)

    return all_results


if __name__ == "__main__":
    results = main()

"""
Benchmark: Entrenamiento de Redes Neuronales en MCW (Multi-Class Weather)

Este script ejecuta un benchmark comparativo usando QPSO y QDPSO
en el dataset de imagenes de clima (MCW).

Dataset MCW:
- 4 clases: cloudy, rain, shine, sunrise
- Features extraidas: histogram (64), haralick (13), hu_moments (7) = 84 total
- Soporta reduccion de dimensionalidad: isomap, pca, mds

Configuracion:
- Optimizadores: QPSO (alpha decay) y QDPSO (factor g)
- Activacion capas ocultas: tanh
- Activacion salida: softmax
- Arquitectura: input -> input*3 -> input*2 -> output
- Split: 70% train, 10% validacion, 20% test
- Cross-validation: 4 folds

Requisitos:
    conda activate pytorch_qpso_gpu
    python ann/main_mcw.py
"""

import torch
import numpy as np
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Agregar path para imports (portable: funciona sin importar el nombre del folder)
sys.path.insert(0, os.path.dirname(__file__))

from models import QPSOCompatibleANN
from optimizers import QPSONNOptimizer, QDPSONNOptimizer
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
from data import load_mcw, MCW_CLASSES, REDUCTION_METHODS


# =============================================================================
# CONFIGURACION DEL BENCHMARK
# =============================================================================

# Configuracion del dataset
MCW_CONFIG = {
    'root_path': './data/img/mcw',  # Ruta a imagenes MCW
    'reduction_method': 'isomap',    # 'isomap', 'pca', 'mds', o None
    'n_components': 7,              # Componentes para reduccion (None = auto)
    'img_size': (150, 150),          # Tamaño de imagenes
    'bins': 4,                       # Bins para histograma HSV
}

# Configuracion del entrenamiento
TRAINING_CONFIG = {
    'activation': 'tanh',            # Activacion capas ocultas
    'output_activation': 'softmax',  # Activacion capa salida
    'n_particles': 50,               # Numero de particulas
    'max_iters': 1000,                # Iteraciones maximas
    'n_folds': 4,                    # Folds para CV
    'train_size': 0.70,              # 70% entrenamiento
    'val_size': 0.10,                # 10% validacion
    'test_size': 0.20,               # 20% test
    'random_state': 42,              # Semilla para reproducibilidad
    'patience': 50,                  # Early stopping patience
}

# Configuracion de optimizadores
QPSO_CONFIG = {
    'alpha': (1.0, 0.5),             # Alpha con decay para QPSO
}

QDPSO_CONFIG = {
    'g': 0.96,                       # Factor g para QDPSO
}

# Optimizadores a ejecutar
OPTIMIZERS = ['QPSO', 'QDPSO']  # Puede ser ['QPSO'], ['QDPSO'], o ambos

# Directorio de salida para graficas
IMG_OUTPUT_DIR = './img/metric/MCW'


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_image_filename(
    optimizer: str,
    plot_type: str,
    reduction: Optional[str],
    n_components: Optional[int],
    config: Dict[str, Any]
) -> str:
    """
    Genera nombre de archivo descriptivo para graficas.

    Args:
        optimizer: Nombre del optimizador ('QPSO' o 'QDPSO')
        plot_type: Tipo de grafica
        reduction: Metodo de reduccion usado
        n_components: Numero de componentes
        config: Configuracion del benchmark

    Returns:
        Nombre de archivo con formato descriptivo
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if optimizer == 'QPSO':
        opt_params = f"alpha_{QPSO_CONFIG['alpha'][0]}-{QPSO_CONFIG['alpha'][1]}"
    else:
        opt_params = f"g_{QDPSO_CONFIG['g']}"

    particles = f"p{config['n_particles']}"
    iters = f"i{config['max_iters']}"

    reduction_str = f"_{reduction}_c{n_components}" if reduction else "_raw"

    filename = f"{optimizer}_MCW_{plot_type}{reduction_str}_{opt_params}_{particles}_{iters}_{timestamp}.png"
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
# FUNCION DE BENCHMARK PARA UN OPTIMIZADOR
# =============================================================================

def run_benchmark_single_optimizer(
    optimizer_name: str,
    data: Any,
    config: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Ejecuta benchmark con un optimizador especifico.

    Args:
        optimizer_name: 'QPSO' o 'QDPSO'
        data: MCWDataResult con los datos cargados
        config: Configuracion del entrenamiento
        verbose: Imprimir progreso

    Returns:
        Diccionario con resultados del benchmark
    """
    print_header(f"BENCHMARK {optimizer_name}: MCW")

    # ==========================================================================
    # 1. Preparar datos
    # ==========================================================================

    print_subheader("1. Datos cargados")

    X_train, X_val, X_test = data.X_train, data.X_val, data.X_test
    y_train, y_val, y_test = data.y_train, data.y_val, data.y_test

    input_dim = data.n_features
    output_dim = data.n_classes

    print(f"Dataset: MCW (Multi-Class Weather)")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Train samples: {len(y_train)}")
    print(f"  Val samples: {len(y_val)}")
    print(f"  Test samples: {len(y_test)}")
    print(f"  Reduccion: {data.reduction_method or 'None'}")
    if data.reduction_method:
        print(f"  Componentes: {data.n_components}")
        print(f"  Features originales: {data.original_features}")
    print(f"  Clases: {data.class_names}")

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

    print_subheader(f"4. Configurando optimizador {optimizer_name}")

    if optimizer_name == 'QPSO':
        opt_config = NNOptimizationConfig(
            n_particles=config['n_particles'],
            max_iters=config['max_iters'],
            alpha=QPSO_CONFIG['alpha'],
            patience=config['patience'],
            seed=config['random_state'],
            track_history=True
        )
        optimizer = QPSONNOptimizer(model, config=opt_config)
        print(f"  Alpha: {QPSO_CONFIG['alpha']}")
    else:  # QDPSO
        opt_config = NNOptimizationConfig(
            n_particles=config['n_particles'],
            max_iters=config['max_iters'],
            g=QDPSO_CONFIG['g'],
            patience=config['patience'],
            seed=config['random_state'],
            track_history=True
        )
        optimizer = QDPSONNOptimizer(model, config=opt_config)
        print(f"  Factor g: {QDPSO_CONFIG['g']}")

    print(f"  Particulas: {config['n_particles']}")
    print(f"  Iteraciones max: {config['max_iters']}")
    print(f"  Patience: {config['patience']}")

    # ==========================================================================
    # 5. Entrenar
    # ==========================================================================

    print_subheader(f"5. Entrenando modelo con {optimizer_name}")

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
        alpha=QPSO_CONFIG['alpha'] if optimizer_name == 'QPSO' else None,
        g=QDPSO_CONFIG['g'] if optimizer_name == 'QDPSO' else None,
        use_qdpso=(optimizer_name == 'QDPSO'),
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

    print(f"  Dataset: MCW")
    print(f"  Optimizador: {optimizer_name}")
    print(f"  Reduccion: {data.reduction_method or 'None'}")
    print(f"  Arquitectura: {input_dim} -> {hidden_layers} -> {output_dim}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  CV Accuracy: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print(f"  Tiempo total: {training_time + cv_time:.2f}s")

    # ==========================================================================
    # 10. Generar graficas
    # ==========================================================================

    print_subheader("10. Generando graficas")

    # Obtener historial del optimizador
    history = optimizer.get_history()

    # Grafica 1: Matriz de confusion
    cm_filename = get_image_filename(
        optimizer_name, 'confusion_matrix',
        data.reduction_method, data.n_components, config
    )
    cm_path = os.path.join(IMG_OUTPUT_DIR, cm_filename)
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=predictions.cpu().numpy(),
        class_names=data.class_names,
        save_path=cm_path
    )

    # Grafica 2: Curvas de Loss (train/val)
    if history and 'train_loss' in history:
        loss_filename = get_image_filename(
            optimizer_name, 'loss_curves',
            data.reduction_method, data.n_components, config
        )
        loss_path = os.path.join(IMG_OUTPUT_DIR, loss_filename)
        plot_loss_curves(
            history=history,
            title=f"{optimizer_name} - Curvas de Loss (MCW)",
            save_path=loss_path
        )

    # Grafica 3: Curvas de Accuracy (train/val)
    if history and 'train_acc' in history:
        acc_filename = get_image_filename(
            optimizer_name, 'accuracy_curves',
            data.reduction_method, data.n_components, config
        )
        acc_path = os.path.join(IMG_OUTPUT_DIR, acc_filename)
        plot_accuracy_curves(
            history=history,
            title=f"{optimizer_name} - Curvas de Accuracy (MCW)",
            save_path=acc_path
        )

    # Grafica 4: Resumen completo de entrenamiento (train/val/test)
    summary_filename = get_image_filename(
        optimizer_name, 'training_summary',
        data.reduction_method, data.n_components, config
    )
    summary_path = os.path.join(IMG_OUTPUT_DIR, summary_filename)
    plot_complete_training_summary(
        history=history if history else {},
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        title=f"{optimizer_name} - Resumen Entrenamiento (MCW)",
        save_path=summary_path
    )

    # Grafica 5: Resumen de Cross-Validation
    if cv_result.fold_results:
        cv_filename = get_image_filename(
            optimizer_name, 'cv_summary',
            data.reduction_method, data.n_components, config
        )
        cv_path = os.path.join(IMG_OUTPUT_DIR, cv_filename)
        plot_cv_summary(
            fold_results=cv_result.fold_results,
            title=f"{optimizer_name} - Cross-Validation (MCW, {config['n_folds']} folds)",
            save_path=cv_path
        )

    # Compilar resultados
    results = {
        'optimizer': optimizer_name,
        'dataset': 'MCW',
        'reduction_method': data.reduction_method,
        'n_components': data.n_components,
        'original_features': data.original_features,
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
    """Ejecuta el benchmark completo MCW con QPSO y QDPSO."""

    # Informacion del sistema
    device = print_system_info()

    # Crear directorio de graficas
    ensure_output_dir()

    # Configuracion
    print_header("CONFIGURACION DEL BENCHMARK MCW")
    print("\nDataset MCW:")
    for key, value in MCW_CONFIG.items():
        print(f"  {key}: {value}")
    print("\nEntrenamiento:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    print("\nQPSO:")
    for key, value in QPSO_CONFIG.items():
        print(f"  {key}: {value}")
    print("\nQDPSO:")
    for key, value in QDPSO_CONFIG.items():
        print(f"  {key}: {value}")
    print(f"\nOptimizadores a ejecutar: {OPTIMIZERS}")

    # ==========================================================================
    # CARGAR DATASET MCW
    # ==========================================================================

    print_header("CARGANDO DATASET MCW")

    try:
        data = load_mcw(
            root_path=MCW_CONFIG['root_path'],
            train_size=TRAINING_CONFIG['train_size'],
            val_size=TRAINING_CONFIG['val_size'],
            test_size=TRAINING_CONFIG['test_size'],
            reduction_method=MCW_CONFIG['reduction_method'],
            n_components=MCW_CONFIG['n_components'],
            random_state=TRAINING_CONFIG['random_state'],
            img_size=MCW_CONFIG['img_size'],
            bins=MCW_CONFIG['bins'],
            verbose=True
        )
        print(f"\nDataset cargado exitosamente:")
        print(data)
    except Exception as e:
        print(f"\nError cargando dataset MCW: {e}")
        print(f"\nAsegurese de que exista la estructura:")
        print(f"  {MCW_CONFIG['root_path']}/")
        print(f"  ├── cloudy/")
        print(f"  ├── rain/")
        print(f"  ├── shine/")
        print(f"  └── sunrise/")
        return {}

    # ==========================================================================
    # EJECUTAR BENCHMARKS
    # ==========================================================================

    all_results = {}
    total_start = time.time()

    for opt_name in OPTIMIZERS:
        try:
            results = run_benchmark_single_optimizer(
                opt_name,
                data,
                TRAINING_CONFIG,
                verbose=True
            )
            all_results[opt_name] = results
        except Exception as e:
            print(f"\nError con {opt_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[opt_name] = {'error': str(e)}

    total_time = time.time() - total_start

    # ==========================================================================
    # RESUMEN FINAL COMPARATIVO
    # ==========================================================================

    print_header("RESUMEN FINAL - COMPARATIVA QPSO vs QDPSO", "=")

    print(f"\nDataset: MCW (Multi-Class Weather)")
    print(f"Reduccion: {data.reduction_method or 'None'}")
    if data.reduction_method:
        print(f"Componentes: {data.n_components} (de {data.original_features} originales)")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dispositivo: {device}")
    print(f"Tiempo total: {total_time:.2f}s\n")

    # Tabla de resultados
    print("-" * 100)
    print(f"{'Optimizador':<12} {'Params':<12} {'Test Acc':<12} {'CV Acc':<20} {'F1':<10} {'Kappa':<10} {'Time':<10}")
    print("-" * 100)

    for opt_name, results in all_results.items():
        if 'error' in results:
            print(f"{opt_name:<12} ERROR: {results['error']}")
        else:
            cv_acc = f"{results['cv_mean']:.4f} +/- {results['cv_std']:.4f}"
            f1 = results['detailed_metrics']['f1_score']['macro']
            kappa = results['detailed_metrics']['cohen_kappa']
            print(f"{opt_name:<12} {results['n_params']:<12,} {results['test_accuracy']:<12.4f} {cv_acc:<20} {f1:<10.4f} {kappa:<10.4f} {results['total_time']:<10.2f}s")

    print("-" * 100)

    # Mejor resultado
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    if valid_results:
        best_opt = max(valid_results, key=lambda x: valid_results[x]['test_accuracy'])
        print(f"\nMejor optimizador: {best_opt} con {all_results[best_opt]['test_accuracy']:.4f} accuracy")

    # Detalles por optimizador
    print("\nDetalles por optimizador:")
    for opt_name, results in all_results.items():
        if 'error' not in results:
            print(f"\n  {opt_name}:")
            print(f"    Arquitectura: {results['input_dim']} -> {results['hidden_layers']} -> {results['output_dim']}")
            print(f"    Train: {results['train_accuracy']:.4f}")
            print(f"    Val:   {results['val_accuracy']:.4f}")
            print(f"    Test:  {results['test_accuracy']:.4f}")
            print(f"    CV:    {results['cv_mean']:.4f} +/- {results['cv_std']:.4f}")
            print(f"    F1:    {results['detailed_metrics']['f1_score']['macro']:.4f}")
            print(f"    Kappa: {results['detailed_metrics']['cohen_kappa']:.4f}")
            print(f"    Iteraciones: {results['iterations']}")
            print(f"    Convergencia: {results['convergence_reason']}")

    # Metricas por clase (del mejor modelo)
    if valid_results:
        best_results = all_results[best_opt]
        class_report = best_results['detailed_metrics']['classification_report']
        print(f"\nMetricas por clase ({best_opt}):")
        for i, class_name in enumerate(data.class_names):
            class_key = str(i)
            if class_key in class_report:
                prec = class_report[class_key]['precision']
                rec = class_report[class_key]['recall']
                f1 = class_report[class_key]['f1-score']
                print(f"  {class_name:<10}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    print("\n" + "=" * 100)
    print(" BENCHMARK MCW COMPLETADO")
    print("=" * 100)

    return all_results


if __name__ == "__main__":
    results = main()

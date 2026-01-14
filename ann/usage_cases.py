"""
Ejemplos de Uso: Entrenamiento de Redes Neuronales con QPSO

Este archivo demuestra como entrenar redes neuronales utilizando
QPSO (Quantum Particle Swarm Optimization) en lugar de backpropagation.

Ejemplos incluidos:
1. Uso basico con dataset Iris
2. Configuracion personalizada
3. Cross-validation
4. Comparativa QPSO vs QDPSO
5. Ejemplo con arquitectura escalada
6. Guardado y carga de modelos
7. Metricas detalladas
8. Visualizacion con graficas (matriz de confusion, curvas de entrenamiento)

Requisitos:
    pip install torch numpy scikit-learn matplotlib seaborn
"""

import torch
import numpy as np
import time
import sys
import os

# Agregar path para imports (portable: funciona sin importar el nombre del folder)
sys.path.insert(0, os.path.dirname(__file__))

from models import QPSOCompatibleANN, create_scaled_architecture
from optimizers import QPSONNOptimizer, QDPSONNOptimizer
from optimizers.qpso_nn import NNOptimizationConfig
from trainers import Trainer, TrainingConfig
from utils import (
    load_dataset, MulticlassMetrics, train_test_split,
    plot_confusion_matrix, plot_training_history
)
from datetime import datetime


def print_header(title: str):
    """Imprime un encabezado formateado."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


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
    return device


# =============================================================================
# 1. USO BASICO
# =============================================================================

def example_basic_usage():
    """Ejemplo basico de entrenamiento con QPSO."""
    print_header("1. USO BASICO - Dataset Iris")

    # Cargar dataset
    X_train, X_test, y_train, y_test = load_dataset('iris', test_size=0.2)
    print(f"Dataset Iris: train={len(y_train)}, test={len(y_test)}")
    print(f"Features: {X_train.shape[1]}, Clases: {len(np.unique(y_train))}")

    # Crear modelo
    model = QPSOCompatibleANN(
        input_dim=X_train.shape[1],
        output_dim=len(np.unique(y_train)),
        hidden_layers=[16, 8],
        activation='tanh',
        output_activation='log_softmax'
    )
    print(f"\nModelo creado:\n{model}")

    # Crear configuracion
    config = NNOptimizationConfig(
        n_particles=30,
        max_iters=100,
        alpha=(1.0, 0.5),
        weight_bound=1.0,
        patience=30,
        seed=42
    )

    # Crear optimizador
    optimizer = QPSONNOptimizer(model, config=config)

    # Convertir a tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Entrenar
    result = optimizer.fit(X_train_t, y_train_t, verbose=True)

    # Evaluar
    test_metrics = optimizer.evaluate(X_test_t, y_test_t)
    print(f"\nResultados en Test:")
    print(f"  Loss: {test_metrics['loss']:.6f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    return test_metrics['accuracy']


# =============================================================================
# 2. CONFIGURACION PERSONALIZADA
# =============================================================================

def example_custom_config():
    """Ejemplo con configuracion personalizada."""
    print_header("2. CONFIGURACION PERSONALIZADA")

    X_train, X_test, y_train, y_test = load_dataset('wine')
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    print(f"Dataset Wine: {input_dim} features, {output_dim} clases")

    # Arquitectura escalada [3:2:1]
    hidden_layers = create_scaled_architecture(input_dim, scale_factor=0.5)
    print(f"Arquitectura escalada: {[input_dim]} -> {hidden_layers} -> {[output_dim]}")

    # Configuracion del trainer
    config = TrainingConfig(
        hidden_layers=hidden_layers,
        activation='tanh',
        n_particles=50,
        max_iters=150,
        alpha=(1.0, 0.5),
        weight_bound=1.5,
        patience=40,
        random_state=42,
        verbose=True,
        save_best_model=False
    )

    # Crear trainer
    trainer = Trainer(
        input_dim=input_dim,
        output_dim=output_dim,
        config=config
    )

    # Entrenar
    result = trainer.fit(
        X_train, y_train,
        X_test=X_test, y_test=y_test
    )

    print(f"\nMejores resultados:")
    print(f"  Train Acc: {result.train_accuracy:.4f}")
    print(f"  Val Acc:   {result.val_accuracy:.4f}")
    print(f"  Test Acc:  {result.test_accuracy:.4f}")

    return result.test_accuracy


# =============================================================================
# 3. CROSS-VALIDATION
# =============================================================================

def example_cross_validation():
    """Ejemplo con cross-validation."""
    print_header("3. CROSS-VALIDATION")

    X_train, X_test, y_train, y_test = load_dataset('breast_cancer')
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    print(f"Dataset Breast Cancer: {len(y_train) + len(y_test)} muestras")
    print(f"  Features: {input_dim}, Clases: {output_dim}")

    # Combinar train para CV
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # Split final
    X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42
    )

    config = TrainingConfig(
        hidden_layers=[32, 16],
        n_particles=40,
        max_iters=80,
        n_folds=5,
        random_state=42,
        verbose=True,
        save_best_model=False
    )

    trainer = Trainer(
        input_dim=input_dim,
        output_dim=output_dim,
        config=config
    )

    # Entrenar con CV
    result = trainer.fit_cv(
        X_train_cv, y_train_cv,
        X_test=X_test_cv, y_test=y_test_cv
    )

    # Mostrar resultados por fold
    if result.fold_results:
        print("\nResultados por fold:")
        for fold in result.fold_results:
            print(f"  Fold {fold['fold']}: val_acc={fold['val_acc']:.4f}")

    return result.test_accuracy


# =============================================================================
# 4. COMPARATIVA QPSO vs QDPSO
# =============================================================================

def example_qpso_vs_qdpso():
    """Compara QPSO y QDPSO."""
    print_header("4. COMPARATIVA QPSO vs QDPSO")

    X_train, X_test, y_train, y_test = load_dataset('digits')
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    print(f"Dataset Digits: {input_dim} features, {output_dim} clases")
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    results = {}

    for use_qdpso in [False, True]:
        algo_name = "QDPSO" if use_qdpso else "QPSO"
        print(f"\n--- Entrenando con {algo_name} ---")

        config = TrainingConfig(
            hidden_layers=[64, 32],
            n_particles=30,
            max_iters=50,
            use_qdpso=use_qdpso,
            g=0.96 if use_qdpso else 0.96,
            alpha=(1.0, 0.5),
            random_state=42,
            verbose=False,
            save_best_model=False
        )

        trainer = Trainer(input_dim, output_dim, config)

        start = time.time()
        result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)
        elapsed = time.time() - start

        results[algo_name] = {
            'test_acc': result.test_accuracy,
            'time': elapsed,
            'iterations': result.n_iterations
        }

        print(f"  Test Accuracy: {result.test_accuracy:.4f}")
        print(f"  Tiempo: {elapsed:.2f}s")

    print("\n--- Resumen Comparativo ---")
    for algo, res in results.items():
        print(f"{algo}: acc={res['test_acc']:.4f}, tiempo={res['time']:.2f}s")

    return results


# =============================================================================
# 5. ARQUITECTURA ESCALADA
# =============================================================================

def example_scaled_architecture():
    """Ejemplo con diferentes factores de escala."""
    print_header("5. ARQUITECTURA ESCALADA")

    X_train, X_test, y_train, y_test = load_dataset('wine')
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    print(f"Input dim: {input_dim}")
    print("\nProbando diferentes factores de escala:")

    results = []

    for scale_factor in [0.5, 1.0, 1.5, 2.0]:
        hidden_layers = create_scaled_architecture(input_dim, scale_factor)
        n_params = sum([
            (input_dim + 1) * hidden_layers[0],
            (hidden_layers[0] + 1) * hidden_layers[1],
            (hidden_layers[1] + 1) * hidden_layers[2],
            (hidden_layers[2] + 1) * output_dim
        ])

        print(f"\n  Scale={scale_factor}: {hidden_layers} ({n_params} params)")

        config = TrainingConfig(
            hidden_layers=hidden_layers,
            n_particles=30,
            max_iters=60,
            random_state=42,
            verbose=False,
            save_best_model=False
        )

        trainer = Trainer(input_dim, output_dim, config)
        result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

        results.append({
            'scale': scale_factor,
            'layers': hidden_layers,
            'params': n_params,
            'acc': result.test_accuracy
        })

        print(f"    Test Accuracy: {result.test_accuracy:.4f}")

    print("\n--- Resumen ---")
    best = max(results, key=lambda x: x['acc'])
    print(f"Mejor escala: {best['scale']} con accuracy {best['acc']:.4f}")

    return results


# =============================================================================
# 6. GUARDADO Y CARGA DE MODELOS
# =============================================================================

def example_save_load():
    """Ejemplo de guardado y carga de modelos."""
    print_header("6. GUARDADO Y CARGA DE MODELOS")

    X_train, X_test, y_train, y_test = load_dataset('iris')
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    # Entrenar modelo
    config = TrainingConfig(
        hidden_layers=[16, 8],
        n_particles=30,
        max_iters=50,
        random_state=42,
        verbose=False,
        output_dir='./output/models',
        save_best_model=True
    )

    trainer = Trainer(input_dim, output_dim, config)
    result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    print(f"Modelo entrenado - Test Acc: {result.test_accuracy:.4f}")

    # Guardar explicitamente
    saved_path = trainer.save_model(filename='iris_model.pth')

    # Crear nuevo trainer y cargar
    new_trainer = Trainer(input_dim, output_dim, config)
    new_trainer.load_model(saved_path)

    # Predecir con modelo cargado
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    predictions = new_trainer.predict(X_test_t)

    loaded_acc = np.mean(predictions == y_test)
    print(f"Accuracy del modelo cargado: {loaded_acc:.4f}")

    return loaded_acc


# =============================================================================
# 7. METRICAS DETALLADAS
# =============================================================================

def example_detailed_metrics():
    """Ejemplo con metricas detalladas."""
    print_header("7. METRICAS DETALLADAS")

    X_train, X_test, y_train, y_test = load_dataset('iris')
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    config = TrainingConfig(
        hidden_layers=[20, 10],
        n_particles=40,
        max_iters=80,
        random_state=42,
        verbose=False,
        save_best_model=False
    )

    trainer = Trainer(input_dim, output_dim, config)
    result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    # Obtener predicciones
    predictions = trainer.predict(X_test)

    # Calcular metricas detalladas
    metrics_calc = MulticlassMetrics(
        class_names=['setosa', 'versicolor', 'virginica']
    )

    metrics_calc.print_summary(y_test, predictions)

    return result.test_accuracy


# =============================================================================
# 8. VISUALIZACION CON GRAFICAS
# =============================================================================

def example_plotting():
    """Ejemplo de generacion de graficas."""
    print_header("8. VISUALIZACION CON GRAFICAS")

    # Crear directorio de salida
    output_dir = './img/metric/examples'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directorio de graficas: {output_dir}")

    X_train, X_test, y_train, y_test = load_dataset('iris')
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    # Entrenar modelo con historial
    config = TrainingConfig(
        hidden_layers=[20, 10],
        n_particles=40,
        max_iters=80,
        alpha=(1.0, 0.5),
        random_state=42,
        verbose=True,
        save_best_model=False
    )

    trainer = Trainer(input_dim, output_dim, config)
    result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    # Obtener predicciones
    predictions = trainer.predict(X_test)

    # Generar nombre de archivo con timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Grafica 1: Matriz de Confusion
    print("\n--- Generando Matriz de Confusion ---")
    cm_path = os.path.join(
        output_dir,
        f"QPSO_iris_confusion_matrix_alpha_1.0-0.5_p40_i80_{timestamp}.png"
    )
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=predictions,
        class_names=['setosa', 'versicolor', 'virginica'],
        save_path=cm_path
    )

    # Grafica 2: Historial de Entrenamiento
    print("\n--- Generando Curvas de Entrenamiento ---")
    if hasattr(result, 'history') and result.history:
        hist_path = os.path.join(
            output_dir,
            f"QPSO_iris_training_history_alpha_1.0-0.5_p40_i80_{timestamp}.png"
        )
        plot_training_history(
            history=result.history,
            save_path=hist_path
        )
    else:
        print("  No hay historial disponible")

    print(f"\nGraficas guardadas en: {output_dir}")
    print(f"Test Accuracy: {result.test_accuracy:.4f}")

    return result.test_accuracy


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Ejecuta todos los ejemplos."""
    device = print_system_info()

    results = {}

    # Ejecutar ejemplos
    examples = [
        ("Uso Basico", example_basic_usage),
        ("Config Personalizada", example_custom_config),
        ("Cross-Validation", example_cross_validation),
        ("QPSO vs QDPSO", example_qpso_vs_qdpso),
        ("Arquitectura Escalada", example_scaled_architecture),
        ("Guardado/Carga", example_save_load),
        ("Metricas Detalladas", example_detailed_metrics),
        ("Visualizacion", example_plotting),
    ]

    for name, func in examples:
        try:
            result = func()
            results[name] = result
        except Exception as e:
            print(f"Error en {name}: {e}")
            results[name] = None

    # Resumen final
    print_header("RESUMEN DE TODOS LOS EJEMPLOS")
    for name, result in results.items():
        if result is not None:
            if isinstance(result, dict):
                print(f"{name}: Completado")
            elif isinstance(result, list):
                print(f"{name}: Completado ({len(result)} resultados)")
            else:
                print(f"{name}: {result:.4f}")
        else:
            print(f"{name}: Error")

    print("\nTodos los ejemplos completados.")
    print("Ver documentacion en docs/docs_qpso_nn.md para mas informacion.")


if __name__ == "__main__":
    main()

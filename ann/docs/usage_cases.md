# usage_cases.py - Ejemplos de Uso

[🏠 README](../../README.md) | [📚 Index](index.md) | [⬅️ HPO](main_hyperparameter_search.md) | **Usage Cases**

---

## Descripcion General

`usage_cases.py` es un archivo que contiene **7 ejemplos completos** que demuestran diferentes formas de usar el framework QPSO para entrenamiento de redes neuronales. Cada ejemplo ilustra un caso de uso especifico.

### Proposito

- Demostrar la API del framework con ejemplos practicos
- Mostrar diferentes niveles de configuracion (basico a avanzado)
- Ilustrar comparativas entre QPSO y QDPSO
- Proporcionar codigo listo para copiar y adaptar

---

## Ejecucion

```bash
# Activar entorno
conda activate pytorch_qpso_gpu

# Ejecutar todos los ejemplos
python ann/usage_cases.py
```

---

## Indice de Ejemplos

| # | Ejemplo | Descripcion | Dataset |
|---|---------|-------------|---------|
| 1 | [Uso Basico](#1-uso-basico) | Entrenamiento simple con QPSO | Iris |
| 2 | [Configuracion Personalizada](#2-configuracion-personalizada) | Trainer con parametros custom | Wine |
| 3 | [Cross-Validation](#3-cross-validation) | Evaluacion con K-Fold CV | Breast Cancer |
| 4 | [QPSO vs QDPSO](#4-qpso-vs-qdpso) | Comparativa de algoritmos | Digits |
| 5 | [Arquitectura Escalada](#5-arquitectura-escalada) | Diferentes factores de escala | Wine |
| 6 | [Guardado y Carga](#6-guardado-y-carga) | Persistencia de modelos | Iris |
| 7 | [Metricas Detalladas](#7-metricas-detalladas) | Evaluacion exhaustiva | Iris |
| 8 | [Visualizacion](#8-visualizacion-con-graficas) | Graficas de resultados | Iris |

---

## 1. Uso Basico

### Descripcion

Ejemplo minimo que muestra como entrenar una red neuronal con QPSO en el dataset Iris.

### Codigo

```python
def example_basic_usage():
    """Ejemplo basico de entrenamiento con QPSO."""

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
```

### Componentes Utilizados

- `load_dataset()` - Carga de datos
- `QPSOCompatibleANN` - Modelo de red neuronal
- `NNOptimizationConfig` - Configuracion del optimizador
- `QPSONNOptimizer` - Optimizador QPSO

### Salida Esperada

```
Dataset Iris: train=120, test=30
Features: 4, Clases: 3

Modelo creado:
QPSOCompatibleANN(
  (layers): Sequential(
    (0): Linear(in_features=4, out_features=16)
    (1): Tanh()
    (2): Linear(in_features=16, out_features=8)
    (3): Tanh()
    (4): Linear(in_features=8, out_features=3)
    (5): LogSoftmax()
  )
)

Iter 001: loss=1.0854, acc=0.3333, best=1.0854
Iter 010: loss=0.5432, acc=0.7500, best=0.5432
...

Resultados en Test:
  Loss: 0.123456
  Accuracy: 0.9667
```

---

## 2. Configuracion Personalizada

### Descripcion

Ejemplo que usa el `Trainer` de alto nivel con arquitectura escalada y configuracion detallada.

### Codigo

```python
def example_custom_config():
    """Ejemplo con configuracion personalizada."""

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
```

### Funcion `create_scaled_architecture`

```python
def create_scaled_architecture(input_dim: int, scale_factor: float = 0.5) -> List[int]:
    """
    Crea arquitectura con proporcion 3:2:1.

    Ejemplo con input_dim=13, scale_factor=0.5:
        Capa 1: 13 * 3 * 0.5 = 19.5 → 20
        Capa 2: 13 * 2 * 0.5 = 13.0 → 13
        Capa 3: 13 * 1 * 0.5 = 6.5  → 7

    Resultado: [20, 13, 7]
    """
```

### Componentes Utilizados

- `TrainingConfig` - Configuracion centralizada
- `Trainer` - Entrenador de alto nivel
- `create_scaled_architecture()` - Generador de arquitectura

---

## 3. Cross-Validation

### Descripcion

Ejemplo que implementa validacion cruzada con K-Fold para evaluacion robusta.

### Codigo

```python
def example_cross_validation():
    """Ejemplo con cross-validation."""

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
        n_folds=5,              # 5-fold CV
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
```

### Funcionamiento del CV

```
┌─────────────────────────────────────────────────────────────┐
│                    5-FOLD CROSS-VALIDATION                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Fold 1: [████████████████] [    ] → val_acc = 0.95        │
│  Fold 2: [████████████] [    ] [████] → val_acc = 0.93     │
│  Fold 3: [████████] [    ] [████████] → val_acc = 0.96     │
│  Fold 4: [████] [    ] [████████████] → val_acc = 0.94     │
│  Fold 5: [    ] [████████████████████] → val_acc = 0.95    │
│                                                             │
│  Media: 0.9460 +/- 0.0110                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Salida Esperada

```
Dataset Breast Cancer: 569 muestras
  Features: 30, Clases: 2

Fold 1/5:
  Train Accuracy: 0.9756
  Val Accuracy: 0.9512
...

Resultados por fold:
  Fold 1: val_acc=0.9512
  Fold 2: val_acc=0.9302
  Fold 3: val_acc=0.9535
  Fold 4: val_acc=0.9419
  Fold 5: val_acc=0.9535
```

---

## 4. QPSO vs QDPSO

### Descripcion

Ejemplo que compara el rendimiento de QPSO y QDPSO en el mismo dataset.

### Codigo

```python
def example_qpso_vs_qdpso():
    """Compara QPSO y QDPSO."""

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
```

### Tabla Comparativa

| Algoritmo | Parametro | Descripcion |
|-----------|-----------|-------------|
| QPSO | `use_qdpso=False` | Usa alpha con decay |
| QDPSO | `use_qdpso=True` | Usa factor g constante |

### Salida Esperada

```
Dataset Digits: 64 features, 10 clases
  Train: 1437, Test: 360

--- Entrenando con QPSO ---
  Test Accuracy: 0.9472
  Tiempo: 45.23s

--- Entrenando con QDPSO ---
  Test Accuracy: 0.9528
  Tiempo: 43.89s

--- Resumen Comparativo ---
QPSO: acc=0.9472, tiempo=45.23s
QDPSO: acc=0.9528, tiempo=43.89s
```

---

## 5. Arquitectura Escalada

### Descripcion

Ejemplo que prueba diferentes factores de escala para la arquitectura de red.

### Codigo

```python
def example_scaled_architecture():
    """Ejemplo con diferentes factores de escala."""

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
```

### Arquitecturas Generadas

| Scale | Capas Ocultas | Parametros |
|-------|---------------|------------|
| 0.5 | [20, 13, 7] | ~500 |
| 1.0 | [39, 26, 13] | ~1,800 |
| 1.5 | [59, 39, 20] | ~3,800 |
| 2.0 | [78, 52, 26] | ~6,200 |

### Salida Esperada

```
Input dim: 13

Probando diferentes factores de escala:

  Scale=0.5: [20, 13, 7] (527 params)
    Test Accuracy: 0.9444

  Scale=1.0: [39, 26, 13] (1827 params)
    Test Accuracy: 0.9722

  Scale=1.5: [59, 39, 20] (3796 params)
    Test Accuracy: 0.9722

  Scale=2.0: [78, 52, 26] (6288 params)
    Test Accuracy: 0.9444

--- Resumen ---
Mejor escala: 1.0 con accuracy 0.9722
```

---

## 6. Guardado y Carga

### Descripcion

Ejemplo que demuestra como guardar un modelo entrenado y cargarlo posteriormente.

### Codigo

```python
def example_save_load():
    """Ejemplo de guardado y carga de modelos."""

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
```

### Estructura de Archivos

```
./output/models/
├── iris_model.pth          # Modelo guardado
└── best_model_*.pth        # Modelos automaticos (si save_best_model=True)
```

### Contenido del Archivo .pth

```python
{
    'model_state_dict': {...},      # Pesos de la red
    'config': {...},                 # Configuracion usada
    'train_accuracy': 0.9714,
    'val_accuracy': 0.9333,
    'test_accuracy': 0.9667,
    'timestamp': '2024-01-15 10:30:00'
}
```

---

## 7. Metricas Detalladas

### Descripcion

Ejemplo que calcula y muestra metricas exhaustivas de evaluacion.

### Codigo

```python
def example_detailed_metrics():
    """Ejemplo con metricas detalladas."""

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
```

### Metricas Calculadas

| Metrica | Descripcion |
|---------|-------------|
| **Accuracy** | Proporcion de predicciones correctas |
| **Precision** | TP / (TP + FP) por clase |
| **Recall** | TP / (TP + FN) por clase |
| **F1-Score** | 2 * (Precision * Recall) / (Precision + Recall) |
| **Cohen's Kappa** | Acuerdo ajustado por azar |
| **Confusion Matrix** | Matriz de confusion completa |

### Salida Esperada

```
============================================================
                    RESUMEN DE METRICAS
============================================================

Metricas Globales:
  Accuracy:     0.9667
  Cohen Kappa:  0.9500

Metricas por Clase:
              Precision   Recall   F1-Score   Support
  setosa        1.0000   1.0000     1.0000        10
  versicolor    0.9091   1.0000     0.9524        10
  virginica     1.0000   0.9000     0.9474        10

  Macro Avg     0.9697   0.9667     0.9666        30
  Weighted Avg  0.9697   0.9667     0.9666        30

Matriz de Confusion:
              setosa  versicolor  virginica
  setosa          10           0          0
  versicolor       0          10          0
  virginica        0           1          9
```

---

## 8. Visualizacion con Graficas

### Descripcion

Ejemplo que genera graficas de resultados incluyendo matriz de confusion y curvas de entrenamiento.

### Codigo

```python
def example_plotting():
    """Ejemplo de generacion de graficas."""

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
    if hasattr(result, 'history') and result.history:
        hist_path = os.path.join(
            output_dir,
            f"QPSO_iris_training_history_alpha_1.0-0.5_p40_i80_{timestamp}.png"
        )
        plot_training_history(
            history=result.history,
            save_path=hist_path
        )

    return result.test_accuracy
```

### Graficas Generadas

#### 1. Matriz de Confusion
- **Archivo**: `QPSO_iris_confusion_matrix_alpha_1.0-0.5_p40_i80_YYYYMMDD_HHMMSS.png`
- **Contenido**: Heatmap con predicciones vs valores reales
- **Dependencias**: matplotlib, seaborn

#### 2. Curvas de Entrenamiento
- **Archivo**: `QPSO_iris_training_history_alpha_1.0-0.5_p40_i80_YYYYMMDD_HHMMSS.png`
- **Contenido**: Loss y Accuracy por iteracion (train y validation)
- **Dependencias**: matplotlib

### Estructura de Nombres de Archivo

```
{Optimizador}_{dataset}_{tipo}_{parametros}_{timestamp}.png

Ejemplos:
- QPSO_iris_confusion_matrix_alpha_1.0-0.5_p40_i80_20240115_103045.png
- QDPSO_wine_training_history_g_0.96_p50_i150_20240115_103512.png
```

### Directorio de Salida

```
./img/metric/
├── QPSO/           # Graficas de main_qpso.py
├── QDPSO/          # Graficas de main_qdpso.py
└── examples/       # Graficas de usage_cases.py
```

### Salida Esperada

```
======================================================================
 8. VISUALIZACION CON GRAFICAS
======================================================================
Directorio de graficas: ./img/metric/examples

Iter 001: loss=1.0854, acc=0.3333
...

--- Generando Matriz de Confusion ---
Figura guardada en: ./img/metric/examples/QPSO_iris_confusion_matrix_...png

--- Generando Curvas de Entrenamiento ---
Figura guardada en: ./img/metric/examples/QPSO_iris_training_history_...png

Graficas guardadas en: ./img/metric/examples
Test Accuracy: 0.9667
```

---

## Funcion Main

### Descripcion

La funcion `main()` ejecuta todos los ejemplos secuencialmente y genera un resumen final.

### Codigo

```python
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
```

### Salida Final

```
======================================================================
 RESUMEN DE TODOS LOS EJEMPLOS
======================================================================
Uso Basico: 0.9667
Config Personalizada: 0.9722
Cross-Validation: 0.9535
QPSO vs QDPSO: Completado
Arquitectura Escalada: Completado (4 resultados)
Guardado/Carga: 0.9667
Metricas Detalladas: 0.9667
Visualizacion: 0.9667

Todos los ejemplos completados.
Ver documentacion en docs/docs_qpso_nn.md para mas informacion.
```

---

## Diferencias con Benchmarks

| Aspecto | `usage_cases.py` | `main_qpso.py` / `main_qdpso.py` |
|---------|------------------|----------------------------------|
| **Proposito** | Ejemplos educativos | Benchmark sistematico |
| **Estructura** | 7 funciones independientes | Flujo unico por dataset |
| **Datasets** | Varios (iris, wine, digits, breast_cancer) | Fijos (iris, wine, breast_cancer) |
| **Configuracion** | Variable por ejemplo | Constante BENCHMARK_CONFIG |
| **Cross-validation** | Ejemplo opcional (#3) | Siempre incluido |
| **Salida** | Educativa con explicaciones | Tablas comparativas |

---

## Ver Tambien

- [main_qpso.md](main_qpso.md) - Benchmark con QPSO
- [main_qdpso.md](main_qdpso.md) - Benchmark con QDPSO
- [examples.md](examples.md) - Mas ejemplos detallados
- [trainers.md](trainers.md) - Documentacion del Trainer

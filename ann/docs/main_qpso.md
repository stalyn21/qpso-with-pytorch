# main_qpso.py - Benchmark QPSO

[🏠 README](../../README.md) | [📚 Index](index.md) | [⬅️ Examples](examples.md) | **QPSO Script** | [Next: QDPSO ➡️](main_qdpso.md)

---

## Descripcion General

`main_qpso.py` es un script de benchmark que evalua el rendimiento del algoritmo **QPSO (Quantum Particle Swarm Optimization)** para entrenar redes neuronales en datasets de clasificacion clasicos.

### Proposito

- Ejecutar benchmarks sistematicos en multiples datasets
- Evaluar el rendimiento del optimizador QPSO
- Generar metricas detalladas incluyendo cross-validation
- Proporcionar resultados comparables y reproducibles

---

## Ejecucion

```bash
# Activar entorno
conda activate pytorch_qpso_gpu

# Ejecutar benchmark
python ann/main_qpso.py
```

---

## Configuracion del Benchmark

El script define una configuracion global `BENCHMARK_CONFIG`:

```python
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
```

### Parametros Clave

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| `activation` | `'tanh'` | Funcion de activacion para capas ocultas |
| `output_activation` | `'softmax'` | Funcion de activacion para capa de salida |
| `n_particles` | `50` | Numero de particulas en el enjambre QPSO |
| `max_iters` | `150` | Maximo de iteraciones de optimizacion |
| `alpha` | `(1.0, 0.5)` | Factor alpha con decay lineal de 1.0 a 0.5 |
| `n_folds` | `4` | Numero de folds para cross-validation |
| `train_size` | `0.70` | Proporcion de datos para entrenamiento (70%) |
| `test_size` | `0.20` | Proporcion de datos para test (20%) |
| `val_size` | `0.10` | Proporcion de datos para validacion (10%) |
| `random_state` | `42` | Semilla aleatoria para reproducibilidad |
| `patience` | `50` | Iteraciones sin mejora para early stopping |

---

## Arquitectura de Red

La arquitectura de la red neuronal se genera dinamicamente segun la especificacion:

```
input -> input*3 -> input*2 -> output
```

### Funcion `get_architecture`

```python
def get_architecture(input_dim: int, output_dim: int) -> List[int]:
    """
    Genera arquitectura segun especificacion: input*3, input*2.

    Ejemplo para Iris (4 features, 3 clases):
        Entrada: 4
        Oculta 1: 12 (4 * 3)
        Oculta 2: 8  (4 * 2)
        Salida: 3
    """
    return [input_dim * 3, input_dim * 2]
```

### Ejemplos de Arquitectura por Dataset

| Dataset | Input | Oculta 1 | Oculta 2 | Output | Parametros |
|---------|-------|----------|----------|--------|------------|
| Iris | 4 | 12 | 8 | 3 | 195 |
| Wine | 13 | 39 | 26 | 3 | 1,607 |
| Breast Cancer | 30 | 90 | 60 | 2 | 8,432 |

---

## Datasets Evaluados

El benchmark evalua tres datasets de sklearn:

### 1. Iris
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Clases**: 3 (setosa, versicolor, virginica)
- **Muestras**: 150

### 2. Wine
- **Features**: 13 (alcohol, malic acid, ash, etc.)
- **Clases**: 3 (cultivadores de vino)
- **Muestras**: 178

### 3. Breast Cancer
- **Features**: 30 (caracteristicas de nucleos celulares)
- **Clases**: 2 (maligno, benigno)
- **Muestras**: 569

---

## Division de Datos

El script implementa una division estratificada en tres conjuntos:

```python
def load_and_split_dataset(
    name: str,
    train_size: float = 0.70,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42
) -> Tuple[X_train, X_val, X_test, y_train, y_val, y_test]:
```

### Proceso de Division

```
Dataset completo (100%)
         │
         ▼
    ┌────────────┐
    │ Primer     │───→ X_test, y_test (20%)
    │ Split      │
    └────────────┘
         │
         ▼
    X_temp (80%)
         │
         ▼
    ┌────────────┐
    │ Segundo    │───→ X_val, y_val (10%)
    │ Split      │
    └────────────┘
         │
         ▼
    X_train, y_train (70%)
```

### Caracteristicas

- **Estratificado**: Mantiene proporciones de clases en cada conjunto
- **Normalizado**: Usa `StandardScaler` para normalizar features
- **Reproducible**: Usa semilla fija `random_state=42`

---

## Flujo de Ejecucion

El benchmark sigue un flujo de 10 pasos para cada dataset:

```
┌────────────────────────────────────────────────────────────────┐
│                    FLUJO POR DATASET                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. CARGAR DATOS                                               │
│     └─→ load_and_split_dataset() → train/val/test             │
│                                                                │
│  2. DEFINIR ARQUITECTURA                                       │
│     └─→ get_architecture() → [input*3, input*2]               │
│                                                                │
│  3. CREAR MODELO                                               │
│     └─→ QPSOCompatibleANN(input, output, hidden_layers)       │
│                                                                │
│  4. CONFIGURAR OPTIMIZADOR QPSO                                │
│     └─→ NNOptimizationConfig(alpha=(1.0, 0.5), ...)           │
│     └─→ QPSONNOptimizer(model, config)                        │
│                                                                │
│  5. ENTRENAR                                                   │
│     └─→ optimizer.fit(X_train, y_train, X_val, y_val)         │
│                                                                │
│  6. EVALUAR                                                    │
│     └─→ optimizer.evaluate() → train/val/test metrics         │
│                                                                │
│  7. METRICAS DETALLADAS                                        │
│     └─→ MulticlassMetrics() → precision, recall, F1, kappa    │
│                                                                │
│  8. CROSS-VALIDATION                                           │
│     └─→ Trainer.fit_cv() → 4 folds                            │
│                                                                │
│  9. RESUMEN                                                    │
│     └─→ Compilar todos los resultados                         │
│                                                                │
│  10. GENERAR GRAFICAS                                          │
│      └─→ plot_confusion_matrix() → matriz de confusion        │
│      └─→ plot_training_history() → curvas de entrenamiento    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Optimizador QPSO

El script usa el optimizador `QPSONNOptimizer` con la siguiente configuracion:

```python
opt_config = NNOptimizationConfig(
    n_particles=config['n_particles'],      # 50 particulas
    max_iters=config['max_iters'],          # 150 iteraciones
    alpha=config['alpha'],                  # (1.0, 0.5) decay
    patience=config['patience'],            # 50 early stopping
    seed=config['random_state'],            # 42
    track_history=True                      # Registrar historial
)

optimizer = QPSONNOptimizer(model, config=opt_config)
```

### Algoritmo QPSO

El algoritmo QPSO usa las siguientes ecuaciones:

```
mbest = (1/N) * Σ pbest_i           # Mean best position
c = φ * pbest + (1-φ) * gbest      # Attractor point
L = α * |mbest - x|                 # Characteristic length
x_new = c ± L * ln(1/u)            # New position
```

### Decay de Alpha

```
α(t) = α_max - (α_max - α_min) * (t / T)
     = 1.0 - (1.0 - 0.5) * (t / 150)
```

Donde `t` es la iteracion actual y `T` es el maximo de iteraciones.

---

## Metricas Generadas

### Metricas por Conjunto

| Metrica | Descripcion |
|---------|-------------|
| `train_accuracy` | Accuracy en conjunto de entrenamiento |
| `val_accuracy` | Accuracy en conjunto de validacion |
| `test_accuracy` | Accuracy en conjunto de test |
| `train_loss` | Loss (NLLLoss) en entrenamiento |
| `val_loss` | Loss en validacion |
| `test_loss` | Loss en test |

### Metricas Detalladas (Test)

| Metrica | Descripcion |
|---------|-------------|
| `accuracy` | Proporcion de predicciones correctas |
| `precision (macro)` | Precision promedio por clase |
| `recall (macro)` | Recall promedio por clase |
| `f1_score (macro)` | F1-Score promedio por clase |
| `cohen_kappa` | Coeficiente Kappa de Cohen |
| `confusion_matrix` | Matriz de confusion |

### Metricas de Cross-Validation

| Metrica | Descripcion |
|---------|-------------|
| `cv_mean` | Media del accuracy en los folds |
| `cv_std` | Desviacion estandar del accuracy |
| `fold_accs` | Lista de accuracy por fold |

---

## Salida del Script

### Ejemplo de Salida

```
======================================================================
 INFORMACION DEL SISTEMA
======================================================================
PyTorch version: 2.0.1
CUDA disponible: True
GPU: NVIDIA GeForce RTX 3080
CUDA version: 11.8
Dispositivo a usar: cuda
Fecha: 2024-01-15 10:30:00

======================================================================
 CONFIGURACION DEL BENCHMARK
======================================================================
  activation: tanh
  output_activation: softmax
  n_particles: 50
  max_iters: 150
  alpha: (1.0, 0.5)
  n_folds: 4
  ...

======================================================================
 BENCHMARK: IRIS
======================================================================

--- 1. Cargando datos ---
Dataset: iris
  Input dim: 4
  Output dim: 3
  Train samples: 105 (70%)
  Val samples: 15 (10%)
  Test samples: 30 (20%)
  ...

======================================================================
 RESUMEN FINAL DEL BENCHMARK
======================================================================

Dataset         Arch                 Params     Test Acc     CV Acc             Time
------------------------------------------------------------------------------------------
iris            4->[12, 8]->3        195        0.9667       0.9524 +/- 0.0311  12.45s
wine            13->[39, 26]->3      1,607      0.9722       0.9601 +/- 0.0289  45.23s
breast_cancer   30->[90, 60]->2      8,432      0.9649       0.9512 +/- 0.0198  89.12s
------------------------------------------------------------------------------------------

Mejor resultado: wine con 0.9722 accuracy
```

---

## Estructura de Resultados

El script retorna un diccionario con resultados detallados:

```python
results = {
    'dataset': 'iris',
    'input_dim': 4,
    'output_dim': 3,
    'hidden_layers': [12, 8],
    'n_params': 195,
    'train_samples': 105,
    'val_samples': 15,
    'test_samples': 30,
    'train_accuracy': 0.9714,
    'val_accuracy': 0.9333,
    'test_accuracy': 0.9667,
    'train_loss': 0.0856,
    'val_loss': 0.1234,
    'test_loss': 0.0987,
    'cv_mean': 0.9524,
    'cv_std': 0.0311,
    'cv_folds': [0.96, 0.92, 0.96, 0.96],
    'detailed_metrics': {...},
    'training_time': 8.45,
    'cv_time': 4.00,
    'total_time': 12.45,
    'iterations': 150,
    'convergence_reason': 'max_iterations'
}
```

---

## Diferencias con main_qdpso.py

| Aspecto | main_qpso.py | main_qdpso.py |
|---------|--------------|---------------|
| **Optimizador** | `QPSONNOptimizer` | `QDPSONNOptimizer` |
| **Parametro clave** | `alpha: (1.0, 0.5)` | `g: 0.96` |
| **Algoritmo** | QPSO con mbest | QDPSO con factor g |
| **Ecuacion L** | `L = α * \|mbest - x\|` | `L = (1/g) * \|x - c\|` |
| **Imports** | `from ann.optimizers import QPSONNOptimizer` | `from ann.optimizers import QDPSONNOptimizer` |

---

## Personalizacion

### Cambiar Datasets

```python
# Modificar la lista de datasets
DATASETS = ['iris', 'wine', 'breast_cancer', 'digits']
```

### Cambiar Arquitectura

```python
def get_architecture(input_dim: int, output_dim: int) -> List[int]:
    # Arquitectura personalizada
    return [input_dim * 4, input_dim * 2, input_dim]
```

### Cambiar Parametros QPSO

```python
BENCHMARK_CONFIG = {
    ...
    'n_particles': 100,         # Mas particulas
    'max_iters': 300,           # Mas iteraciones
    'alpha': (1.2, 0.4),        # Diferente decay
    'patience': 100,            # Mas paciencia
}
```

---

## Graficas Generadas

El script genera automaticamente **5 tipos de graficas** para cada dataset evaluado.

### Directorio de Salida

```
./img/metric/QPSO/
├── QPSO_iris_confusion_matrix_alpha_1.0-0.5_p50_i150_YYYYMMDD_HHMMSS.png
├── QPSO_iris_loss_curves_alpha_1.0-0.5_p50_i150_YYYYMMDD_HHMMSS.png
├── QPSO_iris_accuracy_curves_alpha_1.0-0.5_p50_i150_YYYYMMDD_HHMMSS.png
├── QPSO_iris_training_summary_alpha_1.0-0.5_p50_i150_YYYYMMDD_HHMMSS.png
├── QPSO_iris_cv_summary_alpha_1.0-0.5_p50_i150_YYYYMMDD_HHMMSS.png
├── QPSO_wine_confusion_matrix_...
├── QPSO_wine_loss_curves_...
├── QPSO_wine_accuracy_curves_...
├── QPSO_wine_training_summary_...
├── QPSO_wine_cv_summary_...
├── QPSO_breast_cancer_confusion_matrix_...
├── QPSO_breast_cancer_loss_curves_...
├── QPSO_breast_cancer_accuracy_curves_...
├── QPSO_breast_cancer_training_summary_...
└── QPSO_breast_cancer_cv_summary_...
```

### Estructura del Nombre de Archivo

```
QPSO_{dataset}_{tipo}_{alpha}_{particulas}_{iteraciones}_{timestamp}.png

Componentes:
- QPSO: Identificador del optimizador
- dataset: iris, wine, breast_cancer
- tipo: confusion_matrix, loss_curves, accuracy_curves, training_summary, cv_summary
- alpha: alpha_1.0-0.5 (valores de decay)
- particulas: p50 (numero de particulas)
- iteraciones: i150 (max iteraciones)
- timestamp: YYYYMMDD_HHMMSS (fecha y hora)
```

### Graficas Generadas

| # | Tipo | Descripcion | Contenido |
|---|------|-------------|-----------|
| 1 | `confusion_matrix` | Matriz de confusion | Heatmap de predicciones vs valores reales |
| 2 | `loss_curves` | Curvas de perdida | Train loss y Validation loss por iteracion |
| 3 | `accuracy_curves` | Curvas de accuracy | Train acc y Validation acc por iteracion |
| 4 | `training_summary` | Resumen completo | 4 subgraficas: curvas + barras comparativas Train/Val/Test |
| 5 | `cv_summary` | Resumen CV | Accuracy y Loss por fold con media |

### Descripcion de Cada Grafica

#### 1. Matriz de Confusion (`confusion_matrix`)
- Heatmap con colores que muestran predicciones correctas e incorrectas
- Etiquetas de clases en ejes

#### 2. Curvas de Loss (`loss_curves`)
- Eje X: Iteracion
- Eje Y: Loss
- Linea azul: Train Loss
- Linea roja: Validation Loss
- Linea vertical verde: Mejor iteracion

#### 3. Curvas de Accuracy (`accuracy_curves`)
- Eje X: Iteracion
- Eje Y: Accuracy (0-1)
- Linea azul: Train Accuracy
- Linea roja: Validation Accuracy

#### 4. Resumen de Entrenamiento (`training_summary`)
Panel 2x2:
- Superior izquierda: Curvas de Loss
- Superior derecha: Curvas de Accuracy
- Inferior izquierda: Barras de Loss final (Train/Val/Test)
- Inferior derecha: Barras de Accuracy final (Train/Val/Test)

#### 5. Resumen Cross-Validation (`cv_summary`)
Panel 1x2:
- Izquierda: Barras de Accuracy por Fold (Train vs Val)
- Derecha: Barras de Loss por Fold (Train vs Val)
- Linea roja: Media de validacion

---

## Ver Tambien

- [main_qdpso.md](main_qdpso.md) - Benchmark con QDPSO
- [usage_cases.md](usage_cases.md) - Ejemplos de uso variados
- [optimizers.md](optimizers.md) - Documentacion del optimizador QPSO
- [trainers.md](trainers.md) - Documentacion del Trainer

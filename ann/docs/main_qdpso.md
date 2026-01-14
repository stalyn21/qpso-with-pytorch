# main_qdpso.py - Benchmark QDPSO

[🏠 README](../../README.md) | [📚 Index](index.md) | [⬅️ QPSO](main_qpso.md) | **QDPSO Script** | [Next: MCW ➡️](main_mcw.md)

---

## Descripcion General

`main_qdpso.py` es un script de benchmark que evalua el rendimiento del algoritmo **QDPSO (Quantum Delta Particle Swarm Optimization)** para entrenar redes neuronales en datasets de clasificacion clasicos.

### Proposito

- Ejecutar benchmarks sistematicos usando el optimizador QDPSO
- Comparar resultados con QPSO usando la misma metodologia
- Evaluar el comportamiento del factor `g` en la optimizacion
- Generar metricas detalladas incluyendo cross-validation

---

## Ejecucion

```bash
# Activar entorno
conda activate pytorch_qpso_gpu

# Ejecutar benchmark QDPSO
python ann/main_qdpso.py
```

---

## Configuracion del Benchmark

El script define una configuracion global `BENCHMARK_CONFIG`:

```python
BENCHMARK_CONFIG = {
    'optimizer': 'QDPSO',           # Tipo de optimizador
    'activation': 'tanh',           # Activacion capas ocultas
    'output_activation': 'softmax', # Activacion capa salida
    'n_particles': 50,              # Numero de particulas
    'max_iters': 150,               # Iteraciones maximas
    'g': 0.96,                      # Factor g para QDPSO
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
| `optimizer` | `'QDPSO'` | Identificador del algoritmo usado |
| `activation` | `'tanh'` | Funcion de activacion para capas ocultas |
| `output_activation` | `'softmax'` | Funcion de activacion para capa de salida |
| `n_particles` | `50` | Numero de particulas en el enjambre |
| `max_iters` | `150` | Maximo de iteraciones de optimizacion |
| `g` | `0.96` | **Factor g para QDPSO** (parametro clave) |
| `n_folds` | `4` | Numero de folds para cross-validation |
| `train_size` | `0.70` | Proporcion de datos para entrenamiento (70%) |
| `test_size` | `0.20` | Proporcion de datos para test (20%) |
| `val_size` | `0.10` | Proporcion de datos para validacion (10%) |
| `random_state` | `42` | Semilla aleatoria para reproducibilidad |
| `patience` | `50` | Iteraciones sin mejora para early stopping |

---

## Factor g en QDPSO

El parametro `g` es el factor de control clave en QDPSO que determina el comportamiento de las particulas.

### Ecuacion de Movimiento QDPSO

```
c = (u1 * pbest + u2 * gbest) / (u1 + u2)    # Attractor point
L = (1/g) * |x - c|                           # Characteristic length
x_new = c ± L * ln(1/u)                       # New position
```

### Efecto del Factor g

| Valor de g | Comportamiento |
|------------|----------------|
| `g < 1.0` | Exploracion amplia (particulas se dispersan mas) |
| `g = 1.0` | Comportamiento equilibrado |
| `g > 1.0` | Explotacion intensa (particulas convergen mas rapido) |

### Valor Recomendado

```python
g = 0.96  # Balance optimo entre exploracion y explotacion
```

Este valor permite:
- Suficiente exploracion para evitar optimos locales
- Convergencia adecuada hacia la solucion optima
- Estabilidad numerica en el proceso de optimizacion

---

## Diferencias con QPSO

### Comparativa de Algoritmos

| Aspecto | QPSO | QDPSO |
|---------|------|-------|
| **Parametro clave** | `alpha` (decay) | `g` (constante) |
| **Valor tipico** | `(1.0, 0.5)` | `0.96` |
| **Calculo de L** | `L = α * \|mbest - x\|` | `L = (1/g) * \|x - c\|` |
| **Usa mbest** | Si | No |
| **Adaptabilidad** | Decay durante ejecucion | Factor constante |
| **Complejidad** | Mayor (calcula mbest) | Menor |

### Diferencia Conceptual

```
QPSO:
    mbest = promedio de todas las mejores posiciones personales
    L depende de la distancia a mbest

QDPSO:
    No usa mbest
    L depende directamente de la distancia al attractor c
    Factor g controla la magnitud del salto cuantico
```

---

## Arquitectura de Red

Identica a `main_qpso.py`:

```python
def get_architecture(input_dim: int, output_dim: int) -> List[int]:
    return [input_dim * 3, input_dim * 2]
```

### Ejemplos por Dataset

| Dataset | Arquitectura | Parametros |
|---------|--------------|------------|
| Iris | 4 -> [12, 8] -> 3 | 195 |
| Wine | 13 -> [39, 26] -> 3 | 1,607 |
| Breast Cancer | 30 -> [90, 60] -> 2 | 8,432 |

---

## Datasets Evaluados

El benchmark evalua los mismos tres datasets de sklearn:

1. **Iris**: 4 features, 3 clases, 150 muestras
2. **Wine**: 13 features, 3 clases, 178 muestras
3. **Breast Cancer**: 30 features, 2 clases, 569 muestras

---

## Flujo de Ejecucion

El flujo sigue 10 pasos (similar a `main_qpso.py` pero usa QDPSO):

```
┌────────────────────────────────────────────────────────────────┐
│                    FLUJO POR DATASET (QDPSO)                   │
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
│  4. CONFIGURAR OPTIMIZADOR QDPSO  ← Diferencia clave          │
│     └─→ NNOptimizationConfig(g=0.96, ...)                     │
│     └─→ QDPSONNOptimizer(model, config)                       │
│                                                                │
│  5. ENTRENAR CON QDPSO                                         │
│     └─→ optimizer.fit(X_train, y_train, X_val, y_val)         │
│                                                                │
│  6. EVALUAR                                                    │
│     └─→ optimizer.evaluate() → train/val/test metrics         │
│                                                                │
│  7. METRICAS DETALLADAS                                        │
│     └─→ MulticlassMetrics() → precision, recall, F1, kappa    │
│                                                                │
│  8. CROSS-VALIDATION                                           │
│     └─→ Trainer.fit_cv(use_qdpso=True) → 4 folds              │
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

## Optimizador QDPSO

El script usa el optimizador `QDPSONNOptimizer`:

```python
opt_config = NNOptimizationConfig(
    n_particles=config['n_particles'],      # 50 particulas
    max_iters=config['max_iters'],          # 150 iteraciones
    g=config['g'],                          # 0.96 (factor g)
    patience=config['patience'],            # 50 early stopping
    seed=config['random_state'],            # 42
    track_history=True                      # Registrar historial
)

# Usar QDPSONNOptimizer en lugar de QPSONNOptimizer
optimizer = QDPSONNOptimizer(model, config=opt_config)
```

### Imports Especificos

```python
from ann.optimizers import QDPSONNOptimizer  # QDPSO en lugar de QPSO
```

---

## Configuracion de Cross-Validation

Para CV, el Trainer se configura con `use_qdpso=True`:

```python
trainer_config = TrainingConfig(
    hidden_layers=hidden_layers,
    activation=config['activation'],
    n_particles=config['n_particles'],
    max_iters=config['max_iters'],
    g=config['g'],              # Factor g
    use_qdpso=True,             # Activar QDPSO
    n_folds=config['n_folds'],
    patience=config['patience'],
    random_state=config['random_state'],
    verbose=False,
    save_best_model=False
)
```

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
Fecha: 2024-01-15 10:45:00

======================================================================
 CONFIGURACION DEL BENCHMARK QDPSO
======================================================================
  optimizer: QDPSO
  activation: tanh
  output_activation: softmax
  n_particles: 50
  max_iters: 150
  g: 0.96
  n_folds: 4
  ...

======================================================================
 BENCHMARK QDPSO: IRIS
======================================================================

--- 4. Configurando optimizador QDPSO ---
  Optimizador: QDPSO
  Particulas: 50
  Iteraciones max: 150
  Factor g: 0.96
  Patience: 50

--- 5. Entrenando modelo con QDPSO ---
...

======================================================================
 RESUMEN FINAL DEL BENCHMARK QDPSO
======================================================================

Fecha: 2024-01-15 10:50:00
Dispositivo: cuda
Optimizador: QDPSO (g=0.96)
Tiempo total: 145.67s

------------------------------------------------------------------------------------------
Dataset         Arch                 Params     Test Acc     CV Acc             Time
------------------------------------------------------------------------------------------
iris            4->[12, 8]->3        195        0.9667       0.9476 +/- 0.0356  11.89s
wine            13->[39, 26]->3      1,607      0.9722       0.9650 +/- 0.0245  44.56s
breast_cancer   30->[90, 60]->2      8,432      0.9561       0.9488 +/- 0.0212  89.22s
------------------------------------------------------------------------------------------

Mejor resultado: wine con 0.9722 accuracy

BENCHMARK QDPSO COMPLETADO
```

---

## Estructura de Resultados

El script retorna un diccionario con resultados detallados incluyendo informacion especifica de QDPSO:

```python
results = {
    'dataset': 'iris',
    'optimizer': 'QDPSO',           # Identificador del optimizador
    'g': 0.96,                      # Factor g usado
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
    'train_loss': 0.0892,
    'val_loss': 0.1278,
    'test_loss': 0.1012,
    'cv_mean': 0.9476,
    'cv_std': 0.0356,
    'cv_folds': [0.92, 0.96, 0.96, 0.92],
    'detailed_metrics': {...},
    'training_time': 8.12,
    'cv_time': 3.77,
    'total_time': 11.89,
    'iterations': 150,
    'convergence_reason': 'max_iterations'
}
```

---

## Comparativa QPSO vs QDPSO

### Tabla Comparativa de Codigo

| Aspecto | `main_qpso.py` | `main_qdpso.py` |
|---------|----------------|-----------------|
| **Import** | `from ann.optimizers import QPSONNOptimizer` | `from ann.optimizers import QDPSONNOptimizer` |
| **Config** | `'alpha': (1.0, 0.5)` | `'g': 0.96` |
| **Optimizer** | `QPSONNOptimizer(model, config)` | `QDPSONNOptimizer(model, config)` |
| **Trainer** | Sin `use_qdpso` | `use_qdpso=True` |
| **Header** | `BENCHMARK: {dataset}` | `BENCHMARK QDPSO: {dataset}` |
| **Resumen** | `RESUMEN FINAL DEL BENCHMARK` | `RESUMEN FINAL DEL BENCHMARK QDPSO` |

### Diferencias en la Configuracion

```python
# main_qpso.py
opt_config = NNOptimizationConfig(
    alpha=config['alpha'],  # (1.0, 0.5)
    ...
)

# main_qdpso.py
opt_config = NNOptimizationConfig(
    g=config['g'],  # 0.96
    ...
)
```

---

## Cuando Usar QDPSO vs QPSO

### Preferir QDPSO cuando:

- Se necesita un parametro mas simple (solo `g` vs `alpha` con decay)
- La convergencia con QPSO es inestable
- Se requiere comportamiento mas predecible
- El problema tiene un paisaje de fitness suave

### Preferir QPSO cuando:

- Se beneficia del calculo de mbest (problemas multimodales)
- El decay de alpha mejora la exploracion inicial
- La implementacion original de Sun et al. es requerida

---

## Personalizacion

### Ajustar Factor g

```python
BENCHMARK_CONFIG = {
    ...
    'g': 0.90,  # Mas exploracion
    # o
    'g': 0.98,  # Mas explotacion
}
```

### Experimentar con Diferentes Valores

```python
# Valores tipicos para probar
g_values = [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]

for g in g_values:
    config = {**BENCHMARK_CONFIG, 'g': g}
    results = run_benchmark('iris', config)
    print(f"g={g}: accuracy={results['test_accuracy']:.4f}")
```

---

## Graficas Generadas

El script genera automaticamente **5 tipos de graficas** para cada dataset evaluado.

### Directorio de Salida

```
./img/metric/QDPSO/
├── QDPSO_iris_confusion_matrix_g_0.96_p50_i150_YYYYMMDD_HHMMSS.png
├── QDPSO_iris_loss_curves_g_0.96_p50_i150_YYYYMMDD_HHMMSS.png
├── QDPSO_iris_accuracy_curves_g_0.96_p50_i150_YYYYMMDD_HHMMSS.png
├── QDPSO_iris_training_summary_g_0.96_p50_i150_YYYYMMDD_HHMMSS.png
├── QDPSO_iris_cv_summary_g_0.96_p50_i150_YYYYMMDD_HHMMSS.png
├── QDPSO_wine_confusion_matrix_...
├── QDPSO_wine_loss_curves_...
├── QDPSO_wine_accuracy_curves_...
├── QDPSO_wine_training_summary_...
├── QDPSO_wine_cv_summary_...
├── QDPSO_breast_cancer_confusion_matrix_...
├── QDPSO_breast_cancer_loss_curves_...
├── QDPSO_breast_cancer_accuracy_curves_...
├── QDPSO_breast_cancer_training_summary_...
└── QDPSO_breast_cancer_cv_summary_...
```

### Estructura del Nombre de Archivo

```
QDPSO_{dataset}_{tipo}_{g}_{particulas}_{iteraciones}_{timestamp}.png

Componentes:
- QDPSO: Identificador del optimizador
- dataset: iris, wine, breast_cancer
- tipo: confusion_matrix, loss_curves, accuracy_curves, training_summary, cv_summary
- g: g_0.96 (valor del factor g)
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

- [main_qpso.md](main_qpso.md) - Benchmark con QPSO
- [usage_cases.md](usage_cases.md) - Ejemplos de uso variados
- [optimizers.md](optimizers.md) - Documentacion del optimizador QDPSO
- [trainers.md](trainers.md) - Documentacion del Trainer

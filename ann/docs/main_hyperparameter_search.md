# Documentacion: main_hyperparameter_search.py

[🏠 README](../../README.md) | [📚 Index](index.md) | [⬅️ Training Type](main_training_type.md) | **HPO with Optuna** | [Next: Usage Cases ➡️](usage_cases.md)

---

## Descripcion General

Script de **busqueda automatica de hiperparametros** usando **Optuna** para encontrar la configuracion optima de QPSO/QDPSO en el dataset MCW.

### Por que Hyperparameter Optimization (HPO)?

El rendimiento de QPSO/QDPSO depende criticamente de sus hiperparametros. Encontrar manualmente la mejor configuracion es:
- **Tedioso**: Hay decenas de combinaciones posibles
- **Suboptimo**: Los humanos tendemos a probar pocas configuraciones
- **Sesgado**: Nos inclinamos hacia valores "tipicos" sin explorar

---

## Tabla de Contenidos

1. [Por que Optuna](#por-que-optuna)
2. [Espacio de Busqueda](#espacio-de-busqueda)
3. [Algoritmo de Busqueda](#algoritmo-de-busqueda)
4. [Pruning (Early Stopping)](#pruning-early-stopping)
5. [Funcion Objetivo](#funcion-objetivo)
6. [Uso del Script](#uso-del-script)
7. [Interpretacion de Resultados](#interpretacion-de-resultados)
8. [Configuracion Avanzada](#configuracion-avanzada)

---

## Por que Optuna

### Comparativa de Metodos HPO

| Metodo | Eficiencia | Inteligencia | Paralelismo | Pruning |
|--------|------------|--------------|-------------|---------|
| **Grid Search** | Baja | Ninguna | Si | No |
| **Random Search** | Media | Ninguna | Si | No |
| **Bayesian (Optuna)** | Alta | Alta | Si | Si |
| **Hyperband** | Alta | Media | Si | Si |

### Ventajas de Optuna

1. **TPE Sampler (Tree-structured Parzen Estimator)**
   - Construye modelo probabilistico del espacio de busqueda
   - Sugiere configuraciones prometedoras basado en evaluaciones previas
   - Mas eficiente que random search (encuentra mejores configuraciones en menos trials)

2. **MedianPruner**
   - Detiene trials malos temprano (ahorra tiempo)
   - Compara con la mediana de trials anteriores en el mismo step
   - Reduce tiempo de busqueda hasta 50%

3. **Flexibilidad**
   - Soporta parametros categoricos, enteros, flotantes
   - Espacios de busqueda condicionales (ej: `layer_decay` solo si `strategy='weighted'`)
   - Facil integracion con PyTorch

4. **Visualizacion**
   - Importancia de parametros
   - Historia de optimizacion
   - Coordenadas paralelas
   - Contour plots

```
Eficiencia de Optuna vs otros metodos:

Configuraciones probadas vs Accuracy encontrado:

      ^
  Acc |                    *** Optuna (TPE)
      |               ****
      |           ****
      |        ***    --- Random Search
      |     ***   ----
      |   **  ----
      |  * ----
      | *---
      +-------------------------> Trials
        10   50   100  200

Optuna encuentra mejores configuraciones con menos evaluaciones
```

---

## Espacio de Busqueda

### Hiperparametros Optimizados

```python
SEARCH_SPACE = {
    # Optimizador
    'optimizer': ['QPSO', 'QDPSO'],

    # Parametros QPSO
    'alpha_start': (0.7, 1.0),       # Alpha inicial
    'alpha_end': (0.3, 0.7),         # Alpha final

    # Parametros QDPSO
    'g': (0.90, 0.99),               # Factor g

    # Enjambre
    'n_particles': (20, 80),         # Particulas
    'max_iters': (50, 300),          # Iteraciones

    # Arquitectura
    'n_hidden_layers': (1, 3),       # Capas ocultas
    'neurons_multiplier': (1.5, 4.0),  # Multiplicador neuronas
    'neuron_decay': (0.5, 0.9),      # Decaimiento entre capas

    # Estrategia
    'strategy': ['forward', 'weighted', 'layerwise'],

    # Weighted
    'layer_decay': (0.5, 0.9),
    'regularization': (0.001, 0.1),

    # Layerwise
    'iters_per_layer': (20, 80),
    'fine_tune_iters': (20, 80),

    # Otros
    'weight_bound': (0.5, 2.0),
    'patience': (20, 60),
}
```

### Justificacion de Rangos

| Parametro | Rango | Justificacion |
|-----------|-------|---------------|
| `alpha_start` | 0.7-1.0 | Valores altos al inicio = exploracion |
| `alpha_end` | 0.3-0.7 | Valores bajos al final = explotacion |
| `g` | 0.90-0.99 | Literatura recomienda ~0.96 |
| `n_particles` | 20-80 | Balance entre diversidad y costo |
| `neurons_multiplier` | 1.5-4.0 | Capas mas grandes que input |
| `layer_decay` | 0.5-0.9 | Capas progresivamente mas pequeñas |

---

## Algoritmo de Busqueda

### TPE (Tree-structured Parzen Estimator)

TPE modela la probabilidad de que una configuracion sea buena:

```
P(config | bueno) vs P(config | malo)

1. Divide trials en "buenos" (top 25%) y "malos"
2. Ajusta KDE (Kernel Density Estimator) para cada grupo
3. Sugiere configuraciones que maximizan:

   EI(config) = P(config | bueno) / P(config | malo)

   (Expected Improvement)
```

### Flujo de Optuna

```
┌─────────────────────────────────────────────────────────────┐
│                    OPTUNA OPTIMIZATION                       │
└─────────────────────────────────────────────────────────────┘

  ┌─────────────┐
  │ Trial 1-5   │ ──→ Warmup (random sampling)
  └─────────────┘
         │
         ▼
  ┌─────────────┐
  │ TPE Model   │ ──→ Construir modelo de configuraciones buenas
  └─────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ For trial in range(n_trials):                           │
  │   1. TPE sugiere configuracion                          │
  │   2. Evaluar (3-fold CV)                                │
  │   3. MedianPruner decide si continuar                   │
  │   4. Si no prune: registrar score                       │
  │   5. Actualizar modelo TPE                              │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────┐
  │ Best Config │ ──→ Mejor configuracion encontrada
  └─────────────┘
```

---

## Pruning (Early Stopping)

### MedianPruner

Detiene trials que van mal comparando con trials anteriores:

```
Trial actual vs Mediana de trials anteriores:

Fold 1: Score = 0.65
        Mediana anterior = 0.72
        ¿0.65 < 0.72 - margen? → Si → PRUNE

┌────────────────────────────────────────────────────────┐
│                    PRUNING EXAMPLE                      │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Trial 1: [0.70] [0.72] [0.74] → Score: 0.72 ✓        │
│  Trial 2: [0.68] [0.70] [0.73] → Score: 0.70 ✓        │
│  Trial 3: [0.60] ✗ PRUNED (< mediana 0.71)            │
│  Trial 4: [0.75] [0.78] [0.80] → Score: 0.78 ✓        │
│  Trial 5: [0.55] ✗ PRUNED                              │
│                                                         │
│  Ahorro: 2 trials completos (~40% tiempo)              │
└────────────────────────────────────────────────────────┘
```

### Configuracion del Pruner

```python
pruner = MedianPruner(
    n_startup_trials=5,   # No prune los primeros 5 trials
    n_warmup_steps=1      # No prune hasta fold 2
)
```

---

## Funcion Objetivo

### Estructura

```python
def objective(trial: Trial) -> float:
    """
    1. Sugerir hiperparametros
    2. Cross-validation (3 folds)
    3. Reportar scores intermedios (para pruning)
    4. Retornar F1-score promedio
    """

    # 1. Sugerir parametros
    optimizer = trial.suggest_categorical('optimizer', ['QPSO', 'QDPSO'])
    n_particles = trial.suggest_int('n_particles', 20, 80)
    ...

    # 2. Cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        # Entrenar modelo
        model = create_model(...)
        result = train(...)

        # Evaluar
        score = f1_score(y_val, predictions)
        fold_scores.append(score)

        # 3. Reportar para pruning
        trial.report(np.mean(fold_scores), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 4. Retornar promedio
    return np.mean(fold_scores)
```

### Por que F1-Score Macro?

| Metrica | Problema con datos desbalanceados |
|---------|-----------------------------------|
| Accuracy | Sesgada hacia clase mayoritaria |
| Precision | Ignora falsos negativos |
| Recall | Ignora falsos positivos |
| **F1 Macro** | Equilibra todas las clases |

```
F1 Macro = (1/n_clases) * Σ F1_clase_i

Para MCW (4 clases):
F1 Macro = (F1_cloudy + F1_rain + F1_shine + F1_sunrise) / 4
```

---

## Uso del Script

### Instalacion de Dependencias

```bash
pip install optuna optuna-dashboard plotly kaleido
```

### Ejecucion Basica

```bash
conda activate pytorch_qpso_gpu
python ann/main_hyperparameter_search.py
```

### Configuracion Personalizada

Editar `SearchConfig` en el script:

```python
@dataclass
class SearchConfig:
    # Dataset
    dataset_path: str = './data/img/mcw'
    reduction_method: str = 'isomap'
    n_components: int = 7

    # Estudio
    n_trials: int = 100      # Mas trials = mejor busqueda
    timeout: int = 3600      # 1 hora maximo
    n_jobs: int = 1          # Paralelismo

    # Validacion
    n_folds: int = 3         # Folds CV
```

### Salida Esperada

```
======================================================================
 HYPERPARAMETER SEARCH - QPSO/QDPSO
======================================================================

Fecha: 2024-01-15 10:30:00
Dispositivo: CUDA
GPU: NVIDIA GeForce GTX 1050 Ti

======================================================================
 EJECUTANDO BUSQUEDA DE HIPERPARAMETROS
======================================================================

[I 2024-01-15 10:30:15] Trial 0 finished with value: 0.7234
[I 2024-01-15 10:31:02] Trial 1 finished with value: 0.6891
[I 2024-01-15 10:31:45] Trial 2 pruned.
...
[I 2024-01-15 12:45:30] Trial 99 finished with value: 0.8456

======================================================================
 MEJOR CONFIGURACION ENCONTRADA
======================================================================

  Optimizador: QDPSO
  g: 0.9534

  Estrategia: LAYERWISE
  Iters per Layer: 45
  Fine-tune Iters: 60

  Particulas: 52
  Max Iteraciones: 180
  Arquitectura: [28, 18, 11]
  Parametros: 1,247

  --- Metricas en Test ---
  Accuracy: 0.8750
  F1-Score: 0.8712
  Precision: 0.8801
  Recall: 0.8634
  Kappa: 0.8312
```

---

## Interpretacion de Resultados

### Archivos Generados

```
results/hyperparameter_search/
├── best_params_20240115_103000.json    # Mejor configuracion
├── top_10_configs_20240115_103000.json # Top 10 configuraciones
├── trials_history_20240115_103000.csv  # Historial completo
├── optimization_history.png            # Grafica de convergencia
├── param_importances.png               # Importancia de parametros
├── parallel_coordinate.png             # Coordenadas paralelas
└── slice_plot.png                      # Efecto de parametros individuales
```

### Importancia de Parametros

```
param_importances.png muestra que parametros afectan mas el rendimiento:

Importancia de Hiperparametros
──────────────────────────────────────────────
strategy         ████████████████████  0.25
n_particles      ███████████████       0.19
max_iters        ████████████          0.15
neurons_mult     ██████████            0.13
optimizer        ████████              0.10
g                ██████                0.08
...

Interpretacion:
- strategy es el parametro mas importante
- n_particles tiene impacto significativo
- g (QDPSO) tiene menor impacto que strategy
```

### Coordenadas Paralelas

```
Visualiza configuraciones como lineas a traves de parametros:

          optimizer  strategy  n_particles  max_iters  score
              │          │          │           │         │
    QPSO ─────┼──forward─┼────30────┼────100────┼───0.72──┤
              │          │          │           │         │
    QDPSO ────┼─weighted─┼────50────┼────200────┼───0.78──┤
              │          │          │           │         │
    QDPSO ────┼layerwise─┼────45────┼────180────┼───0.85──┤ ← Mejor
              │          │          │           │         │

Las lineas de configuraciones buenas muestran patrones:
- Mayoría usa QDPSO
- Layerwise domina en las mejores
- n_particles entre 40-60
```

---

## Configuracion Avanzada

### Modificar Espacio de Busqueda

```python
# Agregar nuevo parametro
SEARCH_SPACE['activation'] = ['tanh', 'relu', 'leaky_relu']

# En ObjectiveFunction.__call__:
activation = trial.suggest_categorical('activation', SEARCH_SPACE['activation'])
```

### Busqueda en Paralelo

```python
config = SearchConfig(
    n_jobs=4,  # 4 trials en paralelo
    n_trials=200
)
```

### Continuar Estudio Previo

```python
# Cargar estudio existente
study = optuna.load_study(
    study_name='qpso_mcw_optimization',
    storage='sqlite:///optuna_study.db'
)

# Continuar optimizacion
study.optimize(objective, n_trials=50)
```

### Dashboard Interactivo

```bash
# Iniciar dashboard web
optuna-dashboard sqlite:///optuna_study.db

# Abrir en navegador: http://localhost:8080
```

---

## Recomendaciones

### Numero de Trials

| Complejidad | n_trials | Tiempo estimado |
|-------------|----------|-----------------|
| Rapido | 30-50 | 30-60 min |
| Balanceado | 100-150 | 2-4 horas |
| Exhaustivo | 200-500 | 8-24 horas |

### Cuando el Estudio Converge

```
Señales de convergencia:
1. Mejora marginal en ultimos 20 trials
2. Top 10 configuraciones muy similares
3. Importancia de parametros estabilizada

Si no converge:
1. Aumentar n_trials
2. Expandir espacio de busqueda
3. Verificar que datos sean correctos
```

### Mejores Practicas

1. **Empezar con pocos trials** (30-50) para verificar que funciona
2. **Usar pruning** para ahorrar tiempo
3. **Guardar resultados** por si se interrumpe
4. **Verificar varianza** entre folds (si es alta, aumentar folds)
5. **Evaluar en test** solo con la mejor configuracion final

---

## Referencias

1. **Optuna**: Akiba et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework"

2. **TPE**: Bergstra et al. (2011). "Algorithms for Hyper-Parameter Optimization"

3. **Pruning**: Li et al. (2018). "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"

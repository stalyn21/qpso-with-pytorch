# Ejemplos y Casos de Uso

[🏠 README](../../README.md) | [📚 Index](index.md) | [⬅️ Utils](utils.md) | **Examples** | [Scripts ➡️](main_qpso.md)

---

## Descripcion

Este documento contiene ejemplos completos y casos de uso del paquete QPSO Neural Network Training. Los ejemplos van desde uso basico hasta casos avanzados con configuraciones personalizadas.

---

## Tabla de Ejemplos

| # | Ejemplo | Descripcion | Dificultad |
|---|---------|-------------|------------|
| 1 | [Inicio Rapido](#1-inicio-rapido) | Entrenamiento minimo | Basico |
| 2 | [Clasificacion Iris](#2-clasificacion-iris) | Dataset clasico | Basico |
| 3 | [Cross-Validation](#3-cross-validation) | Evaluacion robusta | Intermedio |
| 4 | [QPSO vs QDPSO](#4-qpso-vs-qdpso) | Comparativa de algoritmos | Intermedio |
| 5 | [Arquitectura Escalada](#5-arquitectura-escalada) | Busqueda de arquitectura | Intermedio |
| 6 | [Configuracion Avanzada](#6-configuracion-avanzada) | Parametros optimizados | Avanzado |
| 7 | [Callbacks Personalizados](#7-callbacks-personalizados) | Monitoreo del entrenamiento | Avanzado |
| 8 | [Datos Personalizados](#8-datos-personalizados) | Datasets propios | Intermedio |
| 9 | [Guardado y Carga](#9-guardado-y-carga) | Persistencia de modelos | Basico |
| 10 | [Pipeline Completo](#10-pipeline-completo) | Proyecto end-to-end | Avanzado |

---

## 1. Inicio Rapido

El ejemplo mas simple para empezar a usar el paquete.

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset

# Cargar datos
X_train, X_test, y_train, y_test = load_dataset('iris')

# Configurar (valores por defecto)
config = TrainingConfig()

# Crear trainer
trainer = Trainer(
    input_dim=4,
    output_dim=3,
    config=config
)

# Entrenar
result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

# Ver resultado
print(f"Test Accuracy: {result.test_accuracy:.4f}")
```

**Output esperado:**
```
============================================================
ENTRENAMIENTO DE RED NEURONAL CON QPSO
============================================================
Modelo: QPSOCompatibleANN(
  architecture: 4 -> 64 -> 32 -> 16 -> 3
  activation: relu
  params: 3,539
  device: cuda
)
...
Test Accuracy: 0.9333
```

---

## 2. Clasificacion Iris

Ejemplo completo con el dataset Iris incluyendo evaluacion detallada.

```python
import numpy as np
from ann.models import QPSOCompatibleANN
from ann.optimizers import QPSONNOptimizer
from ann.optimizers.qpso_nn import NNOptimizationConfig
from ann.utils import load_dataset, MulticlassMetrics
import torch

# =============================================================================
# 1. Cargar y preparar datos
# =============================================================================

X_train, X_test, y_train, y_test = load_dataset('iris')

print(f"Dataset Iris")
print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Test:  {X_test.shape[0]} samples")
print(f"  Classes: {np.unique(y_train)}")

# Convertir a tensores
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# =============================================================================
# 2. Crear modelo
# =============================================================================

model = QPSOCompatibleANN(
    input_dim=4,
    output_dim=3,
    hidden_layers=[16, 8],
    activation='relu'
)

print(f"\nModelo creado:")
print(model)

# =============================================================================
# 3. Configurar optimizador
# =============================================================================

config = NNOptimizationConfig(
    n_particles=30,
    max_iters=100,
    alpha=(1.0, 0.5),
    weight_bound=1.0,
    patience=30,
    seed=42
)

optimizer = QPSONNOptimizer(model, config=config)

# =============================================================================
# 4. Entrenar
# =============================================================================

print("\nIniciando entrenamiento...")
result = optimizer.fit(
    X_train_t, y_train_t,
    X_val=X_test_t, y_val=y_test_t,
    verbose=True
)

# =============================================================================
# 5. Evaluar
# =============================================================================

test_metrics = optimizer.evaluate(X_test_t, y_test_t)
predictions = optimizer.predict(X_test_t)

print(f"\nResultados en Test:")
print(f"  Loss: {test_metrics['loss']:.6f}")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

# Metricas detalladas
metrics = MulticlassMetrics(
    class_names=['setosa', 'versicolor', 'virginica']
)
metrics.print_summary(y_test, predictions.cpu().numpy())
```

---

## 3. Cross-Validation

Evaluacion robusta usando k-fold cross-validation.

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset
import numpy as np

# Cargar dataset completo
X_train, X_test, y_train, y_test = load_dataset('wine')

# Combinar para CV
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

# Separar test final
from sklearn.model_selection import train_test_split
X_cv, X_final, y_cv, y_final = train_test_split(
    X_all, y_all,
    test_size=0.15,
    random_state=42,
    stratify=y_all
)

print(f"Datos para CV: {X_cv.shape[0]} samples")
print(f"Test final: {X_final.shape[0]} samples")

# Configurar
config = TrainingConfig(
    hidden_layers=[32, 16],
    n_particles=40,
    max_iters=80,
    n_folds=5,           # 5-fold CV
    random_state=42,
    verbose=True,
    save_best_model=False
)

# Crear trainer
trainer = Trainer(
    input_dim=X_cv.shape[1],
    output_dim=len(np.unique(y_cv)),
    config=config
)

# Entrenar con CV
result = trainer.fit_cv(X_cv, y_cv, X_test=X_final, y_test=y_final)

# Resultados
print("\n" + "="*60)
print("RESUMEN CROSS-VALIDATION")
print("="*60)
print(f"Train Accuracy: {result.train_accuracy:.4f} +/- {np.std([f['train_acc'] for f in result.fold_results]):.4f}")
print(f"Val Accuracy:   {result.val_accuracy:.4f} +/- {np.std([f['val_acc'] for f in result.fold_results]):.4f}")
print(f"Test Accuracy:  {result.test_accuracy:.4f}")

print("\nResultados por fold:")
for fold in result.fold_results:
    print(f"  Fold {fold['fold']}: train={fold['train_acc']:.4f}, val={fold['val_acc']:.4f}")
```

---

## 4. QPSO vs QDPSO

Comparativa entre los dos algoritmos de optimizacion.

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset
import time

# Dataset mas desafiante
X_train, X_test, y_train, y_test = load_dataset('digits')

print(f"Dataset Digits: {X_train.shape[1]} features, {len(set(y_train))} classes")

# Configuracion comun
common = {
    'hidden_layers': [64, 32],
    'n_particles': 40,
    'max_iters': 50,
    'random_state': 42,
    'verbose': False,
    'save_best_model': False
}

algorithms = {
    'QPSO': TrainingConfig(use_qdpso=False, alpha=(1.0, 0.5), **common),
    'QDPSO': TrainingConfig(use_qdpso=True, g=0.96, **common)
}

results = {}

for name, config in algorithms.items():
    print(f"\nEntrenando con {name}...")

    trainer = Trainer(
        input_dim=X_train.shape[1],
        output_dim=10,
        config=config
    )

    start = time.time()
    result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)
    elapsed = time.time() - start

    results[name] = {
        'train_acc': result.train_accuracy,
        'test_acc': result.test_accuracy,
        'time': elapsed,
        'iterations': result.n_iterations
    }

# Comparativa
print("\n" + "="*60)
print("COMPARATIVA QPSO vs QDPSO")
print("="*60)
print(f"{'Algoritmo':<10} {'Train Acc':<12} {'Test Acc':<12} {'Tiempo':<10} {'Iters':<8}")
print("-"*60)
for name, res in results.items():
    print(f"{name:<10} {res['train_acc']:<12.4f} {res['test_acc']:<12.4f} {res['time']:<10.2f}s {res['iterations']:<8}")

# Determinar ganador
winner = max(results, key=lambda x: results[x]['test_acc'])
print(f"\nMejor algoritmo: {winner} con {results[winner]['test_acc']:.4f} accuracy")
```

---

## 5. Arquitectura Escalada

Busqueda de la mejor arquitectura usando factores de escala.

```python
from ann.trainers import Trainer, TrainingConfig
from ann.models import create_scaled_architecture
from ann.utils import load_dataset
import numpy as np

# Dataset
X_train, X_test, y_train, y_test = load_dataset('breast_cancer')
input_dim = X_train.shape[1]
output_dim = 2

print(f"Input dimension: {input_dim}")
print(f"Probando diferentes factores de escala...\n")

# Factores a probar
scale_factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

results = []

for scale in scale_factors:
    # Generar arquitectura
    hidden_layers = create_scaled_architecture(input_dim, scale)

    # Calcular parametros
    model_params = (
        (input_dim + 1) * hidden_layers[0] +
        (hidden_layers[0] + 1) * hidden_layers[1] +
        (hidden_layers[1] + 1) * hidden_layers[2] +
        (hidden_layers[2] + 1) * output_dim
    )

    print(f"Scale {scale:.2f}: {hidden_layers} ({model_params:,} params)")

    config = TrainingConfig(
        hidden_layers=hidden_layers,
        n_particles=30,
        max_iters=50,
        random_state=42,
        verbose=False,
        save_best_model=False
    )

    trainer = Trainer(input_dim, output_dim, config)
    result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    results.append({
        'scale': scale,
        'layers': hidden_layers,
        'params': model_params,
        'train_acc': result.train_accuracy,
        'test_acc': result.test_accuracy,
        'time': result.training_time
    })

    print(f"  -> Test Accuracy: {result.test_accuracy:.4f}")

# Analisis
print("\n" + "="*70)
print("RESUMEN DE ARQUITECTURAS")
print("="*70)
print(f"{'Scale':<8} {'Layers':<20} {'Params':<10} {'Test Acc':<12} {'Time':<8}")
print("-"*70)

for r in results:
    layers_str = str(r['layers'])
    print(f"{r['scale']:<8.2f} {layers_str:<20} {r['params']:<10,} {r['test_acc']:<12.4f} {r['time']:<8.2f}s")

# Mejor resultado
best = max(results, key=lambda x: x['test_acc'])
print(f"\nMejor arquitectura: scale={best['scale']}, layers={best['layers']}")
print(f"  Test Accuracy: {best['test_acc']:.4f}")
print(f"  Parametros: {best['params']:,}")
```

---

## 6. Configuracion Avanzada

Optimizacion de hiperparametros para mejor rendimiento.

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset
import itertools

X_train, X_test, y_train, y_test = load_dataset('wine')

# Grid de hiperparametros
param_grid = {
    'n_particles': [30, 50, 100],
    'max_iters': [100, 200],
    'alpha': [(1.0, 0.5), (1.0, 0.3), (0.8, 0.4)],
    'boundary_strategy': ['clamp', 'reflect']
}

# Generar combinaciones
keys = param_grid.keys()
combinations = list(itertools.product(*param_grid.values()))

print(f"Probando {len(combinations)} configuraciones...\n")

best_result = None
best_config = None
results = []

for i, values in enumerate(combinations):
    params = dict(zip(keys, values))

    config = TrainingConfig(
        hidden_layers=[32, 16],
        random_state=42,
        verbose=False,
        save_best_model=False,
        **params
    )

    trainer = Trainer(13, 3, config)
    result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    results.append({
        'params': params,
        'test_acc': result.test_accuracy,
        'time': result.training_time
    })

    if best_result is None or result.test_accuracy > best_result:
        best_result = result.test_accuracy
        best_config = params

    print(f"[{i+1}/{len(combinations)}] {params['n_particles']}p, {params['max_iters']}i, "
          f"alpha={params['alpha']}, {params['boundary_strategy']} -> {result.test_accuracy:.4f}")

print("\n" + "="*60)
print("MEJOR CONFIGURACION ENCONTRADA")
print("="*60)
for key, value in best_config.items():
    print(f"  {key}: {value}")
print(f"  Test Accuracy: {best_result:.4f}")
```

---

## 7. Callbacks Personalizados

Monitoreo del entrenamiento con callbacks.

```python
import torch
from ann.models import QPSOCompatibleANN
from ann.optimizers import QPSONNOptimizer
from ann.optimizers.qpso_nn import NNOptimizationConfig
from ann.utils import load_dataset

X_train, X_test, y_train, y_test = load_dataset('iris')

# Convertir a tensores
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

# Modelo
model = QPSOCompatibleANN(4, 3, [16, 8])

# Configuracion
config = NNOptimizationConfig(
    n_particles=30,
    max_iters=100,
    seed=42
)

# Clase para callback con estado
class TrainingMonitor:
    def __init__(self):
        self.best_values = []
        self.improvements = 0
        self.last_best = float('inf')

    def __call__(self, opt):
        current = opt.gbest_value
        self.best_values.append(current)

        if current < self.last_best:
            self.improvements += 1
            self.last_best = current
            if opt.iters % 10 == 0:
                print(f"  Iter {opt.iters}: Nuevo mejor = {current:.6f}")

        # Early stopping personalizado
        if opt.iters > 50 and len(self.best_values) > 20:
            recent_improvement = self.best_values[-20] - self.best_values[-1]
            if recent_improvement < 1e-6:
                print(f"  Early stop en iter {opt.iters} - sin mejora significativa")

# Crear monitor
monitor = TrainingMonitor()

# Optimizador
optimizer = QPSONNOptimizer(model, config=config)

print("Entrenando con callback de monitoreo...")
result = optimizer.fit(X_train_t, y_train_t, callback=monitor, verbose=False)

print(f"\nEstadisticas del monitor:")
print(f"  Total mejoras: {monitor.improvements}")
print(f"  Valor inicial: {monitor.best_values[0]:.6f}")
print(f"  Valor final: {monitor.best_values[-1]:.6f}")
print(f"  Mejora total: {monitor.best_values[0] - monitor.best_values[-1]:.6f}")
```

---

## 8. Datos Personalizados

Usar datasets propios con el paquete.

```python
import numpy as np
import torch
from ann.trainers import Trainer, TrainingConfig
from ann.utils.data import normalize_data, train_test_split

# =============================================================================
# Opcion 1: Desde arrays numpy
# =============================================================================

# Simular datos
np.random.seed(42)
n_samples = 500
n_features = 20
n_classes = 4

X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, n_classes, n_samples)

# Normalizar
X_normalized = normalize_data(X, method='standard')

# Dividir
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y,
    test_size=0.2,
    stratify=True
)

# Entrenar
config = TrainingConfig(
    hidden_layers=[32, 16],
    n_particles=30,
    max_iters=50,
    verbose=True
)

trainer = Trainer(n_features, n_classes, config)
result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

# =============================================================================
# Opcion 2: Desde CSV
# =============================================================================

"""
import pandas as pd

# Cargar CSV
df = pd.read_csv('my_dataset.csv')

# Separar features y target
X = df.drop('target_column', axis=1).values
y = df['target_column'].values

# Asegurar que y es categorico
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Continuar con normalizacion y division...
"""

# =============================================================================
# Opcion 3: Desde tensores PyTorch
# =============================================================================

X_tensor = torch.randn(500, 20)
y_tensor = torch.randint(0, 4, (500,))

# El Trainer acepta tensores directamente
trainer = Trainer(20, 4, config)
result = trainer.fit(X_tensor, y_tensor)
```

---

## 9. Guardado y Carga

Persistir modelos entrenados para uso posterior.

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset
import os

# Preparar directorio
os.makedirs('./saved_models', exist_ok=True)

# =============================================================================
# Entrenar y guardar
# =============================================================================

X_train, X_test, y_train, y_test = load_dataset('iris')

config = TrainingConfig(
    hidden_layers=[16, 8],
    n_particles=30,
    max_iters=50,
    save_best_model=False  # Guardaremos manualmente
)

trainer = Trainer(4, 3, config)
result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

print(f"Modelo entrenado - Test Accuracy: {result.test_accuracy:.4f}")

# Guardar
saved_path = trainer.save_model(
    path='./saved_models',
    filename='iris_qpso_model.pth'
)
print(f"Modelo guardado en: {saved_path}")

# =============================================================================
# Cargar y usar
# =============================================================================

# Crear nuevo trainer
new_trainer = Trainer(4, 3, config)

# Cargar modelo
new_trainer.load_model(saved_path)

# Hacer predicciones
predictions = new_trainer.predict(X_test)
probabilities = new_trainer.predict_proba(X_test)

# Verificar
accuracy = (predictions == y_test).mean()
print(f"\nModelo cargado - Accuracy: {accuracy:.4f}")
print(f"Predicciones shape: {predictions.shape}")
print(f"Probabilidades shape: {probabilities.shape}")

# =============================================================================
# Examinar contenido del archivo
# =============================================================================

import torch
checkpoint = torch.load(saved_path)

print("\nContenido del checkpoint:")
for key in checkpoint.keys():
    if key == 'model_params':
        print(f"  {key}: tensor de {checkpoint[key].shape}")
    else:
        print(f"  {key}: {type(checkpoint[key])}")
```

---

## 10. Pipeline Completo

Proyecto end-to-end con todas las mejores practicas.

```python
"""
Pipeline completo de clasificacion con QPSO Neural Network.

Este ejemplo demuestra un flujo de trabajo profesional incluyendo:
1. Carga y exploracion de datos
2. Preprocesamiento
3. Seleccion de arquitectura
4. Entrenamiento con cross-validation
5. Evaluacion detallada
6. Guardado del modelo
7. Generacion de reportes
"""

import numpy as np
import time
import os
from datetime import datetime

from ann.trainers import Trainer, TrainingConfig
from ann.models import create_scaled_architecture
from ann.utils import load_dataset, MulticlassMetrics, train_test_split
from ann.utils.data import normalize_data

# =============================================================================
# 1. CONFIGURACION
# =============================================================================

DATASET = 'wine'
OUTPUT_DIR = './output/pipeline_example'
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("PIPELINE DE CLASIFICACION CON QPSO NEURAL NETWORK")
print("="*70)

# =============================================================================
# 2. CARGA Y EXPLORACION DE DATOS
# =============================================================================

print("\n[1/7] Cargando datos...")

X_train, X_test, y_train, y_test = load_dataset(DATASET)

print(f"  Dataset: {DATASET}")
print(f"  Samples: train={len(y_train)}, test={len(y_test)}")
print(f"  Features: {X_train.shape[1]}")
print(f"  Classes: {len(np.unique(y_train))}")
print(f"  Class distribution (train): {np.bincount(y_train)}")

# =============================================================================
# 3. SELECCION DE ARQUITECTURA
# =============================================================================

print("\n[2/7] Seleccionando arquitectura optima...")

input_dim = X_train.shape[1]
output_dim = len(np.unique(y_train))

# Probar escalas rapidamente
scale_results = []
for scale in [0.5, 1.0, 1.5]:
    hidden = create_scaled_architecture(input_dim, scale)

    config = TrainingConfig(
        hidden_layers=hidden,
        n_particles=20,
        max_iters=30,
        verbose=False,
        save_best_model=False
    )

    trainer = Trainer(input_dim, output_dim, config)
    result = trainer.fit(X_train, y_train)
    scale_results.append((scale, hidden, result.val_accuracy))
    print(f"  Scale {scale}: {hidden} -> val_acc={result.val_accuracy:.4f}")

# Seleccionar mejor
best_scale, best_hidden, _ = max(scale_results, key=lambda x: x[2])
print(f"  -> Seleccionada: scale={best_scale}, layers={best_hidden}")

# =============================================================================
# 4. ENTRENAMIENTO CON CROSS-VALIDATION
# =============================================================================

print("\n[3/7] Entrenando con cross-validation...")

final_config = TrainingConfig(
    hidden_layers=best_hidden,
    n_particles=50,
    max_iters=100,
    n_folds=5,
    alpha=(1.0, 0.5),
    patience=30,
    random_state=RANDOM_STATE,
    verbose=False,
    save_best_model=False
)

trainer = Trainer(input_dim, output_dim, final_config)

start_time = time.time()
result = trainer.fit_cv(
    np.vstack([X_train, X_test * 0.5]),  # Usar mas datos para CV
    np.concatenate([y_train, y_test[:len(y_test)//2]]),
    X_test=X_test[len(y_test)//2:],
    y_test=y_test[len(y_test)//2:]
)
training_time = time.time() - start_time

print(f"  Tiempo de entrenamiento: {training_time:.2f}s")
print(f"  Val Accuracy (CV): {result.val_accuracy:.4f} +/- {np.std([f['val_acc'] for f in result.fold_results]):.4f}")

# =============================================================================
# 5. ENTRENAMIENTO FINAL
# =============================================================================

print("\n[4/7] Entrenamiento final con todos los datos...")

final_trainer = Trainer(input_dim, output_dim, TrainingConfig(
    hidden_layers=best_hidden,
    n_particles=50,
    max_iters=150,
    patience=50,
    random_state=RANDOM_STATE,
    verbose=False,
    save_best_model=False
))

final_result = final_trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)
print(f"  Train Accuracy: {final_result.train_accuracy:.4f}")
print(f"  Test Accuracy: {final_result.test_accuracy:.4f}")

# =============================================================================
# 6. EVALUACION DETALLADA
# =============================================================================

print("\n[5/7] Evaluacion detallada...")

predictions = final_trainer.predict(X_test)

metrics_calc = MulticlassMetrics()
detailed_metrics = metrics_calc.calculate_all_metrics(y_test, predictions)

print(f"  Accuracy: {detailed_metrics['accuracy']:.4f}")
print(f"  F1 Macro: {detailed_metrics['f1_score']['macro']:.4f}")
print(f"  Cohen's Kappa: {detailed_metrics['cohen_kappa']:.4f}")
print(f"  MCC: {detailed_metrics['matthews_corrcoef']:.4f}")

# =============================================================================
# 7. GUARDADO DEL MODELO
# =============================================================================

print("\n[6/7] Guardando modelo...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = final_trainer.save_model(
    path=OUTPUT_DIR,
    filename=f'{DATASET}_model_{timestamp}.pth'
)
print(f"  Modelo guardado: {model_path}")

# =============================================================================
# 8. REPORTE FINAL
# =============================================================================

print("\n[7/7] Generando reporte...")

report = f"""
================================================================================
REPORTE DE ENTRENAMIENTO
================================================================================
Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset: {DATASET}

CONFIGURACION
-------------
- Arquitectura: {input_dim} -> {best_hidden} -> {output_dim}
- Particulas: {final_config.n_particles}
- Iteraciones: {final_config.max_iters}
- Alpha: {final_config.alpha}

RESULTADOS
----------
- Train Accuracy: {final_result.train_accuracy:.4f}
- Test Accuracy: {final_result.test_accuracy:.4f}
- F1 Score (Macro): {detailed_metrics['f1_score']['macro']:.4f}
- Cohen's Kappa: {detailed_metrics['cohen_kappa']:.4f}
- MCC: {detailed_metrics['matthews_corrcoef']:.4f}

TIEMPO
------
- Entrenamiento: {training_time:.2f}s
- Total: {time.time() - start_time:.2f}s

ARCHIVOS GENERADOS
------------------
- Modelo: {model_path}

================================================================================
"""

report_path = os.path.join(OUTPUT_DIR, f'report_{timestamp}.txt')
with open(report_path, 'w') as f:
    f.write(report)

print(report)
print(f"Reporte guardado: {report_path}")
print("\nPipeline completado exitosamente!")
```

---

## Ejecucion de Ejemplos

Para ejecutar los ejemplos:

```bash
# Activar entorno
conda activate pytorch_qpso

# Ejecutar todos los ejemplos de uso
cd /path/to/qdpso
python ann/usage_cases.py

# Ejecutar ejemplo especifico
python -c "
from ann.usage_cases import example_basic_usage
example_basic_usage()
"
```

---

## Consejos y Mejores Practicas

### Seleccion de Hiperparametros

| Parametro | Pequeno | Mediano | Grande |
|-----------|---------|---------|--------|
| `n_particles` | 20-30 | 40-60 | 80-150 |
| `max_iters` | 50-100 | 100-200 | 200-500 |
| `hidden_layers` | [16, 8] | [32, 16] | [64, 32, 16] |

### Cuando usar QPSO vs QDPSO

- **QPSO**: Mejor exploracion, datasets pequenos-medianos
- **QDPSO**: Mas estable, datasets grandes, menos hiperparametros

### Reproducibilidad

Siempre fijar semilla:
```python
config = TrainingConfig(random_state=42)
```

### Debugging

Si el modelo no converge:
1. Aumentar `n_particles`
2. Aumentar `max_iters`
3. Ajustar `weight_bound` (probar 0.5, 1.0, 2.0)
4. Cambiar `boundary_strategy` a "reflect"

---

## Related Documents

- [📚 Index](index.md) - Module overview
- [🔧 Utils](utils.md) - Data and metrics utilities
- [📜 main_qpso.md](main_qpso.md) - QPSO benchmark script
- [📜 main_qdpso.md](main_qdpso.md) - QDPSO benchmark script
- [📜 usage_cases.md](usage_cases.md) - Additional usage examples

---

<div align="center">

**[⬆️ Back to Top](#ejemplos-y-casos-de-uso)** | **[⬅️ Utils](utils.md)** | **[📚 Index](index.md)** | **[Scripts ➡️](main_qpso.md)**

</div>

# Modulo Trainers - Documentacion

[🏠 README](../../README.md) | [📚 Index](index.md) | [⬅️ Optimizers](optimizers.md) | **Trainers** | [Next: Utils ➡️](utils.md)

---

## Descripcion General

El modulo `trainers` proporciona una interfaz de alto nivel para entrenar, validar y evaluar redes neuronales usando QPSO. Abstrae la complejidad de crear modelos, configurar optimizadores y manejar datos, ofreciendo una API simple similar a scikit-learn.

**Ubicacion:** `ann/trainers/`

**Archivo principal:** `trainer.py`

---

## Componentes

| Clase | Descripcion |
|-------|-------------|
| `Trainer` | Clase principal de entrenamiento |
| `TrainingConfig` | Configuracion centralizada |
| `TrainingResult` | Resultado del entrenamiento |

---

## TrainingConfig

### Descripcion

Dataclass que agrupa toda la configuracion de entrenamiento, incluyendo arquitectura del modelo, parametros QPSO y opciones de evaluacion.

### Importacion

```python
from ann.trainers import TrainingConfig
```

### Definicion Completa

```python
@dataclass
class TrainingConfig:
    # Configuracion del modelo
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    activation: str = "relu"
    dropout: float = 0.0
    weight_bound: float = 1.0

    # Configuracion QPSO
    n_particles: int = 50
    max_iters: int = 100
    alpha: Union[float, Tuple[float, float]] = (1.0, 0.5)
    g: float = 0.96
    use_qdpso: bool = False
    boundary_strategy: str = "clamp"
    tol: float = 1e-12
    patience: int = 50

    # Configuracion de entrenamiento
    n_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    device: str = "auto"

    # Output
    output_dir: str = "./output"
    save_best_model: bool = True
    verbose: bool = True
```

### Grupos de Parametros

#### Configuracion del Modelo

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `hidden_layers` | `List[int]` | `[64, 32, 16]` | Neuronas por capa oculta |
| `activation` | `str` | `"relu"` | Funcion de activacion |
| `dropout` | `float` | `0.0` | Probabilidad de dropout |
| `weight_bound` | `float` | `1.0` | Limite para pesos |

#### Configuracion QPSO

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `n_particles` | `int` | `50` | Numero de particulas |
| `max_iters` | `int` | `100` | Iteraciones maximas |
| `alpha` | `float/tuple` | `(1.0, 0.5)` | Factor alpha (QPSO) |
| `g` | `float` | `0.96` | Factor g (QDPSO) |
| `use_qdpso` | `bool` | `False` | Usar QDPSO en lugar de QPSO |
| `boundary_strategy` | `str` | `"clamp"` | Estrategia de limites |
| `tol` | `float` | `1e-12` | Tolerancia de convergencia |
| `patience` | `int` | `50` | Paciencia para early stopping |

#### Configuracion de Entrenamiento

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `n_folds` | `int` | `5` | Folds para cross-validation |
| `test_size` | `float` | `0.2` | Proporcion de validacion |
| `random_state` | `int` | `42` | Semilla aleatoria |
| `device` | `str` | `"auto"` | Dispositivo de computo |

#### Configuracion de Output

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `output_dir` | `str` | `"./output"` | Directorio de salida |
| `save_best_model` | `bool` | `True` | Guardar mejor modelo |
| `verbose` | `bool` | `True` | Imprimir progreso |

### Metodo to_dict()

Convierte la configuracion a diccionario serializable.

```python
config = TrainingConfig(hidden_layers=[32, 16])
config_dict = config.to_dict()
# {'hidden_layers': [32, 16], 'activation': 'relu', ...}
```

### Ejemplos de Configuracion

```python
# Configuracion minima
config = TrainingConfig()

# Configuracion para dataset pequeno
config_small = TrainingConfig(
    hidden_layers=[16, 8],
    n_particles=30,
    max_iters=50,
    patience=20
)

# Configuracion para alta precision
config_precise = TrainingConfig(
    hidden_layers=[128, 64, 32],
    n_particles=100,
    max_iters=500,
    tol=1e-15,
    patience=200
)

# Configuracion con QDPSO
config_qdpso = TrainingConfig(
    use_qdpso=True,
    g=0.96,
    n_particles=50
)
```

---

## TrainingResult

### Descripcion

Dataclass que contiene todos los resultados del entrenamiento.

### Atributos

| Atributo | Tipo | Descripcion |
|----------|------|-------------|
| `best_model_params` | `torch.Tensor` | Mejores pesos encontrados |
| `train_accuracy` | `float` | Accuracy en entrenamiento |
| `val_accuracy` | `float` | Accuracy en validacion |
| `test_accuracy` | `float` | Accuracy en test (si aplica) |
| `train_loss` | `float` | Loss en entrenamiento |
| `val_loss` | `float` | Loss en validacion |
| `test_loss` | `float` | Loss en test (si aplica) |
| `training_time` | `float` | Tiempo total de entrenamiento |
| `n_iterations` | `int` | Numero de iteraciones ejecutadas |
| `convergence_reason` | `str` | Razon de convergencia |
| `history` | `Dict[str, List]` | Historial de metricas |
| `config` | `TrainingConfig` | Configuracion usada |
| `fold_results` | `List[Dict]` | Resultados por fold (CV) |

### Ejemplo de Acceso

```python
result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

print(f"Train Accuracy: {result.train_accuracy:.4f}")
print(f"Val Accuracy: {result.val_accuracy:.4f}")
print(f"Test Accuracy: {result.test_accuracy:.4f}")
print(f"Training Time: {result.training_time:.2f}s")
print(f"Iterations: {result.n_iterations}")
print(f"Convergence: {result.convergence_reason}")

# Historial
for epoch, (loss, acc) in enumerate(zip(
    result.history['train_loss'],
    result.history['train_acc']
)):
    print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")
```

---

## Trainer

### Descripcion

Clase principal que orquesta el proceso completo de entrenamiento. Internamente crea el modelo (`QPSOCompatibleANN`) y el optimizador (`QPSONNOptimizer`), maneja la preparacion de datos y proporciona metodos para entrenamiento simple y con cross-validation.

### Importacion

```python
from ann.trainers import Trainer
```

### Constructor

```python
Trainer(
    input_dim: int,
    output_dim: int,
    config: Optional[TrainingConfig] = None
)
```

| Parametro | Tipo | Descripcion |
|-----------|------|-------------|
| `input_dim` | `int` | Dimension de entrada (features) |
| `output_dim` | `int` | Dimension de salida (clases) |
| `config` | `TrainingConfig` | Configuracion (opcional) |

### Ejemplo Basico

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset

# Cargar datos
X_train, X_test, y_train, y_test = load_dataset('iris')

# Configurar
config = TrainingConfig(
    hidden_layers=[16, 8],
    n_particles=30,
    max_iters=100
)

# Crear trainer
trainer = Trainer(
    input_dim=X_train.shape[1],  # 4
    output_dim=3,                 # 3 clases
    config=config
)

# Entrenar
result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)
```

---

## Metodos del Trainer

### fit()

Entrenamiento estandar con split train/val automatico.

```python
def fit(
    self,
    X_train: Union[np.ndarray, torch.Tensor],
    y_train: Union[np.ndarray, torch.Tensor],
    X_val: Optional[...] = None,
    y_val: Optional[...] = None,
    X_test: Optional[...] = None,
    y_test: Optional[...] = None
) -> TrainingResult
```

**Parametros:**

| Parametro | Tipo | Descripcion |
|-----------|------|-------------|
| `X_train` | `ndarray` o `Tensor` | Features de entrenamiento |
| `y_train` | `ndarray` o `Tensor` | Labels de entrenamiento |
| `X_val` | `ndarray` o `Tensor` | Features de validacion (opcional) |
| `y_val` | `ndarray` o `Tensor` | Labels de validacion (opcional) |
| `X_test` | `ndarray` o `Tensor` | Features de test (opcional) |
| `y_test` | `ndarray` o `Tensor` | Labels de test (opcional) |

**Comportamiento:**
- Si `X_val` no se proporciona, crea split automatico segun `config.test_size`
- Si `X_test` se proporciona, evalua el modelo final en test
- Convierte automaticamente numpy arrays a tensores

**Ejemplo:**

```python
# Con split automatico
result = trainer.fit(X_train, y_train)

# Con validacion explicita
result = trainer.fit(X_train, y_train, X_val, y_val)

# Con test set
result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

# Completo
result = trainer.fit(X_train, y_train, X_val, y_val, X_test, y_test)
```

---

### fit_cv()

Entrenamiento con cross-validation k-fold.

```python
def fit_cv(
    self,
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    X_test: Optional[...] = None,
    y_test: Optional[...] = None
) -> TrainingResult
```

**Parametros:**

| Parametro | Tipo | Descripcion |
|-----------|------|-------------|
| `X` | `ndarray` o `Tensor` | Features completas |
| `y` | `ndarray` o `Tensor` | Labels completas |
| `X_test` | `ndarray` o `Tensor` | Test set separado (opcional) |
| `y_test` | `ndarray` o `Tensor` | Labels de test (opcional) |

**Comportamiento:**
- Divide datos en `config.n_folds` folds
- Entrena un modelo por fold
- Retorna metricas promediadas
- Guarda resultados por fold en `result.fold_results`

**Ejemplo:**

```python
config = TrainingConfig(
    n_folds=5,
    hidden_layers=[32, 16],
    n_particles=30,
    max_iters=50
)

trainer = Trainer(input_dim=4, output_dim=3, config=config)

# Cross-validation
result = trainer.fit_cv(X, y)

# Ver resultados por fold
for fold in result.fold_results:
    print(f"Fold {fold['fold']}: val_acc={fold['val_acc']:.4f}")

# Metricas promedio
print(f"Mean Val Acc: {result.val_accuracy:.4f}")
```

---

### predict()

Realiza predicciones con el modelo entrenado.

```python
def predict(
    self,
    X: Union[np.ndarray, torch.Tensor]
) -> np.ndarray
```

**Retorna:** Array numpy con predicciones de clase

**Ejemplo:**

```python
# Entrenar primero
result = trainer.fit(X_train, y_train)

# Predecir
predictions = trainer.predict(X_test)
print(predictions)  # [0, 1, 2, 1, 0, ...]

# Calcular accuracy manualmente
accuracy = (predictions == y_test).mean()
```

---

### predict_proba()

Obtiene probabilidades de clase.

```python
def predict_proba(
    self,
    X: Union[np.ndarray, torch.Tensor]
) -> np.ndarray
```

**Retorna:** Array numpy con probabilidades `[n_samples, n_classes]`

**Ejemplo:**

```python
probabilities = trainer.predict_proba(X_test)
# probabilities.shape = (30, 3)

# Clase mas probable
predictions = probabilities.argmax(axis=1)

# Confianza de prediccion
confidence = probabilities.max(axis=1)
print(f"Mean confidence: {confidence.mean():.4f}")
```

---

### save_model()

Guarda el modelo entrenado.

```python
def save_model(
    self,
    path: Optional[str] = None,
    filename: Optional[str] = None
) -> str
```

**Parametros:**

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `path` | `str` | `config.output_dir` | Directorio de destino |
| `filename` | `str` | Auto-generado | Nombre del archivo |

**Retorna:** Ruta completa del archivo guardado

**Contenido del archivo:**

```python
{
    'model_params': tensor,           # Pesos del modelo
    'model_config': dict,             # Arquitectura
    'training_config': dict,          # Configuracion usada
    'results': {                      # Metricas
        'train_accuracy': float,
        'val_accuracy': float,
        'test_accuracy': float,
        'training_time': float,
        'n_iterations': int
    }
}
```

**Ejemplo:**

```python
# Guardar con nombre automatico
filepath = trainer.save_model()
print(f"Saved to: {filepath}")
# output/qpso_nn_acc0.9333_20260111_123456.pth

# Guardar con nombre personalizado
filepath = trainer.save_model(
    path='./models',
    filename='iris_model_v1.pth'
)
```

---

### load_model()

Carga un modelo guardado.

```python
def load_model(self, filepath: str) -> None
```

**Ejemplo:**

```python
# Crear trainer vacio
trainer = Trainer(input_dim=4, output_dim=3)

# Cargar modelo
trainer.load_model('models/iris_model_v1.pth')

# Usar para predicciones
predictions = trainer.predict(X_new)
```

---

## Propiedades

| Propiedad | Tipo | Descripcion |
|-----------|------|-------------|
| `model` | `QPSOCompatibleANN` | Modelo actual |
| `best_result` | `TrainingResult` | Mejor resultado |

---

## Flujo de Entrenamiento

```
┌──────────────────────────────────────────────────────────────┐
│                        trainer.fit()                         │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  1. Preparar datos                                           │
│     - Convertir a tensores                                   │
│     - Crear split train/val si no se proporciona             │
│     - Mover a dispositivo (GPU/CPU)                          │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  2. Crear modelo                                             │
│     - QPSOCompatibleANN con arquitectura de config           │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  3. Crear optimizador                                        │
│     - QPSONNOptimizer o QDPSONNOptimizer                     │
│     - Configurar con NNOptimizationConfig                    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  4. Entrenar                                                 │
│     - optimizer.fit(X_train, y_train, X_val, y_val)          │
│     - Ejecutar QPSO para optimizar pesos                     │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  5. Evaluar                                                  │
│     - Calcular metricas en train, val, test                  │
│     - Crear TrainingResult                                   │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  6. Guardar (opcional)                                       │
│     - Si config.save_best_model == True                      │
│     - Guardar pesos y configuracion                          │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     TrainingResult
```

---

## Ejemplos Completos

### Ejemplo 1: Entrenamiento Simple

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset

# Cargar datos
X_train, X_test, y_train, y_test = load_dataset('iris')

# Configurar
config = TrainingConfig(
    hidden_layers=[16, 8],
    n_particles=30,
    max_iters=100,
    verbose=True
)

# Crear y entrenar
trainer = Trainer(input_dim=4, output_dim=3, config=config)
result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

# Resultados
print(f"Test Accuracy: {result.test_accuracy:.4f}")
```

### Ejemplo 2: Cross-Validation

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset
import numpy as np

# Cargar datos completos
X_train, X_test, y_train, y_test = load_dataset('wine')
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

# Separar test final
from sklearn.model_selection import train_test_split
X_cv, X_final_test, y_cv, y_final_test = train_test_split(
    X_all, y_all, test_size=0.15, random_state=42
)

# Configurar CV
config = TrainingConfig(
    hidden_layers=[32, 16],
    n_particles=40,
    max_iters=80,
    n_folds=5,
    verbose=True
)

# Entrenar con CV
trainer = Trainer(
    input_dim=X_cv.shape[1],
    output_dim=len(np.unique(y_cv)),
    config=config
)

result = trainer.fit_cv(X_cv, y_cv, X_test=X_final_test, y_test=y_final_test)

# Resultados
print(f"\nCross-Validation Results:")
print(f"  Mean Train Acc: {result.train_accuracy:.4f}")
print(f"  Mean Val Acc: {result.val_accuracy:.4f}")
print(f"  Final Test Acc: {result.test_accuracy:.4f}")

# Por fold
print("\nPer-fold results:")
for fold in result.fold_results:
    print(f"  Fold {fold['fold']}: {fold['val_acc']:.4f}")
```

### Ejemplo 3: Comparar Arquitecturas

```python
from ann.trainers import Trainer, TrainingConfig
from ann.models import create_scaled_architecture
from ann.utils import load_dataset

X_train, X_test, y_train, y_test = load_dataset('breast_cancer')
input_dim = X_train.shape[1]

# Probar diferentes escalas
architectures = {
    'small': create_scaled_architecture(input_dim, 0.25),
    'medium': create_scaled_architecture(input_dim, 0.5),
    'large': create_scaled_architecture(input_dim, 1.0),
}

results = {}

for name, hidden_layers in architectures.items():
    print(f"\nTraining {name}: {hidden_layers}")

    config = TrainingConfig(
        hidden_layers=hidden_layers,
        n_particles=30,
        max_iters=50,
        verbose=False,
        save_best_model=False
    )

    trainer = Trainer(input_dim, 2, config)
    result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    results[name] = {
        'layers': hidden_layers,
        'params': trainer.model.num_params,
        'test_acc': result.test_accuracy,
        'time': result.training_time
    }

# Comparar
print("\n" + "="*60)
print("Comparison:")
print("="*60)
for name, res in results.items():
    print(f"{name:10} | params={res['params']:5} | acc={res['test_acc']:.4f} | time={res['time']:.2f}s")
```

### Ejemplo 4: QPSO vs QDPSO

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset

X_train, X_test, y_train, y_test = load_dataset('digits')

common_config = {
    'hidden_layers': [64, 32],
    'n_particles': 40,
    'max_iters': 50,
    'verbose': False,
    'save_best_model': False
}

# QPSO
config_qpso = TrainingConfig(
    use_qdpso=False,
    alpha=(1.0, 0.5),
    **common_config
)

# QDPSO
config_qdpso = TrainingConfig(
    use_qdpso=True,
    g=0.96,
    **common_config
)

for name, config in [('QPSO', config_qpso), ('QDPSO', config_qdpso)]:
    trainer = Trainer(64, 10, config)
    result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)
    print(f"{name}: acc={result.test_accuracy:.4f}, time={result.training_time:.2f}s")
```

### Ejemplo 5: Guardar y Cargar

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset

# Entrenar y guardar
X_train, X_test, y_train, y_test = load_dataset('iris')

config = TrainingConfig(
    hidden_layers=[16, 8],
    n_particles=30,
    max_iters=50
)

trainer = Trainer(4, 3, config)
result = trainer.fit(X_train, y_train)

# Guardar
filepath = trainer.save_model(
    path='./saved_models',
    filename='iris_classifier.pth'
)
print(f"Model saved to: {filepath}")

# Cargar en nueva sesion
new_trainer = Trainer(4, 3, config)
new_trainer.load_model(filepath)

# Verificar
predictions = new_trainer.predict(X_test)
accuracy = (predictions == y_test).mean()
print(f"Loaded model accuracy: {accuracy:.4f}")
```

---

## Mejoras vs Implementacion Original

| Aspecto | Original (`QPSOFineTuner`) | Nueva (`Trainer`) |
|---------|---------------------------|-------------------|
| Lineas de codigo | ~700 en un archivo | ~300, modular |
| Configuracion | Hardcodeada | `TrainingConfig` |
| Optuna | Integrado, obligatorio | No requerido |
| CV | Manual con KFold | `fit_cv()` integrado |
| Guardado | Codigo complejo | `save_model()` simple |
| Carga | No estandarizado | `load_model()` |
| Metricas | Externas | Integradas en result |
| API | Multiples metodos | 2 metodos principales |

---

## Related Documents

- [📚 Index](index.md) - Module overview
- [⚙️ Optimizers](optimizers.md) - QPSO optimizers for neural networks
- [🔧 Utils](utils.md) - Data and metrics utilities
- [📖 Examples](examples.md) - Complete usage examples

---

<div align="center">

**[⬆️ Back to Top](#modulo-trainers---documentacion)** | **[⬅️ Optimizers](optimizers.md)** | **[📚 Index](index.md)** | **[Next: Utils ➡️](utils.md)**

</div>

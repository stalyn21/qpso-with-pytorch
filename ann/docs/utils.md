# Modulo Utils - Documentacion

[🏠 README](../../README.md) | [📚 Index](index.md) | [⬅️ Trainers](trainers.md) | **Utils** | [Next: Examples ➡️](examples.md)

---

## Descripcion General

El modulo `utils` proporciona funciones utilitarias para el manejo de datos y calculo de metricas de clasificacion. Incluye funciones para cargar datasets, preprocesar datos y evaluar modelos.

**Ubicacion:** `ann/utils/`

**Archivos:**
- `data.py` - Funciones de carga y preprocesamiento de datos
- `metrics.py` - Metricas de evaluacion para clasificacion

---

## Modulo data.py

### Funciones Disponibles

| Funcion | Descripcion |
|---------|-------------|
| `load_dataset` | Carga datasets predefinidos (iris, wine, etc.) |
| `train_test_split` | Divide datos en train/test |
| `normalize_data` | Normaliza datos con diferentes metodos |
| `create_dataloaders` | Crea DataLoaders de PyTorch |
| `generate_synthetic_data` | Genera datos sinteticos |

---

### load_dataset()

Carga datasets predefinidos de scikit-learn, normalizados y divididos.

```python
def load_dataset(
    name: str,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

**Parametros:**

| Parametro | Tipo | Descripcion |
|-----------|------|-------------|
| `name` | `str` | Nombre del dataset |
| `test_size` | `float` | Proporcion de test (default: 0.2) |
| `random_state` | `int` | Semilla aleatoria (default: 42) |

**Datasets disponibles:**

| Nombre | Descripcion | Samples | Features | Clases |
|--------|-------------|---------|----------|--------|
| `"iris"` | Clasificacion de flores | 150 | 4 | 3 |
| `"wine"` | Clasificacion de vinos | 178 | 13 | 3 |
| `"breast_cancer"` | Cancer de mama | 569 | 30 | 2 |
| `"digits"` | Digitos escritos | 1797 | 64 | 10 |

**Retorna:** `(X_train, X_test, y_train, y_test)`

**Ejemplo:**

```python
from ann.utils import load_dataset

# Cargar Iris
X_train, X_test, y_train, y_test = load_dataset('iris')
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
# Train: (120, 4), Test: (30, 4)

# Con test_size personalizado
X_train, X_test, y_train, y_test = load_dataset(
    'wine',
    test_size=0.3,
    random_state=123
)
```

**Procesamiento automatico:**
1. Carga dataset de sklearn
2. Aplica StandardScaler a features
3. Divide con estratificacion
4. Retorna arrays numpy

---

### train_test_split()

Divide datos en conjuntos de entrenamiento y test.

```python
def train_test_split(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

**Parametros:**

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `X` | `ndarray/Tensor` | - | Features |
| `y` | `ndarray/Tensor` | - | Labels |
| `test_size` | `float` | `0.2` | Proporcion de test |
| `random_state` | `int` | `42` | Semilla |
| `stratify` | `bool` | `True` | Mantener proporcion de clases |

**Ejemplo:**

```python
from ann.utils import train_test_split
import numpy as np

X = np.random.randn(100, 10)
y = np.random.randint(0, 3, 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=True
)
```

---

### normalize_data()

Normaliza datos usando diferentes metodos.

```python
def normalize_data(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    method: str = 'standard'
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
```

**Parametros:**

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `X_train` | `ndarray` | - | Datos de entrenamiento |
| `X_test` | `ndarray` | `None` | Datos de test (opcional) |
| `method` | `str` | `"standard"` | Metodo de normalizacion |

**Metodos disponibles:**

| Metodo | Descripcion | Cuando usar |
|--------|-------------|-------------|
| `"standard"` | Mean=0, Std=1 | Default, datos normales |
| `"minmax"` | Rango [0, 1] | Datos acotados |
| `"robust"` | Usa mediana/IQR | Datos con outliers |

**Ejemplo:**

```python
from ann.utils.data import normalize_data

# Solo train
X_train_norm = normalize_data(X_train, method='standard')

# Train y test
X_train_norm, X_test_norm = normalize_data(
    X_train, X_test,
    method='minmax'
)
```

---

### create_dataloaders()

Crea DataLoaders de PyTorch para entrenamiento.

```python
def create_dataloaders(
    X_train: Union[np.ndarray, torch.Tensor],
    y_train: Union[np.ndarray, torch.Tensor],
    X_val: Optional[...] = None,
    y_val: Optional[...] = None,
    batch_size: int = 32,
    shuffle_train: bool = True,
    device: str = 'cpu'
) -> Tuple[DataLoader, Optional[DataLoader]]
```

**Ejemplo:**

```python
from ann.utils.data import create_dataloaders

train_loader, val_loader = create_dataloaders(
    X_train, y_train,
    X_val, y_val,
    batch_size=64,
    device='cuda'
)

for X_batch, y_batch in train_loader:
    # Entrenar con batch
    pass
```

---

### generate_synthetic_data()

Genera datos sinteticos para pruebas.

```python
def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 3,
    n_informative: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]
```

**Ejemplo:**

```python
from ann.utils.data import generate_synthetic_data

# Generar dataset sintetico
X, y = generate_synthetic_data(
    n_samples=500,
    n_features=20,
    n_classes=4,
    n_informative=10
)

print(f"X: {X.shape}, y: {y.shape}")
# X: (500, 20), y: (500,)
```

---

## Modulo metrics.py

### Funciones Disponibles

| Funcion | Descripcion |
|---------|-------------|
| `calculate_accuracy` | Calcula accuracy simple |
| `calculate_confusion_matrix` | Genera matriz de confusion |
| `calculate_classification_report` | Reporte detallado por clase |
| `plot_confusion_matrix` | Grafica matriz de confusion |
| `plot_training_history` | Grafica historial de entrenamiento |

### Clases Disponibles

| Clase | Descripcion |
|-------|-------------|
| `MulticlassMetrics` | Calculador completo de metricas |

---

### calculate_accuracy()

Calcula la accuracy de clasificacion.

```python
def calculate_accuracy(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float
```

**Ejemplo:**

```python
from ann.utils import calculate_accuracy

accuracy = calculate_accuracy(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

---

### calculate_confusion_matrix()

Genera la matriz de confusion.

```python
def calculate_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> np.ndarray
```

**Ejemplo:**

```python
from ann.utils import calculate_confusion_matrix

cm = calculate_confusion_matrix(y_true, y_pred)
print(cm)
# [[15  0  0]
#  [ 0 14  1]
#  [ 0  2 13]]
```

---

### calculate_classification_report()

Genera reporte detallado de clasificacion.

```python
def calculate_classification_report(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    target_names: Optional[List[str]] = None,
    output_dict: bool = True
) -> Union[str, Dict]
```

**Ejemplo:**

```python
from ann.utils import calculate_classification_report

# Como diccionario
report = calculate_classification_report(
    y_true, y_pred,
    target_names=['setosa', 'versicolor', 'virginica']
)

print(report['setosa'])
# {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 10}

# Como string
report_str = calculate_classification_report(
    y_true, y_pred,
    output_dict=False
)
print(report_str)
```

---

## MulticlassMetrics

### Descripcion

Clase para calcular metricas completas de clasificacion multiclase.

### Importacion

```python
from ann.utils import MulticlassMetrics
```

### Constructor

```python
MulticlassMetrics(class_names: Optional[List[str]] = None)
```

### Metodo: calculate_all_metrics()

Calcula todas las metricas disponibles.

```python
def calculate_all_metrics(
    self,
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> Dict
```

**Retorna:**

```python
{
    'accuracy': 0.9333,
    'precision': {
        'macro': 0.9400,
        'weighted': 0.9350
    },
    'recall': {
        'macro': 0.9333,
        'weighted': 0.9333
    },
    'f1_score': {
        'macro': 0.9356,
        'weighted': 0.9340
    },
    'cohen_kappa': 0.9000,
    'matthews_corrcoef': 0.9010,
    'confusion_matrix': np.array([[...]]),
    'classification_report': {...}
}
```

**Ejemplo:**

```python
from ann.utils import MulticlassMetrics

metrics = MulticlassMetrics(
    class_names=['setosa', 'versicolor', 'virginica']
)

results = metrics.calculate_all_metrics(y_true, y_pred)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1 Macro: {results['f1_score']['macro']:.4f}")
print(f"Kappa: {results['cohen_kappa']:.4f}")
```

### Metodo: print_summary()

Imprime un resumen formateado de las metricas.

```python
def print_summary(
    self,
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> None
```

**Ejemplo:**

```python
metrics = MulticlassMetrics()
metrics.print_summary(y_true, y_pred)
```

**Output:**
```
==================================================
METRICAS DE CLASIFICACION
==================================================
Accuracy:           0.9333
Precision (macro):  0.9400
Recall (macro):     0.9333
F1-Score (macro):   0.9356
Cohen's Kappa:      0.9000
MCC:                0.9010
--------------------------------------------------
Matriz de Confusion:
[[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]
==================================================
```

---

## Funciones de Visualizacion

### plot_confusion_matrix()

Grafica la matriz de confusion usando seaborn.

```python
def plot_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
) -> None
```

**Ejemplo:**

```python
from ann.utils.metrics import plot_confusion_matrix

plot_confusion_matrix(
    y_true, y_pred,
    class_names=['setosa', 'versicolor', 'virginica'],
    save_path='confusion_matrix.png'
)
```

**Requiere:** `matplotlib`, `seaborn`

---

### plot_training_history()

Grafica el historial de entrenamiento.

```python
def plot_training_history(
    history: Dict[str, List],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4)
) -> None
```

**Ejemplo:**

```python
from ann.utils.metrics import plot_training_history

# Despues de entrenar
history = trainer._optimizer.get_history()

plot_training_history(
    history,
    save_path='training_history.png'
)
```

**Graficas generadas:**
1. Curva de Loss (train y val)
2. Curva de Accuracy (train y val)

---

## Ejemplos Completos

### Ejemplo 1: Pipeline Completo de Datos

```python
from ann.utils import load_dataset, train_test_split
from ann.utils.data import normalize_data, generate_synthetic_data

# Opcion 1: Dataset predefinido
X_train, X_test, y_train, y_test = load_dataset('iris')

# Opcion 2: Datos personalizados
import pandas as pd
df = pd.read_csv('my_data.csv')
X = df.drop('target', axis=1).values
y = df['target'].values

# Normalizar
X_norm = normalize_data(X, method='standard')

# Dividir
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y,
    test_size=0.2,
    stratify=True
)

# Opcion 3: Datos sinteticos para pruebas
X, y = generate_synthetic_data(
    n_samples=1000,
    n_features=50,
    n_classes=5
)
```

### Ejemplo 2: Evaluacion Completa

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset, MulticlassMetrics
from ann.utils.metrics import plot_confusion_matrix, plot_training_history

# Entrenar modelo
X_train, X_test, y_train, y_test = load_dataset('wine')
config = TrainingConfig(hidden_layers=[32, 16], max_iters=100)
trainer = Trainer(13, 3, config)
result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

# Predicciones
y_pred = trainer.predict(X_test)

# Metricas completas
metrics = MulticlassMetrics(
    class_names=['class_0', 'class_1', 'class_2']
)
metrics.print_summary(y_test, y_pred)

# Visualizaciones
plot_confusion_matrix(
    y_test, y_pred,
    class_names=['class_0', 'class_1', 'class_2'],
    save_path='wine_confusion.png'
)

plot_training_history(
    result.history,
    save_path='wine_training.png'
)
```

### Ejemplo 3: Comparativa de Normalizacion

```python
from ann.utils import load_dataset
from ann.utils.data import normalize_data
from ann.trainers import Trainer, TrainingConfig
import numpy as np

# Cargar datos sin normalizar
from sklearn.datasets import load_wine
data = load_wine()
X, y = data.data, data.target

from sklearn.model_selection import train_test_split as sk_split
X_train, X_test, y_train, y_test = sk_split(X, y, test_size=0.2, random_state=42)

# Probar diferentes normalizaciones
methods = ['standard', 'minmax', 'robust']
results = {}

for method in methods:
    # Normalizar
    X_train_norm, X_test_norm = normalize_data(X_train, X_test, method=method)

    # Entrenar
    config = TrainingConfig(
        hidden_layers=[16, 8],
        max_iters=50,
        verbose=False,
        save_best_model=False
    )
    trainer = Trainer(X_train.shape[1], 3, config)
    result = trainer.fit(
        X_train_norm, y_train,
        X_test=X_test_norm, y_test=y_test
    )

    results[method] = result.test_accuracy
    print(f"{method:10}: {result.test_accuracy:.4f}")

# Mejor metodo
best = max(results, key=results.get)
print(f"\nBest method: {best}")
```

---

## Metricas Explicadas

### Accuracy

```
Accuracy = (TP + TN) / Total
```

Proporcion de predicciones correctas. Puede ser enganosa con clases desbalanceadas.

### Precision

```
Precision = TP / (TP + FP)
```

De todas las predicciones positivas, cuantas son correctas.

### Recall (Sensitivity)

```
Recall = TP / (TP + FN)
```

De todos los positivos reales, cuantos se identificaron.

### F1-Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Media armonica de precision y recall.

### Cohen's Kappa

```
Kappa = (Accuracy - Pe) / (1 - Pe)
```

Mide acuerdo considerando el azar. Valores:
- < 0: Peor que azar
- 0-0.2: Bajo
- 0.2-0.4: Aceptable
- 0.4-0.6: Moderado
- 0.6-0.8: Sustancial
- 0.8-1.0: Casi perfecto

### Matthews Correlation Coefficient (MCC)

```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

Mejor metrica para clases desbalanceadas. Rango: -1 a 1.

---

## Related Documents

- [📚 Index](index.md) - Module overview
- [🏋️ Trainers](trainers.md) - High-level training interface
- [📖 Examples](examples.md) - Complete usage examples
- [🔧 QPSO Algorithms](../../QPSO-PyTorch/docs/index.md) - QPSO implementation details

---

<div align="center">

**[⬆️ Back to Top](#modulo-utils---documentacion)** | **[⬅️ Trainers](trainers.md)** | **[📚 Index](index.md)** | **[Next: Examples ➡️](examples.md)**

</div>

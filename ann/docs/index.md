# ANN Module - QPSO Neural Network Training

[🏠 Main README](../../README.md) | [🔧 QPSO Algorithms](../../QPSO-PyTorch/docs/index.md) | **ANN Module** | [🇪🇸 Español](index_es.md)

---

> **Version**: 2.0.0
> **Note**: This module is part of the [PyTorch QPSO Suite](../../README.md).
> For a complete overview of the repository, see the [main README](../../README.md).

---

## Overview

This package provides a modular and optimized implementation for training artificial neural networks using **QPSO (Quantum Particle Swarm Optimization)** as the optimization algorithm, instead of traditional backpropagation with gradient descent.

### Core Concept

In the traditional approach, neural networks are trained by calculating gradients and updating weights using algorithms like SGD or Adam. In contrast, QPSO treats network weights as **particle positions** in a multidimensional search space, where each particle represents a complete weight configuration.

```
Traditional Approach:          QPSO Approach:

  Forward Pass                  Particle 1: [w1, w2, ..., wn]
       |                        Particle 2: [w1, w2, ..., wn]
  Compute Loss                  ...
       |                        Particle N: [w1, w2, ..., wn]
  Backward Pass                        |
       |                        Evaluate fitness (loss)
  Update Weights                       |
                                Update positions (QPSO)
                                       |
                                Best particle = Best weights
```

---

## Table of Contents

### Module Documentation

| Document | Description |
|----------|-------------|
| [models.md](models.md) | QPSO-compatible neural network models |
| [optimizers.md](optimizers.md) | QPSO/QDPSO optimizers for neural networks |
| [trainers.md](trainers.md) | High-level Trainer with cross-validation |
| [utils.md](utils.md) | Data and metrics utilities |
| [examples.md](examples.md) | Complete examples and use cases |

### Executable Scripts Documentation

| Document | Description |
|----------|-------------|
| [main_qpso.md](main_qpso.md) | QPSO training benchmark |
| [main_qdpso.md](main_qdpso.md) | QDPSO training benchmark |
| [main_mcw.md](main_mcw.md) | MCW image classification benchmark (QPSO vs QDPSO) |
| [main_training_type.md](main_training_type.md) | Training strategies benchmark (Forward, Weighted, Layerwise) |
| [main_hyperparameter_search.md](main_hyperparameter_search.md) | Automated hyperparameter search with Optuna |
| [usage_cases.md](usage_cases.md) | 8 usage examples |

---

## Package Architecture

```
ann/
├── __init__.py                 # Main package exports
├── main_qpso.py                # QPSO benchmark (Iris, Wine, Breast Cancer)
├── main_qdpso.py               # QDPSO benchmark (Iris, Wine, Breast Cancer)
├── main_mcw.py                 # MCW benchmark with QPSO and QDPSO (weather images)
├── main_training_type.py       # Training strategies benchmark
├── main_hyperparameter_search.py  # Hyperparameter search with Optuna
├── start_hyperparameter_search.py # HPO configuration script
├── usage_cases.py              # 8 usage examples
│
├── tensor_qpso/                # QPSO optimization module (17 improvements)
│   ├── __init__.py             # Module exports
│   └── qpso_tensor_optimized.py  # Optimized implementation
│
├── data/                       # Data loading modules
│   ├── __init__.py             # Module exports
│   └── mcw.py                  # MCWDataset, load_mcw (Multi-Class Weather)
│
├── models/                     # Neural network models
│   ├── __init__.py
│   └── ann.py                  # QPSOCompatibleANN
│
├── optimizers/                 # QPSO-based optimizers
│   ├── __init__.py
│   ├── qpso_nn.py              # QPSONNOptimizer, QDPSONNOptimizer
│   └── training_strategies.py  # Strategies: Forward, Weighted, Layerwise
│
├── trainers/                   # Training logic
│   ├── __init__.py
│   └── trainer.py              # Trainer, TrainingConfig, TrainingResult
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── data.py                 # Loading and preprocessing functions
│   └── metrics.py              # Evaluation metrics and visualization
│
├── results/                    # Experiment results
│   └── hyperparameter_search/  # HPO results
│
└── docs/                       # Documentation
    ├── index.md                # This file
    ├── models.md               # Models documentation
    ├── optimizers.md           # Optimizers documentation
    ├── trainers.md             # Trainers documentation
    ├── utils.md                # Utilities documentation
    ├── examples.md             # General examples
    ├── main_qpso.md            # QPSO benchmark documentation
    ├── main_qdpso.md           # QDPSO benchmark documentation
    ├── main_mcw.md             # MCW benchmark documentation
    ├── main_training_type.md   # Training strategies documentation
    ├── main_hyperparameter_search.md  # HPO with Optuna documentation
    └── usage_cases.md          # Usage examples documentation
```

---

## Installation and Requirements

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- scikit-learn >= 0.24.0
- OpenCV >= 4.0.0 (for MCW dataset)
- mahotas >= 1.4.0 (for image features)
- matplotlib (optional, for visualization)

### Installation

```bash
# Activate conda environment
conda activate pytorch_qpso

# Verify installation
python -c "from ann import QPSOCompatibleANN, Trainer; print('OK')"
```

---

## Quick Start

### Minimal Example

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset

# 1. Load data
X_train, X_test, y_train, y_test = load_dataset('iris')

# 2. Configure
config = TrainingConfig(
    hidden_layers=[16, 8],
    n_particles=30,
    max_iters=100
)

# 3. Create trainer
trainer = Trainer(
    input_dim=X_train.shape[1],
    output_dim=3,
    config=config
)

# 4. Train
result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

# 5. Results
print(f"Test Accuracy: {result.test_accuracy:.4f}")
```

### Using the Optimizer Directly

```python
import torch
from ann.models import QPSOCompatibleANN
from ann.optimizers import QPSONNOptimizer

# Create model
model = QPSOCompatibleANN(
    input_dim=4,
    output_dim=3,
    hidden_layers=[16, 8]
)

# Create optimizer
optimizer = QPSONNOptimizer(model)

# Train
X = torch.randn(100, 4)
y = torch.randint(0, 3, (100,))
result = optimizer.fit(X, y)
```

---

## Supported Algorithms

### QPSO (Quantum Particle Swarm Optimization)

Original algorithm by Sun et al. (2004). Uses the **mean best (mbest)** concept.

**Update equation:**
```
mbest = (1/N) * sum(pbest_i)
c = phi * pbest + (1-phi) * gbest
L = alpha * |mbest - x|
x_new = c +/- L * ln(1/u)
```

**Parameters:**
- `alpha`: Contraction-expansion factor (typical: 0.75 or (1.0, 0.5) for decay)

### QDPSO (Quantum Delta PSO)

Variant that uses delta instead of mbest.

**Update equation:**
```
c = (u1*pbest + u2*gbest) / (u1+u2)
L = (1/g) * |x - c|
x_new = c +/- L * ln(1/u)
```

**Parameters:**
- `g`: Control factor (typical: 0.96)

---

## Training Strategies

The module supports three training strategies:

| Strategy | Description | Speed | Accuracy | Best For |
|----------|-------------|-------|----------|----------|
| **Forward** | Optimizes all weights simultaneously | Fast | Good | Small networks |
| **Weighted** | Layer-weighted fitness with decay | Medium | Good | Medium networks |
| **Layerwise** | Sequential layer-by-layer training | Slow | Best | Deep networks |

### Forward Strategy
Standard approach - all network weights are optimized simultaneously using QPSO.

### Weighted Strategy
Applies different weights to each layer's contribution to the fitness function, prioritizing output layers.

### Layerwise Strategy
Trains layers sequentially from output to input, with optional fine-tuning of all layers at the end.

---

## Recommended Configuration

### For Small Datasets (< 1000 samples)

```python
config = TrainingConfig(
    hidden_layers=[32, 16],
    n_particles=30,
    max_iters=100,
    alpha=(1.0, 0.5),
    patience=30
)
```

### For Medium Datasets (1000-10000 samples)

```python
config = TrainingConfig(
    hidden_layers=[64, 32, 16],
    n_particles=50,
    max_iters=200,
    alpha=(1.0, 0.5),
    patience=50
)
```

### For Large Datasets (> 10000 samples)

```python
config = TrainingConfig(
    hidden_layers=[128, 64, 32],
    n_particles=100,
    max_iters=500,
    alpha=(1.0, 0.5),
    patience=100,
    use_qdpso=True  # QDPSO can be more stable
)
```

---

## Training Workflow

```
+-------------------------------------------------------------+
|                    TRAINING WORKFLOW                         |
+-------------------------------------------------------------+

1. DATA PREPARATION
   +---------------+
   | load_dataset  | --> X_train, X_test, y_train, y_test
   +---------------+

2. CONFIGURATION
   +------------------+
   | TrainingConfig   | --> hidden_layers, n_particles, max_iters...
   +------------------+

3. TRAINER CREATION
   +----------+
   | Trainer  | --> Initializes model and optimizer internally
   +----------+

4. TRAINING
   +--------------+
   | trainer.fit  | --> Executes QPSO to optimize weights
   +--------------+
         |
         v
   +-------------------------------------------+
   |  For each QPSO iteration:                  |
   |  1. Evaluate fitness of each particle     |
   |  2. Update pbest and gbest                |
   |  3. Move particles according to QPSO eq   |
   |  4. Record metrics                        |
   +-------------------------------------------+

5. EVALUATION
   +-------------------+
   | TrainingResult    | --> train_acc, val_acc, test_acc, history...
   +-------------------+

6. SAVING (optional)
   +--------------------+
   | trainer.save_model | --> model.pth
   +--------------------+
```

---

## Executable Scripts

### Script Descriptions

| Script | Purpose | Dataset | Usage |
|--------|---------|---------|-------|
| `main_qpso.py` | QPSO benchmark | Iris, Wine, Breast Cancer | `python ann/main_qpso.py` |
| `main_qdpso.py` | QDPSO benchmark | Iris, Wine, Breast Cancer | `python ann/main_qdpso.py` |
| `main_mcw.py` | QPSO vs QDPSO comparison | MCW (weather images) | `python ann/main_mcw.py` |
| `main_training_type.py` | Training strategies benchmark | Iris, Wine, Breast Cancer | `python ann/main_training_type.py` |
| `main_hyperparameter_search.py` | Automated HPO (Optuna) | MCW | `python ann/main_hyperparameter_search.py` |
| `usage_cases.py` | Educational examples | Various | `python ann/usage_cases.py` |

### QPSO vs QDPSO Comparison

| Aspect | QPSO | QDPSO |
|--------|------|-------|
| **Optimizer** | `QPSONNOptimizer` | `QDPSONNOptimizer` |
| **Key parameter** | `alpha: (1.0, 0.5)` | `g: 0.96` |
| **Base algorithm** | QPSO with mbest | QDPSO with g factor |
| **L equation** | `L = alpha * |mbest - x|` | `L = (1/g) * |x - c|` |
| **Adaptability** | Alpha with linear decay | Constant g factor |
| **Complexity** | Higher (computes mbest) | Lower |

---

## References

1. Sun, J., Feng, B., & Xu, W. (2004). *Particle swarm optimization with particles having quantum behavior*. Congress on Evolutionary Computation.

2. Sun, J., Xu, W., & Feng, B. (2004). *A global search strategy of quantum-behaved particle swarm optimization*. IEEE Conference on Cybernetics and Intelligent Systems.

3. Sun, J., Fang, W., Wu, X., Palade, V., & Xu, W. (2012). *Quantum-behaved particle swarm optimization: Analysis of individual particle behavior and parameter selection*. Evolutionary Computation.

---

## License

This project is under the MIT license.

---

## Next Steps

Continue with the detailed documentation for each module:

### Module Documentation
- [models.md](models.md) - Neural network models
- [optimizers.md](optimizers.md) - QPSO optimizers
- [trainers.md](trainers.md) - High-level trainer
- [utils.md](utils.md) - Utilities
- [examples.md](examples.md) - Complete examples

### Scripts Documentation
- [main_qpso.md](main_qpso.md) - QPSO benchmark
- [main_qdpso.md](main_qdpso.md) - QDPSO benchmark
- [main_mcw.md](main_mcw.md) - MCW benchmark (QPSO vs QDPSO)
- [main_training_type.md](main_training_type.md) - Training strategies (Forward, Weighted, Layerwise)
- [main_hyperparameter_search.md](main_hyperparameter_search.md) - Hyperparameter search with Optuna
- [usage_cases.md](usage_cases.md) - Usage examples

---

<div align="center">

**[⬆️ Back to Top](#ann-module---qpso-neural-network-training)** | **[🏠 Main README](../../README.md)** | **[🔧 QPSO Algorithms](../../QPSO-PyTorch/docs/index.md)**

</div>

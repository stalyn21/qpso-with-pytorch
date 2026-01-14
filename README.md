# QPSO with PyTorch

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**[Version en Espanol](README_ES.md)**

A suite of **Quantum Particle Swarm Optimization (QPSO)** applications implemented in PyTorch with full GPU acceleration.

---

## Table of Contents

- [Description](#description)
- [What is QPSO?](#what-is-qpso)
- [Repository Architecture](#repository-architecture)
- [Modules Overview](#modules-overview)
- [Installation](#installation)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
- [Detailed Documentation](#detailed-documentation)
- [API Reference](#api-reference)
- [Example Results](#example-results)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Changelog](#changelog)

---

## Description

This repository contains optimized implementations of the QPSO algorithm and its variants, along with practical applications for neural network training and hyperparameter optimization.

### Key Features

- **GPU-Accelerated Optimization**: Full PyTorch tensor support for CUDA acceleration
- **Multiple Algorithm Variants**: QPSO (original) and QDPSO (delta variant)
- **Neural Network Training**: Train neural networks using QPSO and QDPSO instead of classical gradient-based implementations
- **Multiple Training Strategies**: Forward, Weighted, and Layerwise optimization
- **Hyperparameter Optimization**: Integrated Optuna support for automated HPO
- **Comprehensive Metrics**: Cross-validation, detailed classification reports, visualizations

---

## What is QPSO?

**Quantum Particle Swarm Optimization (QPSO)** is a metaheuristic optimization algorithm inspired by the quantum behavior of particles. Unlike classical PSO, QPSO does not require velocity parameters, which simplifies its implementation and improves global convergence.

### Fundamental Equation

```
x_new = c +/- L * ln(1/u)

where:
  c = attractor point (combination of pbest and gbest)
  L = characteristic length
  u ~ U(0,1)
```

### Implemented Variants

| Algorithm | Description | Key Parameter | Formula for L |
|-----------|-------------|---------------|---------------|
| **QPSO** | Original (Sun et al., 2004). Uses mean best position (mbest) | `alpha` (0.5-1.0) | L = alpha * \|mbest - x\| |
| **QDPSO** | Delta variant. Uses distance to attractor point | `g` (~0.96) | L = (1/g) * \|x - c\| |

---

## Repository Architecture

```
qdpso/
├── README.md                    # This file (English)
├── README_ES.md                 # Spanish version
│
├── QPSO-PyTorch/                # Base QPSO algorithm implementations
│   ├── tensor_qpso/             # Core module
│   │   ├── __init__.py          # Module exports
│   │   ├── qpso.py              # NumPy implementation (reference, based on pypi qpso 0.0.1)
│   │   ├── qpso_tensor.py       # Basic PyTorch tensor implementation
│   │   └── qpso_tensor_optimized.py  # Optimized implementation (17 improvements)
│   ├── docs/                    # Algorithm documentation
│   │   ├── docs_qpso.md         # NumPy implementation docs
│   │   ├── docs_qpso_tensor.md  # Basic tensor docs
│   │   └── docs_qpso_tensor_optimized.md  # Optimized docs
│   ├── main_pypi.py             # Original QPSO example (pypi-based)
│   ├── main_qpso.py             # QPSO wrapper example
│   ├── main_qpso_tensor.py      # Tensor QPSO example
│   ├── main_qpso_tensor_optimized.py  # Optimized tensor example
│   └── get_device.py            # Device detection utility
│
└── ann/                         # Neural Network Training with QPSO
    ├── __init__.py              # Package exports
    ├── tensor_qpso/             # Local optimized QPSO module
    │   ├── __init__.py
    │   └── qpso_tensor_optimized.py
    │
    ├── models/                  # Neural network architectures
    │   ├── __init__.py
    │   └── ann.py               # QPSOCompatibleANN model
    │
    ├── optimizers/              # QPSO optimizers for neural networks
    │   ├── __init__.py
    │   ├── qpso_nn.py           # QPSONNOptimizer, QDPSONNOptimizer
    │   └── training_strategies.py  # Forward, Weighted, Layerwise strategies
    │
    ├── trainers/                # High-level training interface
    │   ├── __init__.py
    │   └── trainer.py           # Trainer class with CV support
    │
    ├── utils/                   # Utilities
    │   ├── __init__.py
    │   ├── data.py              # Data loading and preprocessing
    │   └── metrics.py           # Classification metrics and visualization
    │
    ├── data/                    # Dataset loaders
    │   ├── __init__.py
    │   └── mcw.py               # MCW (Multi-Class Weather) dataset
    │
    ├── docs/                    # Detailed module documentation
    │   ├── index.md             # Documentation index
    │   ├── models.md            # Models documentation
    │   ├── optimizers.md        # Optimizers documentation
    │   ├── trainers.md          # Trainers documentation
    │   └── *.md                 # Script-specific documentation
    │
    ├── main_qpso.py             # QPSO benchmark script
    ├── main_qdpso.py            # QDPSO benchmark script
    ├── main_mcw.py              # MCW image classification benchmark
    ├── main_training_type.py    # Training strategies comparison
    ├── main_hyperparameter_search.py  # HPO with Optuna
    ├── start_hyperparameter_search.py # HPO configuration script
    └── usage_cases.py           # Usage examples
```

---

## Modules Overview

### 1. QPSO-PyTorch Module

The base QPSO algorithm implementations with progressive optimizations:

| File | Description | Use Case |
|------|-------------|----------|
| `qpso.py` | NumPy implementation based on pypi qpso 0.0.1 | Reference/learning |
| `qpso_tensor.py` | Basic PyTorch tensor implementation | Simple GPU optimization |
| `qpso_tensor_optimized.py` | Fully optimized implementation | Production use |

**Optimizations in qpso_tensor_optimized.py (17 improvements):**

- **Performance**: Efficient sign generation, memory pooling, `torch.no_grad()`
- **Stability**: Safe division, configurable dtype, epsilon handling
- **Functionality**: Boundary handling (clamp/reflect/wrap/random), early convergence, history tracking
- **Robustness**: Parameter validation, NaN/Inf handling
- **Usability**: `OptimizationResult` dataclass, context manager support
- **Extensibility**: Event-based callback system (ON_INIT, ON_ITERATION_START, ON_NEW_BEST, etc.)

### 2. ANN Module

Neural network training framework using QPSO:

| Submodule | Description | Key Classes |
|-----------|-------------|-------------|
| `models/` | QPSO-compatible neural networks | `QPSOCompatibleANN` |
| `optimizers/` | Neural network optimizers | `QPSONNOptimizer`, `QDPSONNOptimizer` |
| `trainers/` | High-level training interface | `Trainer`, `TrainingConfig` |
| `utils/` | Data and metrics utilities | `load_dataset`, `MulticlassMetrics` |
| `data/` | Dataset loaders | `MCWDataset`, `load_mcw` |

**Training Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Forward** | Optimizes all weights simultaneously | Small networks, fast training |
| **Weighted** | Layer-weighted fitness with decay | Medium networks, balanced |
| **Layerwise** | Sequential layer-by-layer training | Deep networks, better convergence |

---

## Installation

> **Detailed Guide**: See [docs/installation.md](docs/installation.md) for complete installation instructions.

### Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | >= 3.10 | 3.12 |
| **PyTorch** | >= 2.0.0 | 2.5.1 |
| **CUDA** | - | 12.4 |
| **cuDNN** | - | 9.1.0 |

### Quick Installation with Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/stalyn21/qpso-with-pytorch.git
cd pytorch-qpso-suite

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate pytorch_qpso_gpu
```

### Installation with Pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install -r requirements.txt
```

### CPU-Only Installation

```bash
# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
pip install -r requirements.txt
```

### Verification

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check ann module
python -c "from ann import QPSOCompatibleANN, Trainer; print('ann module OK')"
```

---

## Datasets

### MCW (Multi-Class Weather) Dataset

The MCW dataset is **not included** in this repository due to its size. You must download it separately to run the weather image classification examples.

#### Download

Download the dataset from Kaggle:

**[Multi-Class Weather Dataset](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset)**

#### Installation

1. Download and extract the dataset
2. Place the images in the following structure:

```
ann/data/img/mcw/
├── cloudy/
│   ├── cloudy1.jpg
│   ├── cloudy2.jpg
│   └── ...
├── rain/
│   ├── rain1.jpg
│   ├── rain2.jpg
│   └── ...
├── shine/
│   ├── shine1.jpg
│   ├── shine2.jpg
│   └── ...
└── sunrise/
    ├── sunrise1.jpg
    ├── sunrise2.jpg
    └── ...
```

#### Troubleshooting

Some images in the dataset may cause loading issues due to corruption or unsupported formats. If you encounter errors during data loading:

1. **Identify problematic images**: The error message will show which file caused the issue
2. **Remove or replace**: Delete the problematic image or replace it with a valid one
3. **Common issues**:
   - Truncated/corrupted JPEG files
   - Images with unusual color spaces
   - Files with wrong extensions

```python
# Example: Test if an image loads correctly
import cv2
img = cv2.imread('path/to/image.jpg')
if img is None:
    print("Image failed to load - remove or replace it")
```

### Other Datasets

The following datasets are automatically downloaded via scikit-learn:
- **Iris**: 150 samples, 4 features, 3 classes
- **Wine**: 178 samples, 13 features, 3 classes
- **Breast Cancer**: 569 samples, 30 features, 2 classes
- **Digits**: 1797 samples, 64 features, 10 classes

No additional setup required for these datasets.

---

## Quick Start

### 1. Basic QPSO Optimization

```python
from QPSO_PyTorch.tensor_qpso import QPSOTensorOptimized

# Define cost function
def sphere(x):
    return (x ** 2).sum()

# Create optimizer
optimizer = QPSOTensorOptimized(
    cf=sphere,
    size=50,                    # Number of particles
    dim=10,                     # Dimensions
    bounds=[(-5, 5)] * 10,      # Search bounds
    maxIters=1000,              # Maximum iterations
    alpha=(1.0, 0.5),           # Linear decay from 1.0 to 0.5
    device='cuda',              # Use GPU
    track_history=True          # Track optimization history
)

# Run optimization
result = optimizer.optimize()
print(f"Best value: {result.best_value:.6e}")
print(f"Iterations: {result.iterations}")
print(f"Time: {result.elapsed_time:.2f}s")
```

### 2. Neural Network Training with QPSO

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset

# Load dataset
X_train, X_test, y_train, y_test = load_dataset('iris')

# Configure training
config = TrainingConfig(
    hidden_layers=[16, 8],      # Network architecture
    activation='tanh',          # Activation function
    n_particles=30,             # Swarm size
    max_iters=100,              # Maximum iterations
    alpha=(1.0, 0.5),           # QPSO alpha parameter
    patience=30,                # Early stopping patience
    random_state=42             # Reproducibility
)

# Create trainer and train
trainer = Trainer(input_dim=4, output_dim=3, config=config)
result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

print(f"Test Accuracy: {result.test_accuracy:.4f}")
```

### 3. Compare Training Strategies

```python
from ann.optimizers.training_strategies import create_training_strategy, StrategyConfig
from ann.models import QPSOCompatibleANN

# Create model
model = QPSOCompatibleANN(
    input_dim=4,
    output_dim=3,
    hidden_layers=[16, 8]
)

# Try different strategies
for strategy_name in ['forward', 'weighted', 'layerwise']:
    config = StrategyConfig(
        strategy=strategy_name,
        n_particles=30,
        max_iters=100
    )

    strategy = create_training_strategy(model, config)
    result = strategy.train(X_train, y_train, X_val, y_val)
    print(f"{strategy_name}: {result['val_accuracy']:.4f}")
```

### 4. Hyperparameter Optimization

```bash
# Edit start_hyperparameter_search.py to configure search space
cd ann
python start_hyperparameter_search.py
```

---

## Detailed Documentation

### Documentation Indexes

| Module | Index | Description |
|--------|-------|-------------|
| **QPSO-PyTorch** | [📚 index.md](QPSO-PyTorch/docs/index.md) | QPSO algorithm implementations |
| **ANN** | [📚 index.md](ann/docs/index.md) | Neural network training with QPSO |

### QPSO-PyTorch Module

| Document | Description |
|----------|-------------|
| [docs_qpso.md](QPSO-PyTorch/docs/docs_qpso.md) | NumPy reference implementation (pypi-based) |
| [docs_qpso_tensor.md](QPSO-PyTorch/docs/docs_qpso_tensor.md) | Basic PyTorch tensor implementation |
| [docs_qpso_tensor_optimized.md](QPSO-PyTorch/docs/docs_qpso_tensor_optimized.md) | Optimized implementation (17 improvements) |
| [implementation_comparison.md](QPSO-PyTorch/docs/implementation_comparison.md) | **Implementation comparison**: coding approach, performance, features |

### ANN Module

| Document | Description |
|----------|-------------|
| [index.md](ann/docs/index.md) | Module overview and architecture |
| [models.md](ann/docs/models.md) | QPSOCompatibleANN model documentation |
| [optimizers.md](ann/docs/optimizers.md) | QPSONNOptimizer, training strategies |
| [trainers.md](ann/docs/trainers.md) | Trainer class and configuration |
| [utils.md](ann/docs/utils.md) | Data loading and metrics |

### Executable Scripts

| Script | Document | Description |
|--------|----------|-------------|
| `main_qpso.py` | [main_qpso.md](ann/docs/main_qpso.md) | QPSO benchmark on classic datasets |
| `main_qdpso.py` | [main_qdpso.md](ann/docs/main_qdpso.md) | QDPSO benchmark |
| `main_mcw.py` | [main_mcw.md](ann/docs/main_mcw.md) | MCW image classification |
| `main_training_type.py` | [main_training_type.md](ann/docs/main_training_type.md) | Strategy comparison |
| `main_hyperparameter_search.py` | [main_hyperparameter_search.md](ann/docs/main_hyperparameter_search.md) | HPO with Optuna |
| `usage_cases.py` | [usage_cases.md](ann/docs/usage_cases.md) | Educational examples |

---

## API Reference

### QPSOTensorOptimized

```python
QPSOTensorOptimized(
    cf: Callable,                       # Cost function to minimize/maximize
    size: int,                          # Number of particles
    dim: int,                           # Problem dimensions
    bounds: List[Tuple[float, float]],  # Bounds per dimension [(min, max), ...]
    maxIters: int,                      # Maximum iterations
    alpha: Union[float, Tuple] = 0.75,  # Contraction-expansion coefficient
                                        # float: fixed value
                                        # tuple: (max, min) for linear decay
    device: str = 'auto',               # 'cpu', 'cuda', 'cuda:N', 'mps', 'auto'
    dtype: torch.dtype = torch.float32, # Tensor data type
    seed: Optional[int] = None,         # Random seed for reproducibility
    boundary_strategy: str = 'clamp',   # 'none', 'clamp', 'reflect', 'wrap', 'random'
    tol: float = 1e-12,                 # Convergence tolerance
    patience: int = 100,                # Iterations without improvement before stopping
    track_history: bool = False,        # Record optimization history
    minimize: bool = True               # True to minimize, False to maximize
)

# Methods
result = optimizer.optimize(callback=None, interval=None)  # Returns OptimizationResult
```

### QDPSOTensorOptimized

```python
QDPSOTensorOptimized(
    # Same parameters as QPSOTensorOptimized, except:
    g: float = 0.96,                    # Delta parameter instead of alpha
)
```

### QPSOCompatibleANN

```python
QPSOCompatibleANN(
    input_dim: int,                     # Input dimension
    output_dim: int,                    # Output dimension (classes)
    hidden_layers: List[int],           # Neurons per hidden layer [64, 32, 16]
    activation: str = 'relu',           # 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu'
    output_activation: str = 'softmax', # 'softmax', 'sigmoid', 'log_softmax', None
    dropout: float = 0.0,               # Dropout probability
    use_batch_norm: bool = False,       # Use batch normalization
    device: str = 'auto'                # Device
)

# Properties
model.num_params                        # Total number of parameters

# Methods
params = model.get_flat_params()        # Get all params as 1D tensor
model.set_flat_params(params)           # Set params from 1D tensor
bounds = model.get_param_bounds(1.0)    # Get bounds for QPSO
```

### Trainer

```python
Trainer(
    input_dim: int,
    output_dim: int,
    config: TrainingConfig
)

# Methods
result = trainer.fit(X_train, y_train, X_test=None, y_test=None)
result = trainer.fit_cv(X, y, X_test=None, y_test=None)  # Cross-validation
predictions = trainer.predict(X)
trainer.save_model(path)
trainer.load_model(path)
```

### TrainingConfig

```python
TrainingConfig(
    # Architecture
    hidden_layers: List[int] = [32, 16],
    activation: str = 'tanh',

    # QPSO parameters
    n_particles: int = 50,
    max_iters: int = 100,
    alpha: Tuple[float, float] = (1.0, 0.5),
    g: float = 0.96,                    # For QDPSO
    use_qdpso: bool = False,

    # Training
    weight_bound: float = 1.0,
    patience: int = 50,

    # Cross-validation
    n_folds: int = 5,

    # Other
    random_state: int = 42,
    verbose: bool = True,
    save_best_model: bool = False,
    output_dir: str = './output'
)
```

---

## Example Results

### Classic Dataset Benchmarks

| Dataset | Classes | Features | QPSO Acc | QDPSO Acc |
|---------|---------|----------|----------|-----------|
| Iris | 3 | 4 | 96.7% | 97.3% |
| Wine | 3 | 13 | 94.4% | 95.6% |
| Breast Cancer | 2 | 30 | 96.5% | 97.1% |
| Digits | 10 | 64 | 92.3% | 93.8% |

### MCW (Weather Image Classification)

| Metric | QPSO | QDPSO |
|--------|------|-------|
| Accuracy | 82.5% | 85.0% |
| F1-Score | 0.81 | 0.84 |
| Precision | 0.83 | 0.86 |
| Recall | 0.80 | 0.83 |
| Cohen's Kappa | 0.76 | 0.80 |

### Training Strategy Comparison

| Strategy | Accuracy | Training Time | Best For |
|----------|----------|---------------|----------|
| Forward | 94.2% | Fast | Small networks |
| Weighted | 95.1% | Medium | Balanced |
| Layerwise | 96.3% | Slow | Deep networks |

### HPO Best Configuration

```
Optimizer: QDPSO
Strategy: Layerwise
g: 0.9534
Particles: 52
Architecture: [28, 18, 11]
Iterations per layer: 45
Fine-tune iterations: 30

Results:
  CV F1-Score: 0.865
  Test Accuracy: 87.5%
  Test F1-Score: 0.871
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## References

### Fundamental Papers

1. **Original QPSO**: Sun, J., Feng, B., & Xu, W. (2004). *Particle swarm optimization with particles having quantum behavior*. Congress on Evolutionary Computation.

2. **QPSO Analysis**: Sun, J., Fang, W., Wu, X., Palade, V., & Xu, W. (2012). *Quantum-behaved particle swarm optimization: Analysis of individual particle behavior and parameter selection*. Evolutionary Computation.

3. **Optuna**: Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD.

### Reference Implementations

- [QPSO PyPI Package](https://pypi.org/project/qpso/) - Original Python package (v0.0.1)

---

## Citation

If you use this software in your research or project, please cite it:

### BibTeX Format

```bibtex
@software{chancay2024pytorchqpso,
  author       = {Chancay Moreira, Stalyn Javier},
  title        = {PyTorch QPSO Suite: Quantum Particle Swarm Optimization for Neural Network Training},
  year         = {2024},
  version      = {2.0.0},
  url          = {https://github.com/stalyn21/qpso-with-pytorch}
}
```

### APA Format

> Chancay Moreira, S. J. (2024). *PyTorch QPSO Suite: Quantum Particle Swarm Optimization for Neural Network Training* (Version 2.0.0) [Software]. https://github.com/stalyn21/qpso-with-pytorch

---

## License

This project is under the MIT license. See [LICENSE](LICENSE) for more details.

**Copyright (c) 2024 Stalyn Javier Chancay Moreira**

Use, copy, modification, and distribution of this software is permitted provided that:
- The original copyright notice is retained
- The MIT license is included in any redistribution

---

## Author

**Stalyn Javier Chancay Moreira**

- GitHub: [@stalyn21](https://github.com/stalyn21)

---

## Acknowledgments

This project is based on the fundamental works of:
- Sun, J., Feng, B., & Xu, W. - Creators of the original QPSO algorithm
- The PyTorch community for the excellent deep learning framework
- The Optuna team for the hyperparameter optimization framework

---

## Changelog

### v2.0.0 (2024)
- **Complete refactoring** of the codebase
- Renamed `src/` module to `ann/` for clarity
- Improved module organization and imports
- Enhanced documentation with detailed API reference
- Added comprehensive type hints throughout
- Performance optimizations in tensor operations
- Improved error handling and validation
- Added Apple Silicon (MPS) support
- Updated all documentation to reflect new structure

### v1.0.0 (2024)
- Initial implementation of QPSO/QDPSO with PyTorch tensors
- Modular framework for neural network training without backpropagation
- Three training strategies: Forward, Weighted, Layerwise
- HPO (Hyperparameter Optimization) support with Optuna
- Full GPU acceleration
- Complete documentation in Spanish and English

# QPSO Implementation Comparison

[🏠 README](../../README.md) | [📚 Index](index.md) | [⬅️ Prev: Optimized](docs_qpso_tensor_optimized.md) | **Comparison** | [🇪🇸 Español](implementation_comparison_es.md)

---

> **Version**: 2.0.0
> **Module**: QPSO-PyTorch/tensor_qpso/

This document provides a detailed comparison of the three QPSO implementations in terms of coding approach, performance, and features.

---

## Overview

The `tensor_qpso/` module contains three evolutionary implementations of QPSO:

| File | Name | Approach | GPU Support |
|------|------|----------|-------------|
| `qpso.py` | PyPI Reference | Scalar/Iterative (NumPy) | No |
| `qpso_tensor.py` | Tensor Basic | Vectorized (PyTorch) | Yes |
| `qpso_tensor_optimized.py` | Tensor Optimized | Vectorized + 17 Optimizations | Yes + MPS |

---

## 1. Architecture Comparison

### 1.1 Class Hierarchy

#### qpso.py (PyPI Reference)
```
Particle          <- Individual particle with position and best
    |
Swarm             <- Collection of particles, manages gbest
    |
QPSOBase          <- Base optimization logic
    |
+---+---+
|       |
QPSO    QDPSO     <- Specific algorithms
```

#### qpso_tensor.py (Tensor Basic)
```
SwarmTensor       <- All particles as tensors
    |
QPSOBaseTensor    <- Base optimization with tensor operations
    |
+---+---+
|       |
QPSOTensor  QDPSOTensor  <- Specific algorithms
```

#### qpso_tensor_optimized.py (Tensor Optimized)
```
CallbackManager           <- Event-based callback system
OptimizationResult        <- Structured result dataclass
BoundaryStrategy (Enum)   <- Boundary handling options
    |
SwarmTensorOptimized      <- Optimized swarm with validation
    |
QPSOBaseTensorOptimized   <- Full-featured base class
    |
+---+---+
|       |
QPSOTensorOptimized  QDPSOTensorOptimized
```

---

## 2. Data Structure Comparison

### 2.1 Particle Representation

#### qpso.py - Object-Oriented
```python
class Particle(object):
    def __init__(self, bounds):
        self._x = np.zeros(len(bounds))        # Position: 1D array
        for idx, (lo, hi) in enumerate(bounds):
            self._x[idx] = random.uniform(lo, hi)
        self._best = self._x.copy()            # Personal best
        self._best_value = np.nan              # Scalar value
```
- Each particle is a separate object
- Properties accessed via getters/setters
- Memory: N objects with individual arrays

#### qpso_tensor.py - Tensor-Based
```python
class SwarmTensor:
    def __init__(self, size, dim, bounds, device='auto'):
        # All particles in single tensor
        self._positions = torch.rand(size, dim, device=self._device) * \
                         (self._upper - self._lower) + self._lower
        self._pbest = self._positions.clone()
        self._pbest_values = torch.full((size,), float('inf'), device=self._device)
```
- All particles in a single 2D tensor `(n_particles, dim)`
- Efficient memory layout for GPU
- Single allocation for all particles

#### qpso_tensor_optimized.py - Optimized Tensors
```python
class SwarmTensorOptimized:
    def __init__(self, size, dim, bounds, device='auto',
                 dtype=torch.float32, seed=None):
        # Validation
        if size <= 0:
            raise ValueError(f"size must be positive")
        validate_bounds(bounds, dim)

        # Configurable dtype and seed
        if seed is not None:
            torch.manual_seed(seed)

        self._dtype = dtype
        self._eps = torch.finfo(dtype).eps  # Numerical stability

        # Optimized initialization
        self._positions = self._random_positions(size)
```
- Parameter validation
- Configurable dtype (float32, float64, float16)
- Reproducibility with seed
- Epsilon for numerical stability

---

## 3. Core Algorithm Implementation

### 3.1 QPSO Kernel Update

#### qpso.py - Scalar Loops
```python
def kernel_update(self, **kwargs):
    mbest = self.mean_best()
    alpha = self._get_alpha()

    for p in self._particles:              # Loop over particles
        for i in range(0, self._dim):      # Loop over dimensions
            phi = random.uniform(0., 1.)
            u = random.uniform(0., 1.)
            rand_sign = 1 if random.random() > 0.5 else -1

            c = phi * p.best[i] + (1 - phi) * self._gbest[i]
            L = alpha * abs(mbest[i] - p[i])
            p[i] = c + rand_sign * L * np.log(1. / u)
```
**Characteristics:**
- Double nested loop: O(n * d) iterations
- Individual random number generation per dimension
- No GPU acceleration possible
- Simple but slow for high dimensions

#### qpso_tensor.py - Vectorized
```python
def kernel_update(self) -> None:
    mbest = self.mean_best()  # (dim,)
    alpha = self._get_alpha()

    # Generate all random numbers at once
    phi = torch.rand(n, d, device=self._device)
    u = torch.rand(n, d, device=self._device)
    u = torch.clamp(u, min=1e-10)  # Avoid log(0)

    # Random signs using torch.where
    rand_sign = torch.where(
        torch.rand(n, d, device=self._device) > 0.5,
        torch.ones(n, d, device=self._device),
        -torch.ones(n, d, device=self._device)
    )

    c = phi * self._pbest + (1 - phi) * self._gbest
    L = alpha * torch.abs(mbest - self._positions)
    self._positions = c + rand_sign * L * torch.log(1.0 / u)
```
**Characteristics:**
- No explicit loops - all vectorized
- Parallel execution on GPU
- Multiple tensor allocations per iteration
- 3x torch.rand() calls per iteration

#### qpso_tensor_optimized.py - Optimized Vectorized
```python
def kernel_update(self) -> None:
    mbest = self.mean_best()
    alpha = self._get_alpha()

    # OPTIMIZATION 1: Single batch random generation
    all_random = self._generate_random_batch(num_channels=2)
    phi = all_random[:, :, 0]
    u = all_random[:, :, 1]

    # OPTIMIZATION 2: Proper epsilon from dtype
    u = torch.clamp(u, min=self._eps, max=1.0 - self._eps)

    # OPTIMIZATION 3: Efficient sign generation
    rand_sign = self._generate_signs()

    c = phi * self._pbest + (1.0 - phi) * self._gbest
    L = alpha * torch.abs(mbest - self._positions)
    self._positions = c + rand_sign * L * torch.log(1.0 / u)

def _generate_random_batch(self, num_channels: int = 4) -> torch.Tensor:
    """Single call for all random numbers"""
    return torch.rand(
        self._size, self._dim, num_channels,
        dtype=self._dtype, device=self._device
    )

def _generate_signs(self) -> torch.Tensor:
    """Efficient sign generation using randint"""
    return torch.randint(
        0, 2, (self._size, self._dim),
        dtype=self._dtype, device=self._device
    ) * 2 - 1
```
**Characteristics:**
- Single torch.rand() call instead of 3
- randint for signs (faster than where + comparisons)
- Correct epsilon based on dtype
- Pre-allocated work tensors

---

## 4. Feature Comparison Table

| Feature | qpso.py | qpso_tensor.py | qpso_tensor_optimized.py |
|---------|---------|----------------|--------------------------|
| **Execution** | | | |
| GPU Support | No | Yes (CUDA) | Yes (CUDA + MPS) |
| Vectorized Operations | No | Yes | Yes |
| Parallel Particle Update | No | Yes | Yes |
| **Memory** | | | |
| Pre-allocated Memory | No | No | Yes (memory pool) |
| Configurable dtype | No | No | Yes (float16/32/64) |
| Memory Cleanup | No | No | Yes (context manager) |
| **Numerical Stability** | | | |
| Division by Zero Protection | No | Partial | Yes |
| NaN/Inf Handling | No | No | Yes |
| Epsilon based on dtype | No | Hardcoded | Yes |
| **Functionality** | | | |
| Boundary Handling | No | No | Yes (5 strategies) |
| Early Convergence | No | No | Yes (tolerance + patience) |
| History Tracking | No | No | Yes |
| Reproducibility (seed) | No | No | Yes |
| Maximize/Minimize | Minimize only | Minimize only | Both |
| **Extensibility** | | | |
| Callback System | Basic | Basic | Advanced (6 events) |
| Structured Results | No | No | Yes (OptimizationResult) |
| Context Manager | No | No | Yes |
| Parameter Validation | No | No | Yes |

---

## 5. Random Number Generation Comparison

### 5.1 qpso.py - Python random module
```python
# 4 calls per particle per dimension per iteration
phi = random.uniform(0., 1.)
u = random.uniform(0., 1.)
rand_sign = 1 if random.random() > 0.5 else -1
# For QDPSO: u1, u2, u3 = 3 more calls
```
**Total calls per iteration**: 4 * n_particles * dim (QPSO) or 5 * n_particles * dim (QDPSO)

### 5.2 qpso_tensor.py - Multiple torch.rand()
```python
phi = torch.rand(n, d, device=self._device)      # Call 1
u = torch.rand(n, d, device=self._device)        # Call 2
torch.rand(n, d, device=self._device)            # Call 3 (for signs)
```
**Total calls per iteration**: 3 (or 4 for QDPSO)

### 5.3 qpso_tensor_optimized.py - Batch generation
```python
# Single call generates all random numbers
all_random = torch.rand(n, d, num_channels, ...)  # Call 1
# Signs use different efficient method
rand_sign = torch.randint(0, 2, (n, d), ...)      # Call 2
```
**Total calls per iteration**: 2

---

## 6. Boundary Handling Comparison

### 6.1 qpso.py & qpso_tensor.py
```python
# No boundary handling - particles can go outside bounds
```

### 6.2 qpso_tensor_optimized.py - 5 Strategies
```python
class BoundaryStrategy(Enum):
    NONE = "none"       # No restriction
    CLAMP = "clamp"     # Clamp to bounds
    REFLECT = "reflect" # Bounce off bounds
    WRAP = "wrap"       # Circular wrap-around
    RANDOM = "random"   # Re-initialize randomly

def _apply_boundary(self, positions):
    if strategy == BoundaryStrategy.CLAMP:
        return torch.clamp(positions, min=self._lower, max=self._upper)

    elif strategy == BoundaryStrategy.REFLECT:
        # Reflect formula handling multiple bounces
        normalized = (result - lower) % (2 * range_size)
        result = torch.where(
            normalized > range_size,
            2 * range_size - normalized + lower,
            normalized + lower
        )

    elif strategy == BoundaryStrategy.WRAP:
        return self._lower + (positions - self._lower) % range_size

    elif strategy == BoundaryStrategy.RANDOM:
        outside = (result < lower) | (result > upper)
        if outside.any():
            new_positions = lower + random_vals * range_size
            result = torch.where(outside, new_positions, result)
```

---

## 7. Callback System Comparison

### 7.1 qpso.py & qpso_tensor.py - Basic Callback
```python
def update(self, callback=None, interval=None):
    while self._iters <= self._maxIters:
        self.kernel_update()
        self.update_best()
        if callback and (self._iters % interval == 0):
            callback(self)  # Simple callback
        self._iters += 1
```

### 7.2 qpso_tensor_optimized.py - Event-Based System
```python
class CallbackEvent(Enum):
    ON_INIT = "on_init"
    ON_ITERATION_START = "on_iteration_start"
    ON_ITERATION_END = "on_iteration_end"
    ON_NEW_BEST = "on_new_best"
    ON_CONVERGENCE = "on_convergence"
    ON_FINISH = "on_finish"

class CallbackManager:
    def register(self, event: CallbackEvent, callback: Callable):
        self._callbacks[event].append(callback)

    def trigger(self, event: CallbackEvent, optimizer):
        for callback in self._callbacks[event]:
            callback(optimizer)

# Usage in update loop:
def update(self, ...):
    while self._iters <= self._maxIters:
        self._callbacks.trigger(CallbackEvent.ON_ITERATION_START, self)
        self.kernel_update()

        gbest_improved = self.update_best()
        if gbest_improved:
            self._callbacks.trigger(CallbackEvent.ON_NEW_BEST, self)

        if self._check_convergence():
            self._callbacks.trigger(CallbackEvent.ON_CONVERGENCE, self)
            break

        self._callbacks.trigger(CallbackEvent.ON_ITERATION_END, self)
```

---

## 8. Result Structure Comparison

### 8.1 qpso.py & qpso_tensor.py
```python
# Access results via properties after optimization
optimizer.update()
best_position = optimizer.gbest
best_value = optimizer.gbest_value
iterations = optimizer.iters
```

### 8.2 qpso_tensor_optimized.py - OptimizationResult
```python
@dataclass
class OptimizationResult:
    best_position: torch.Tensor
    best_value: float
    iterations: int
    converged: bool
    convergence_reason: str
    history: Optional[Dict[str, List]]
    device: str
    elapsed_time: float

    def to_numpy(self) -> Dict[str, Any]:
        """Convert to dictionary with NumPy arrays"""
        ...

# Usage
result = optimizer.optimize()
print(result)
# OptimizationResult(
#   best_value=1.234567E-10,
#   iterations=543,
#   converged=True,
#   reason='Convergence: no improvement > 1e-12 for 100 iterations',
#   device='cuda:0',
#   time=2.345s
# )
```

---

## 9. Performance Summary

### Relative Performance (higher is better)

| Metric | qpso.py | qpso_tensor.py | qpso_tensor_optimized.py |
|--------|---------|----------------|--------------------------|
| **CPU (small dim)** | 1.0x | 2-3x | 3-5x |
| **CPU (large dim)** | 1.0x | 5-10x | 10-20x |
| **GPU (small dim)** | N/A | 5-10x | 10-20x |
| **GPU (large dim)** | N/A | 50-100x | 100-200x |
| **Memory efficiency** | 1.0x | 2x | 3-4x |

### When to Use Each Implementation

| Implementation | Best Use Case |
|---------------|---------------|
| **qpso.py** | Learning, reference, debugging |
| **qpso_tensor.py** | Quick GPU acceleration, simple problems |
| **qpso_tensor_optimized.py** | Production, large-scale problems, research |

---

## 10. Code Examples

### 10.1 Basic Optimization (All Implementations)

#### qpso.py
```python
from tensor_qpso.qpso import QPSO

def sphere(x):
    return sum(xi**2 for xi in x)

optimizer = QPSO(
    cf=sphere,
    size=50,
    dim=10,
    bounds=[(-5, 5)] * 10,
    maxIters=1000,
    alpha=(1.0, 0.5)
)
optimizer.update()
print(f"Best: {optimizer.gbest_value}")
```

#### qpso_tensor.py
```python
from tensor_qpso.qpso_tensor import QPSOTensor

def sphere(x):
    return (x ** 2).sum(dim=-1)  # Vectorized

optimizer = QPSOTensor(
    cf=sphere,
    size=50,
    dim=10,
    bounds=[(-5, 5)] * 10,
    maxIters=1000,
    alpha=(1.0, 0.5),
    device='cuda'
)
optimizer.update()
print(f"Best: {optimizer.gbest_value}")
```

#### qpso_tensor_optimized.py
```python
from tensor_qpso.qpso_tensor_optimized import QPSOTensorOptimized

def sphere(x):
    return (x ** 2).sum(dim=-1)

# Full-featured usage
optimizer = QPSOTensorOptimized(
    cf=sphere,
    size=50,
    dim=10,
    bounds=[(-5, 5)] * 10,
    maxIters=1000,
    alpha=(1.0, 0.5),
    device='cuda',
    seed=42,
    boundary_strategy='clamp',
    tol=1e-12,
    patience=100,
    track_history=True
)

result = optimizer.optimize()
print(result)
print(f"History length: {len(result.history['gbest_value'])}")
```

---

## 11. Summary

The three implementations represent an evolutionary progression:

1. **qpso.py**: Faithful PyPI reference implementation for learning and debugging
2. **qpso_tensor.py**: Tensor-based vectorization for basic GPU acceleration
3. **qpso_tensor_optimized.py**: Production-ready with all optimizations and features

For new projects, **qpso_tensor_optimized.py** is recommended as it provides:
- Best performance through multiple optimizations
- Numerical stability for robust execution
- Comprehensive features for research and production
- Full compatibility with the `ann/` neural network training module

---

## Related Documents

- [📘 NumPy Implementation](docs_qpso.md) - Detailed documentation of the reference implementation
- [📗 Tensor Implementation](docs_qpso_tensor.md) - PyTorch tensor version documentation
- [📙 Optimized Implementation](docs_qpso_tensor_optimized.md) - Full documentation with 17 improvements
- [📦 ANN Module](../../ann/docs/index.md) - Neural network training using QPSO

---

<div align="center">

**[⬆️ Back to Top](#qpso-implementation-comparison)** | **[📚 Index](index.md)** | **[🏠 README](../../README.md)** | **[🇪🇸 Español](implementation_comparison_es.md)**

</div>

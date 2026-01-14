# QPSO-PyTorch Documentation

[🏠 Main README](../../README.md) | [📦 ANN Module](../../ann/docs/index.md) | **QPSO Algorithms** | [🇪🇸 Español](index_es.md)

---

> **Module**: QPSO-PyTorch/tensor_qpso/
> **Version**: 2.0.0

This documentation covers the core QPSO (Quantum Particle Swarm Optimization) algorithm implementations.

---

## Quick Navigation

### Implementation Documentation

| Document | Description | Level |
|----------|-------------|-------|
| [📘 docs_qpso.md](docs_qpso.md) | NumPy reference implementation (pypi-based) | Beginner |
| [📗 docs_qpso_tensor.md](docs_qpso_tensor.md) | PyTorch tensor implementation | Intermediate |
| [📙 docs_qpso_tensor_optimized.md](docs_qpso_tensor_optimized.md) | Optimized implementation (17 improvements) | Advanced |

### Comparative Analysis

| Document | Description |
|----------|-------------|
| [📊 implementation_comparison.md](implementation_comparison.md) | **Detailed comparison** of all three implementations |
| [📊 implementation_comparison_es.md](implementation_comparison_es.md) | Comparison in Spanish |

---

## Implementations Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    QPSO Implementations                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   qpso.py    │───▶│qpso_tensor.py│───▶│qpso_tensor_      │  │
│  │              │    │              │    │  optimized.py    │  │
│  │  Reference   │    │  Vectorized  │    │  Production      │  │
│  │  NumPy       │    │  PyTorch     │    │  17 improvements │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│        ▲                    ▲                     ▲             │
│        │                    │                     │             │
│    Learning            GPU Accel.            Production         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recommended Reading Order

### For Beginners

1. **Start here**: [docs_qpso.md](docs_qpso.md) - Understand the basic QPSO/QDPSO algorithms
2. **Then**: [implementation_comparison.md](implementation_comparison.md) - See how implementations differ

### For Intermediate Users

1. **Start here**: [docs_qpso_tensor.md](docs_qpso_tensor.md) - Learn tensor-based optimization
2. **Then**: [docs_qpso_tensor_optimized.md](docs_qpso_tensor_optimized.md) - Advanced features

### For Production Use

1. **Go directly to**: [docs_qpso_tensor_optimized.md](docs_qpso_tensor_optimized.md) - Full-featured implementation
2. **Reference**: [implementation_comparison.md](implementation_comparison.md) - Performance benchmarks

---

## Quick Links

### Main Executable Scripts

| Script | Description |
|--------|-------------|
| [main_pypi.py](../main_pypi.py) | Original QPSO example (pypi-based) |
| [main_qpso.py](../main_qpso.py) | QPSO wrapper example |
| [main_qpso_tensor.py](../main_qpso_tensor.py) | Tensor QPSO example |
| [main_qpso_tensor_optimized.py](../main_qpso_tensor_optimized.py) | Optimized tensor example |

### Source Files

| File | Description |
|------|-------------|
| [tensor_qpso/qpso.py](../tensor_qpso/qpso.py) | NumPy implementation |
| [tensor_qpso/qpso_tensor.py](../tensor_qpso/qpso_tensor.py) | Basic tensor implementation |
| [tensor_qpso/qpso_tensor_optimized.py](../tensor_qpso/qpso_tensor_optimized.py) | Optimized implementation |

---

## Related Documentation

### ANN Module (Neural Network Training)

The ANN module uses QPSO to train neural networks without backpropagation:

- [📚 ANN Module Index](../../ann/docs/index.md) - Main documentation
- [🧠 Models](../../ann/docs/models.md) - QPSOCompatibleANN
- [⚙️ Optimizers](../../ann/docs/optimizers.md) - QPSONNOptimizer, training strategies
- [🏋️ Trainers](../../ann/docs/trainers.md) - High-level training interface

---

## See Also

- [🏠 Main README](../../README.md) - Project overview
- [🏠 Main README (ES)](../../README_ES.md) - Project overview in Spanish

---

<div align="center">

**[⬆️ Back to Top](#qpso-pytorch-documentation)** | **[🏠 Main README](../../README.md)** | **[📦 ANN Module](../../ann/docs/index.md)**

</div>

# Documentacion QPSO-PyTorch

[🏠 README Principal](../../README_ES.md) | [📦 Modulo ANN](../../ann/docs/index_es.md) | **Algoritmos QPSO** | [🇬🇧 English](index.md)

---

> **Modulo**: QPSO-PyTorch/tensor_qpso/
> **Version**: 2.0.0

Esta documentacion cubre las implementaciones del algoritmo QPSO (Quantum Particle Swarm Optimization).

---

## Navegacion Rapida

### Documentacion de Implementaciones

| Documento | Descripcion | Nivel |
|-----------|-------------|-------|
| [📘 docs_qpso.md](docs_qpso.md) | Implementacion de referencia NumPy (basada en pypi) | Principiante |
| [📗 docs_qpso_tensor.md](docs_qpso_tensor.md) | Implementacion con tensores PyTorch | Intermedio |
| [📙 docs_qpso_tensor_optimized.md](docs_qpso_tensor_optimized.md) | Implementacion optimizada (17 mejoras) | Avanzado |

### Analisis Comparativo

| Documento | Descripcion |
|-----------|-------------|
| [📊 implementation_comparison_es.md](implementation_comparison_es.md) | **Comparacion detallada** de las tres implementaciones |
| [📊 implementation_comparison.md](implementation_comparison.md) | Comparacion en ingles |

---

## Resumen de Implementaciones

```
┌─────────────────────────────────────────────────────────────────┐
│                  Implementaciones QPSO                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   qpso.py    │───▶│qpso_tensor.py│───▶│qpso_tensor_      │  │
│  │              │    │              │    │  optimized.py    │  │
│  │  Referencia  │    │ Vectorizado  │    │  Produccion      │  │
│  │  NumPy       │    │  PyTorch     │    │  17 mejoras      │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│        ▲                    ▲                     ▲             │
│        │                    │                     │             │
│   Aprendizaje          Aceleracion          Produccion          │
│                          GPU                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Orden de Lectura Recomendado

### Para Principiantes

1. **Empezar aqui**: [docs_qpso.md](docs_qpso.md) - Entender los algoritmos QPSO/QDPSO basicos
2. **Luego**: [implementation_comparison_es.md](implementation_comparison_es.md) - Ver diferencias entre implementaciones

### Para Usuarios Intermedios

1. **Empezar aqui**: [docs_qpso_tensor.md](docs_qpso_tensor.md) - Aprender optimizacion basada en tensores
2. **Luego**: [docs_qpso_tensor_optimized.md](docs_qpso_tensor_optimized.md) - Caracteristicas avanzadas

### Para Uso en Produccion

1. **Ir directamente a**: [docs_qpso_tensor_optimized.md](docs_qpso_tensor_optimized.md) - Implementacion completa
2. **Referencia**: [implementation_comparison_es.md](implementation_comparison_es.md) - Benchmarks de rendimiento

---

## Enlaces Rapidos

### Scripts Ejecutables Principales

| Script | Descripcion |
|--------|-------------|
| [main_pypi.py](../main_pypi.py) | Ejemplo QPSO original (basado en pypi) |
| [main_qpso.py](../main_qpso.py) | Ejemplo wrapper QPSO |
| [main_qpso_tensor.py](../main_qpso_tensor.py) | Ejemplo QPSO con tensores |
| [main_qpso_tensor_optimized.py](../main_qpso_tensor_optimized.py) | Ejemplo tensores optimizados |

### Archivos Fuente

| Archivo | Descripcion |
|---------|-------------|
| [tensor_qpso/qpso.py](../tensor_qpso/qpso.py) | Implementacion NumPy |
| [tensor_qpso/qpso_tensor.py](../tensor_qpso/qpso_tensor.py) | Implementacion basica con tensores |
| [tensor_qpso/qpso_tensor_optimized.py](../tensor_qpso/qpso_tensor_optimized.py) | Implementacion optimizada |

---

## Documentacion Relacionada

### Modulo ANN (Entrenamiento de Redes Neuronales)

El modulo ANN utiliza QPSO para entrenar redes neuronales sin backpropagation:

- [📚 Indice Modulo ANN](../../ann/docs/index_es.md) - Documentacion principal
- [🧠 Modelos](../../ann/docs/models.md) - QPSOCompatibleANN
- [⚙️ Optimizadores](../../ann/docs/optimizers.md) - QPSONNOptimizer, estrategias de entrenamiento
- [🏋️ Trainers](../../ann/docs/trainers.md) - Interfaz de entrenamiento de alto nivel

---

## Ver Tambien

- [🏠 README Principal](../../README_ES.md) - Vision general del proyecto
- [🏠 Main README (EN)](../../README.md) - Vision general en ingles

---

<div align="center">

**[⬆️ Volver Arriba](#documentacion-qpso-pytorch)** | **[🏠 README Principal](../../README_ES.md)** | **[📦 Modulo ANN](../../ann/docs/index_es.md)**

</div>

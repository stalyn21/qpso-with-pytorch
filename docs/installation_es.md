# Guia de Instalacion

<div align="center">

**[English Version](installation.md)**

---

[Inicio](../README_ES.md) | **Instalacion** | [Docs QPSO-PyTorch](../QPSO-PyTorch/docs/index_es.md) | [Docs ANN](../ann/docs/index_es.md)

---

</div>

Esta guia proporciona instrucciones detalladas para configurar el entorno de PyTorch QPSO Suite.

---

## Tabla de Contenidos

- [Requisitos del Sistema](#requisitos-del-sistema)
- [Metodos de Instalacion](#metodos-de-instalacion)
  - [Metodo 1: Entorno Conda (Recomendado)](#metodo-1-entorno-conda-recomendado)
  - [Metodo 2: Instalacion con Pip](#metodo-2-instalacion-con-pip)
  - [Metodo 3: Instalacion Manual](#metodo-3-instalacion-manual)
- [Configuracion de GPU](#configuracion-de-gpu)
- [Verificacion](#verificacion)
- [Solucion de Problemas](#solucion-de-problemas)

---

## Requisitos del Sistema

### Requisitos Minimos

| Componente | Version |
|------------|---------|
| **Python** | >= 3.10 |
| **PyTorch** | >= 2.0.0 |
| **RAM** | 8 GB |
| **Espacio en Disco** | 5 GB |

### Requisitos Recomendados (GPU)

| Componente | Version |
|------------|---------|
| **Python** | 3.12 |
| **PyTorch** | 2.5.1 |
| **CUDA** | 12.4 |
| **cuDNN** | 9.1.0 |
| **GPU** | NVIDIA con >= 4GB VRAM |
| **RAM** | 16 GB |
| **Espacio en Disco** | 10 GB |

### Configuracion Probada

Este proyecto fue desarrollado y probado con la siguiente configuracion:

| Componente | Version | Fuente |
|------------|---------|--------|
| Python | 3.12.8 | conda-forge |
| PyTorch | 2.5.1 | canal pytorch |
| CUDA | 12.4 | canal nvidia |
| cuDNN | 9.1.0 | canal pytorch |
| NumPy | 2.2.1 | conda-forge |
| Plataforma | Linux x86_64 | - |

---

## Metodos de Instalacion

### Metodo 1: Entorno Conda (Recomendado)

Este es el metodo recomendado ya que maneja las dependencias de CUDA automaticamente.

#### Opcion A: Usando environment.yml

```bash
# Clonar el repositorio
git clone https://github.com/stalyn21/qpso-with-pytorch.git
cd pytorch-qpso-suite

# Crear entorno desde archivo
conda env create -f environment.yml

# Activar entorno
conda activate pytorch_qpso_gpu
```

#### Opcion B: Instalacion Paso a Paso

```bash
# Crear nuevo entorno
conda create -n pytorch_qpso_gpu python=3.12
conda activate pytorch_qpso_gpu

# Instalar PyTorch con soporte CUDA 12.4
conda install pytorch=2.5.* torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Instalar paquetes de computacion cientifica
conda install numpy pillow pyyaml -c conda-forge

# Instalar dependencias restantes via pip
pip install torch-pso>=1.2.0 \
            scipy>=1.15.0 \
            scikit-learn>=1.6.0 \
            pandas>=2.0.0 \
            matplotlib>=3.10.0 \
            seaborn>=0.13.0 \
            plotly>=6.0.0 \
            kaleido>=1.2.0 \
            opencv-python>=4.10.0 \
            mahotas>=1.4.0 \
            optuna>=4.0.0 \
            tqdm>=4.60.0 \
            pytest>=9.0.0 \
            pytest-timeout>=2.4.0
```

#### Actualizar Entorno Existente

```bash
conda env update -f environment.yml --prune
```

---

### Metodo 2: Instalacion con Pip

Para usuarios que prefieren pip o no tienen conda instalado.

#### Con Soporte GPU (CUDA 12.4)

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# o
.\venv\Scripts\activate  # Windows

# Instalar PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Instalar dependencias del proyecto
pip install -r requirements.txt
```

#### Solo CPU

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Instalar version CPU de PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar dependencias del proyecto
pip install -r requirements.txt
```

#### Otras Versiones de CUDA

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Metodo 3: Instalacion Manual

Para usuarios avanzados o configuraciones personalizadas.

#### Dependencias Principales

```bash
# PyTorch (elegir version apropiada para tu CUDA)
pip install torch>=2.5.0 torchvision>=0.20.0 torchaudio>=2.5.0

# PSO para Redes Neuronales
pip install torch-pso>=1.2.0

# Computacion Cientifica
pip install numpy>=2.0.0 scipy>=1.15.0 scikit-learn>=1.6.0 pandas>=2.0.0

# Visualizacion
pip install matplotlib>=3.10.0 seaborn>=0.13.0 plotly>=6.0.0 kaleido>=1.2.0

# Procesamiento de Imagenes
pip install opencv-python>=4.10.0 pillow>=11.0.0 mahotas>=1.4.0

# Optimizacion de Hiperparametros
pip install optuna>=4.0.0

# Utilidades
pip install tqdm>=4.60.0 pyyaml>=6.0.0

# Testing
pip install pytest>=9.0.0 pytest-timeout>=2.4.0
```

---

## Configuracion de GPU

### Requisitos del Driver NVIDIA

| Version CUDA | Driver Minimo |
|--------------|---------------|
| CUDA 12.4 | >= 550.54.14 |
| CUDA 12.1 | >= 530.30.02 |
| CUDA 11.8 | >= 520.61.05 |

### Verificar Driver NVIDIA

```bash
nvidia-smi
```

Salida esperada:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.XX.XX    Driver Version: 550.XX.XX    CUDA Version: 12.4    |
+-----------------------------------------------------------------------------+
```

### Verificar PyTorch CUDA

```python
import torch

print(f"Version PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Version CUDA: {torch.version.cuda}")
print(f"Version cuDNN: {torch.backends.cudnn.version()}")
print(f"Dispositivo GPU: {torch.cuda.get_device_name(0)}")
```

---

## Verificacion

### Verificacion Rapida

```bash
# Verificar PyTorch y CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Verificar modulo ann
python -c "from ann import QPSOCompatibleANN, Trainer; print('modulo ann: OK')"

# Verificar modulo QPSO-PyTorch
python -c "from QPSO_PyTorch.tensor_qpso import QPSOTensorOptimized; print('QPSO-PyTorch: OK')"
```

### Script de Verificacion Completa

```python
#!/usr/bin/env python
"""Verificar instalacion de QPSO-PyTorch."""

def verificar_instalacion():
    print("=" * 60)
    print("Verificacion de Instalacion QPSO-PyTorch")
    print("=" * 60)

    # Verificar PyTorch
    try:
        import torch
        print(f"\n[OK] PyTorch: {torch.__version__}")
        print(f"     CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     Version CUDA: {torch.version.cuda}")
            print(f"     Version cuDNN: {torch.backends.cudnn.version()}")
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"[FALLO] PyTorch: {e}")
        return False

    # Verificar dependencias principales
    deps = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly'),
        ('cv2', 'opencv-python'),
        ('PIL', 'pillow'),
        ('mahotas', 'mahotas'),
        ('optuna', 'optuna'),
        ('tqdm', 'tqdm'),
        ('torch_pso', 'torch-pso'),
    ]

    print("\n[Verificando Dependencias]")
    todo_ok = True
    for modulo, nombre in deps:
        try:
            __import__(modulo)
            print(f"  [OK] {nombre}")
        except ImportError:
            print(f"  [FALLO] {nombre}")
            todo_ok = False

    # Verificar modulos del proyecto
    print("\n[Verificando Modulos del Proyecto]")
    try:
        from ann import QPSOCompatibleANN, Trainer
        print("  [OK] modulo ann")
    except ImportError as e:
        print(f"  [FALLO] modulo ann: {e}")
        todo_ok = False

    print("\n" + "=" * 60)
    if todo_ok:
        print("Verificacion de instalacion: EXITOSA")
    else:
        print("Verificacion de instalacion: FALLIDA")
        print("Por favor revisa las dependencias faltantes arriba.")
    print("=" * 60)

    return todo_ok

if __name__ == "__main__":
    verificar_instalacion()
```

### Ejecutar Tests

```bash
# Ejecutar todos los tests
pytest

# Ejecutar con salida detallada
pytest -v

# Ejecutar archivo de test especifico
pytest ann/tests/test_trainer.py
```

---

## Solucion de Problemas

### Problemas Comunes

#### 1. CUDA No Disponible

**Problema**: `torch.cuda.is_available()` retorna `False`

**Soluciones**:
- Verificar que el driver NVIDIA este instalado: `nvidia-smi`
- Reinstalar PyTorch con la version correcta de CUDA
- Verificar compatibilidad del toolkit CUDA con el driver

```bash
# Reinstalar PyTorch con CUDA 12.4
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 2. Errores de Importacion

**Problema**: `ModuleNotFoundError: No module named 'ann'`

**Solucion**: Asegurate de ejecutar desde el directorio raiz del proyecto:
```bash
cd pytorch-qpso-suite
python -c "from ann import Trainer"
```

O agrega el proyecto al PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/ruta/a/pytorch-qpso-suite"
```

#### 3. Sin Memoria (GPU)

**Problema**: `CUDA out of memory`

**Soluciones**:
- Reducir `n_particles` en la configuracion de entrenamiento
- Reducir el tamano de batch
- Usar arquitectura de red mas pequena
- Limpiar cache de GPU:
```python
import torch
torch.cuda.empty_cache()
```

#### 4. Error de Importacion OpenCV

**Problema**: `ImportError: libGL.so.1: cannot open shared object file`

**Solucion** (Linux):
```bash
sudo apt-get install libgl1-mesa-glx
# o
conda install -c conda-forge libgl
```

#### 5. Conflictos en Entorno Conda

**Problema**: Conflictos de paquetes durante la instalacion

**Solucion**: Crear un entorno nuevo:
```bash
conda deactivate
conda env remove -n pytorch_qpso_gpu
conda env create -f environment.yml
```

### Obtener Ayuda

Si encuentras problemas no cubiertos aqui:

1. Revisa los [Issues en GitHub](https://github.com/stalyn21/qpso-with-pytorch/issues)
2. Abre un nuevo issue con:
   - Tu sistema operativo y version
   - Version de Python
   - Version de PyTorch
   - Version de CUDA (si aplica)
   - Traceback completo del error

---

## Siguientes Pasos

Despues de una instalacion exitosa:

1. Lee la [Guia de Inicio Rapido](../README_ES.md#inicio-rapido)
2. Explora la [Referencia de API](../README_ES.md#referencia-de-api)
3. Prueba los [Ejemplos de Uso](../ann/docs/usage_cases.md)
4. Descarga el [Dataset MCW](../README_ES.md#datasets) para ejemplos de clasificacion de imagenes

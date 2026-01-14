# Installation Guide

<div align="center">

**[Version en Espanol](installation_es.md)**

---

[Home](../README.md) | **Installation** | [QPSO-PyTorch Docs](../QPSO-PyTorch/docs/index.md) | [ANN Docs](../ann/docs/index.md)

---

</div>

This guide provides detailed instructions for setting up the PyTorch QPSO Suite environment.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
  - [Method 1: Conda Environment (Recommended)](#method-1-conda-environment-recommended)
  - [Method 2: Pip Installation](#method-2-pip-installation)
  - [Method 3: Manual Installation](#method-3-manual-installation)
- [GPU Configuration](#gpu-configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

| Component | Version |
|-----------|---------|
| **Python** | >= 3.10 |
| **PyTorch** | >= 2.0.0 |
| **RAM** | 8 GB |
| **Disk Space** | 5 GB |

### Recommended Requirements (GPU)

| Component | Version |
|-----------|---------|
| **Python** | 3.12 |
| **PyTorch** | 2.5.1 |
| **CUDA** | 12.4 |
| **cuDNN** | 9.1.0 |
| **GPU** | NVIDIA with >= 4GB VRAM |
| **RAM** | 16 GB |
| **Disk Space** | 10 GB |

### Tested Configuration

This project was developed and tested with the following configuration:

| Component | Version | Source |
|-----------|---------|--------|
| Python | 3.12.8 | conda-forge |
| PyTorch | 2.5.1 | pytorch channel |
| CUDA | 12.4 | nvidia channel |
| cuDNN | 9.1.0 | pytorch channel |
| NumPy | 2.2.1 | conda-forge |
| Platform | Linux x86_64 | - |

---

## Installation Methods

### Method 1: Conda Environment (Recommended)

This is the recommended method as it handles CUDA dependencies automatically.

#### Option A: Using environment.yml

```bash
# Clone the repository
git clone https://github.com/stalyn21/qpso-with-pytorch.git
cd pytorch-qpso-suite

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate pytorch_qpso_gpu
```

#### Option B: Step-by-Step Installation

```bash
# Create new environment
conda create -n pytorch_qpso_gpu python=3.12
conda activate pytorch_qpso_gpu

# Install PyTorch with CUDA 12.4 support
conda install pytorch=2.5.* torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install scientific computing packages
conda install numpy pillow pyyaml -c conda-forge

# Install remaining dependencies via pip
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

#### Update Existing Environment

```bash
conda env update -f environment.yml --prune
```

---

### Method 2: Pip Installation

For users who prefer pip or don't have conda installed.

#### With GPU Support (CUDA 12.4)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install -r requirements.txt
```

#### CPU Only

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
pip install -r requirements.txt
```

#### Other CUDA Versions

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Method 3: Manual Installation

For advanced users or custom configurations.

#### Core Dependencies

```bash
# PyTorch (choose appropriate version for your CUDA)
pip install torch>=2.5.0 torchvision>=0.20.0 torchaudio>=2.5.0

# PSO for Neural Networks
pip install torch-pso>=1.2.0

# Scientific Computing
pip install numpy>=2.0.0 scipy>=1.15.0 scikit-learn>=1.6.0 pandas>=2.0.0

# Visualization
pip install matplotlib>=3.10.0 seaborn>=0.13.0 plotly>=6.0.0 kaleido>=1.2.0

# Image Processing
pip install opencv-python>=4.10.0 pillow>=11.0.0 mahotas>=1.4.0

# Hyperparameter Optimization
pip install optuna>=4.0.0

# Utilities
pip install tqdm>=4.60.0 pyyaml>=6.0.0

# Testing
pip install pytest>=9.0.0 pytest-timeout>=2.4.0
```

---

## GPU Configuration

### NVIDIA Driver Requirements

| CUDA Version | Minimum Driver |
|--------------|----------------|
| CUDA 12.4 | >= 550.54.14 |
| CUDA 12.1 | >= 530.30.02 |
| CUDA 11.8 | >= 520.61.05 |

### Check NVIDIA Driver

```bash
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.XX.XX    Driver Version: 550.XX.XX    CUDA Version: 12.4    |
+-----------------------------------------------------------------------------+
```

### Verify PyTorch CUDA

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

---

## Verification

### Quick Verification

```bash
# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Verify ann module
python -c "from ann import QPSOCompatibleANN, Trainer; print('ann module: OK')"

# Verify QPSO-PyTorch module
python -c "from QPSO_PyTorch.tensor_qpso import QPSOTensorOptimized; print('QPSO-PyTorch: OK')"
```

### Full Verification Script

```python
#!/usr/bin/env python
"""Verify QPSO-PyTorch installation."""

def verify_installation():
    print("=" * 60)
    print("QPSO-PyTorch Installation Verification")
    print("=" * 60)

    # Check PyTorch
    try:
        import torch
        print(f"\n[OK] PyTorch: {torch.__version__}")
        print(f"     CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     CUDA version: {torch.version.cuda}")
            print(f"     cuDNN version: {torch.backends.cudnn.version()}")
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"[FAIL] PyTorch: {e}")
        return False

    # Check core dependencies
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

    print("\n[Checking Dependencies]")
    all_ok = True
    for module, name in deps:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name}")
            all_ok = False

    # Check project modules
    print("\n[Checking Project Modules]")
    try:
        from ann import QPSOCompatibleANN, Trainer
        print("  [OK] ann module")
    except ImportError as e:
        print(f"  [FAIL] ann module: {e}")
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("Installation verification: PASSED")
    else:
        print("Installation verification: FAILED")
        print("Please check the missing dependencies above.")
    print("=" * 60)

    return all_ok

if __name__ == "__main__":
    verify_installation()
```

### Run Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest ann/tests/test_trainer.py
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Not Available

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions**:
- Verify NVIDIA driver is installed: `nvidia-smi`
- Reinstall PyTorch with correct CUDA version
- Check CUDA toolkit compatibility with driver

```bash
# Reinstall PyTorch with CUDA 12.4
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 2. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'ann'`

**Solution**: Ensure you're running from the project root directory:
```bash
cd pytorch-qpso-suite
python -c "from ann import Trainer"
```

Or add the project to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/pytorch-qpso-suite"
```

#### 3. Out of Memory (GPU)

**Problem**: `CUDA out of memory`

**Solutions**:
- Reduce `n_particles` in training configuration
- Reduce batch size
- Use smaller network architecture
- Clear GPU cache:
```python
import torch
torch.cuda.empty_cache()
```

#### 4. OpenCV Import Error

**Problem**: `ImportError: libGL.so.1: cannot open shared object file`

**Solution** (Linux):
```bash
sudo apt-get install libgl1-mesa-glx
# or
conda install -c conda-forge libgl
```

#### 5. Conda Environment Conflicts

**Problem**: Package conflicts during installation

**Solution**: Create a fresh environment:
```bash
conda deactivate
conda env remove -n pytorch_qpso_gpu
conda env create -f environment.yml
```

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/stalyn21/qpso-with-pytorch/issues)
2. Open a new issue with:
   - Your OS and version
   - Python version
   - PyTorch version
   - CUDA version (if applicable)
   - Full error traceback

---

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](../README.md#quick-start)
2. Explore the [API Reference](../README.md#api-reference)
3. Try the [Usage Examples](../ann/docs/usage_cases.md)
4. Download the [MCW Dataset](../README.md#datasets) for image classification examples

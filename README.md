# QPSO with PyTorch

This repository implements [Quantum Particle Swarm Optimization](https://github.com/ngroup/qpso) (QPSO) for training Artificial Neural Networks using PyTorch tensors. The implementation modifies the original SciPy QPSO algorithm to leverage PyTorch's capabilities.

## Features

- [ANN PSO implementation using Torch PSO]()
- [QPSO implementation using PyTorch tensors](./QPSO-PyTorch/implementation_analysis.md)
- Single-swarm ANN implementation
- Multi-swarm ANN implementation
- GPU acceleration compatibility

## Prerequisites

Before getting started, ensure you have the following installed:

1. [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) - Required for GPU acceleration
2. [PyTorch](https://pytorch.org/get-started/locally/) - Deep learning framework

### CUDA Installation

#### For Arch Linux Users:
```bash
sudo pacman -S cuda
```

Verify CUDA installation:
```bash
nvcc --version
```

If `nvcc` command is not available, check if it exists:
```bash
ls /opt/cuda/bin/nvcc
```

Add CUDA to your PATH by adding these lines to `~/.bashrc`:
```bash
# Cuda Toolkit path
if [[ ":$PATH:" != *":/opt/cuda/bin:"* ]]; then
    export PATH=/opt/cuda/bin:$PATH
fi
```

Then run:
```bash
source ~/.bashrc
```

## Environment Setup

1. Remove existing environment (if necessary):
```bash
conda remove --name pytorch_qpso_gpu --all
```

2. Create and configure new environment:
```bash
conda create --name pytorch_qpso_gpu python=3.12 pip
conda activate pytorch_qpso_gpu
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -U scikit-learn
```

3. Verify installation:
```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
```

Expected output:
```
2.5.1
True
```

## Usage

[Add usage instructions and examples here]

## Contributing

[Add contribution guidelines here]

## License

[Add license information here]

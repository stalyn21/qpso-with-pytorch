"""
Ejemplos de uso: QPSO y QDPSO con Tensores PyTorch

Este archivo demuestra el uso de las implementaciones con tensores:
- QPSOTensor: QPSO original del paper con operaciones vectorizadas
- QDPSOTensor: Variante QDPSO con operaciones vectorizadas

Caracteristicas:
- Soporte para GPU (CUDA) y CPU
- Operaciones vectorizadas para mejor rendimiento
- Comparacion de rendimiento CPU vs GPU

Requisitos:
    conda activate pytorch_qpso_gpu
"""

import torch
import time
from tensor_qpso.qpso_tensor import QPSOTensor, QDPSOTensor, get_device


# =============================================================================
# Funciones de prueba (benchmark) - Version Tensor
# =============================================================================

def sphere_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Funcion Esfera (vectorizada): f(x) = sum(x_i^2)
    Minimo global: f(0, 0, ..., 0) = 0

    Args:
        x: Tensor de forma (n_particles, dim) o (dim,)

    Returns:
        Tensor de forma (n_particles,) o escalar
    """
    if x.dim() == 1:
        return torch.sum(x ** 2)
    return torch.sum(x ** 2, dim=1)


def rastrigin_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Funcion Rastrigin (vectorizada): f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Minimo global: f(0, 0, ..., 0) = 0

    Args:
        x: Tensor de forma (n_particles, dim) o (dim,)

    Returns:
        Tensor de forma (n_particles,) o escalar
    """
    if x.dim() == 1:
        n = x.shape[0]
        return 10 * n + torch.sum(x ** 2 - 10 * torch.cos(2 * torch.pi * x))
    n = x.shape[1]
    return 10 * n + torch.sum(x ** 2 - 10 * torch.cos(2 * torch.pi * x), dim=1)


def rosenbrock_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Funcion Rosenbrock (vectorizada): f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2)
    Minimo global: f(1, 1, ..., 1) = 0

    Args:
        x: Tensor de forma (n_particles, dim) o (dim,)

    Returns:
        Tensor de forma (n_particles,) o escalar
    """
    if x.dim() == 1:
        return torch.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    return torch.sum(
        100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2,
        dim=1
    )


# =============================================================================
# Funcion de logging
# =============================================================================

def log(optimizer, algorithm_name: str = ""):
    """Muestra el progreso de la optimizacion."""
    pbest_values = optimizer.pbest_values

    if optimizer.iters == 0:
        print(f"\n{'='*65}")
        print(f" {algorithm_name}")
        print(f" Device: {optimizer.device}")
        print(f"{'='*65}")
        print("{0: >6}  {1: >12}  {2: >12}  {3: >12}".format(
            "Iter", "Best", "Mean", "STD"))
        print("-" * 65)

    print("{0: >6}  {1: >12.6E}  {2: >12.6E}  {3: >12.6E}".format(
        optimizer.iters,
        optimizer.gbest_value,
        pbest_values.mean().item(),
        pbest_values.std().item()
    ))


# =============================================================================
# Configuracion
# =============================================================================

print("\n" + "="*70)
print(" QPSO/QDPSO con Tensores PyTorch")
print("="*70)

# Detectar dispositivos disponibles
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Configuracion comun
NParticle = 40
MaxIters = 1000
NDim = 10
bounds = [(-5.12, 5.12) for _ in range(NDim)]


# =============================================================================
# Ejemplo 1: QPSOTensor en CPU
# =============================================================================

print("\n" + "="*70)
print(" QPSOTensor - QPSO Original con Tensores (CPU)")
print("="*70)

start_time = time.time()
qpso_cpu = QPSOTensor(
    sphere_tensor, NParticle, NDim, bounds, MaxIters,
    alpha=(1.0, 0.5),
    device='cpu'
)
qpso_cpu.update(
    callback=lambda s: log(s, "QPSOTensor (CPU)"),
    interval=200
)
cpu_time_qpso = time.time() - start_time

print(f"\nResultado QPSOTensor (CPU):")
print(f"  Mejor valor: {qpso_cpu.gbest_value:.6E}")
print(f"  Tiempo: {cpu_time_qpso:.3f} segundos")


# =============================================================================
# Ejemplo 2: QDPSOTensor en CPU
# =============================================================================

print("\n" + "="*70)
print(" QDPSOTensor - QDPSO Variante con Tensores (CPU)")
print("="*70)

start_time = time.time()
qdpso_cpu = QDPSOTensor(
    sphere_tensor, NParticle, NDim, bounds, MaxIters,
    g=0.96,
    device='cpu'
)
qdpso_cpu.update(
    callback=lambda s: log(s, "QDPSOTensor (CPU)"),
    interval=200
)
cpu_time_qdpso = time.time() - start_time

print(f"\nResultado QDPSOTensor (CPU):")
print(f"  Mejor valor: {qdpso_cpu.gbest_value:.6E}")
print(f"  Tiempo: {cpu_time_qdpso:.3f} segundos")


# =============================================================================
# Ejemplo 3: GPU (si esta disponible)
# =============================================================================

if torch.cuda.is_available():
    print("\n" + "="*70)
    print(" QPSOTensor - QPSO Original con Tensores (GPU/CUDA)")
    print("="*70)

    # Warmup GPU
    _ = torch.rand(100, 100, device='cuda')

    start_time = time.time()
    qpso_gpu = QPSOTensor(
        sphere_tensor, NParticle, NDim, bounds, MaxIters,
        alpha=(1.0, 0.5),
        device='cuda'
    )
    qpso_gpu.update(
        callback=lambda s: log(s, "QPSOTensor (GPU)"),
        interval=200
    )
    torch.cuda.synchronize()  # Asegurar que GPU termino
    gpu_time_qpso = time.time() - start_time

    print(f"\nResultado QPSOTensor (GPU):")
    print(f"  Mejor valor: {qpso_gpu.gbest_value:.6E}")
    print(f"  Tiempo: {gpu_time_qpso:.3f} segundos")

    print("\n" + "="*70)
    print(" QDPSOTensor - QDPSO Variante con Tensores (GPU/CUDA)")
    print("="*70)

    start_time = time.time()
    qdpso_gpu = QDPSOTensor(
        sphere_tensor, NParticle, NDim, bounds, MaxIters,
        g=0.96,
        device='cuda'
    )
    qdpso_gpu.update(
        callback=lambda s: log(s, "QDPSOTensor (GPU)"),
        interval=200
    )
    torch.cuda.synchronize()
    gpu_time_qdpso = time.time() - start_time

    print(f"\nResultado QDPSOTensor (GPU):")
    print(f"  Mejor valor: {qdpso_gpu.gbest_value:.6E}")
    print(f"  Tiempo: {gpu_time_qdpso:.3f} segundos")


# =============================================================================
# Ejemplo 4: Problema de alta dimensionalidad (GPU vs CPU)
# =============================================================================

print("\n" + "="*70)
print(" COMPARACION: Alta Dimensionalidad (dim=100, particles=100)")
print("="*70)

NDim_large = 100
NParticle_large = 100
MaxIters_large = 500
bounds_large = [(-5.12, 5.12) for _ in range(NDim_large)]

# CPU
print("\nEjecutando en CPU...")
start_time = time.time()
qpso_large_cpu = QPSOTensor(
    sphere_tensor, NParticle_large, NDim_large, bounds_large, MaxIters_large,
    alpha=0.75,
    device='cpu'
)
qpso_large_cpu.update()
cpu_time_large = time.time() - start_time
print(f"  CPU - Mejor valor: {qpso_large_cpu.gbest_value:.6E}")
print(f"  CPU - Tiempo: {cpu_time_large:.3f} segundos")

# GPU (si disponible)
if torch.cuda.is_available():
    print("\nEjecutando en GPU...")
    start_time = time.time()
    qpso_large_gpu = QPSOTensor(
        sphere_tensor, NParticle_large, NDim_large, bounds_large, MaxIters_large,
        alpha=0.75,
        device='cuda'
    )
    qpso_large_gpu.update()
    torch.cuda.synchronize()
    gpu_time_large = time.time() - start_time
    print(f"  GPU - Mejor valor: {qpso_large_gpu.gbest_value:.6E}")
    print(f"  GPU - Tiempo: {gpu_time_large:.3f} segundos")
    print(f"\n  Speedup GPU vs CPU: {cpu_time_large / gpu_time_large:.2f}x")


# =============================================================================
# Resumen
# =============================================================================

print("\n" + "="*70)
print(" RESUMEN DE RESULTADOS")
print("="*70)
print(f"{'Algoritmo':<30} {'Device':<8} {'Mejor Valor':>15} {'Tiempo':>10}")
print("-"*70)
print(f"{'QPSOTensor (alpha=1.0->0.5)':<30} {'CPU':<8} {qpso_cpu.gbest_value:>15.6E} {cpu_time_qpso:>9.3f}s")
print(f"{'QDPSOTensor (g=0.96)':<30} {'CPU':<8} {qdpso_cpu.gbest_value:>15.6E} {cpu_time_qdpso:>9.3f}s")

if torch.cuda.is_available():
    print(f"{'QPSOTensor (alpha=1.0->0.5)':<30} {'GPU':<8} {qpso_gpu.gbest_value:>15.6E} {gpu_time_qpso:>9.3f}s")
    print(f"{'QDPSOTensor (g=0.96)':<30} {'GPU':<8} {qdpso_gpu.gbest_value:>15.6E} {gpu_time_qdpso:>9.3f}s")

print("="*70)


# =============================================================================
# Ejemplo de uso con device='auto'
# =============================================================================

print("\n" + "="*70)
print(" USO CON device='auto' (seleccion automatica)")
print("="*70)

auto_device = get_device('auto')
print(f"Device seleccionado automaticamente: {auto_device}")

qpso_auto = QPSOTensor(
    sphere_tensor, 40, 10,
    [(-5.12, 5.12) for _ in range(10)],
    500,
    alpha=0.75,
    device='auto'
)
qpso_auto.update()
print(f"Resultado: {qpso_auto.gbest_value:.6E}")
print(f"Ejecutado en: {qpso_auto.device}")

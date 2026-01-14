"""
Ejemplos de Uso: QPSO y QDPSO Optimizado con Tensores PyTorch

Este archivo demuestra todas las caracteristicas de la version optimizada:

1. Uso basico
2. Configuracion de boundary handling
3. Convergencia temprana
4. Historial y graficas
5. Sistema de callbacks
6. Maximizacion
7. Reproducibilidad con semilla
8. Context manager
9. Comparativa de rendimiento
10. Factory function

Requisitos:
    conda activate pytorch_qpso_gpu
    pip install matplotlib  # Para graficas (opcional)
"""

import torch
import time
from typing import Callable

from tensor_qpso.qpso_tensor_optimized import (
    QPSOTensorOptimized,
    QDPSOTensorOptimized,
    OptimizationResult,
    CallbackEvent,
    BoundaryStrategy,
    create_optimizer,
    get_device
)


# =============================================================================
# Funciones de Prueba (Benchmark) - Vectorizadas
# =============================================================================

def sphere(x: torch.Tensor) -> torch.Tensor:
    """
    Funcion Esfera: f(x) = sum(x_i^2)
    Minimo global: f(0, 0, ..., 0) = 0
    """
    if x.dim() == 1:
        return torch.sum(x ** 2)
    return torch.sum(x ** 2, dim=1)


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """
    Funcion Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Minimo global: f(0, 0, ..., 0) = 0
    Altamente multimodal
    """
    if x.dim() == 1:
        n = x.shape[0]
        return 10 * n + torch.sum(x ** 2 - 10 * torch.cos(2 * torch.pi * x))
    n = x.shape[1]
    return 10 * n + torch.sum(x ** 2 - 10 * torch.cos(2 * torch.pi * x), dim=1)


def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    """
    Funcion Rosenbrock: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2)
    Minimo global: f(1, 1, ..., 1) = 0
    Valle estrecho curvo
    """
    if x.dim() == 1:
        return torch.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    return torch.sum(
        100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2,
        dim=1
    )


def negative_sphere(x: torch.Tensor) -> torch.Tensor:
    """
    Esfera negativa para probar maximizacion.
    Maximo global: f(0, 0, ..., 0) = 0
    """
    return -sphere(x)


# =============================================================================
# Utilidades
# =============================================================================

def print_header(title: str):
    """Imprime un encabezado formateado."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result(result: OptimizationResult, name: str = ""):
    """Imprime el resultado de una optimizacion."""
    print(f"\n{name} Resultado:")
    print(f"  Mejor valor:    {result.best_value:.10E}")
    print(f"  Iteraciones:    {result.iterations}")
    print(f"  Convergio:      {result.converged}")
    print(f"  Razon:          {result.convergence_reason}")
    print(f"  Tiempo:         {result.elapsed_time:.3f}s")
    print(f"  Dispositivo:    {result.device}")


# =============================================================================
# Informacion del Sistema
# =============================================================================

print_header("QPSO/QDPSO Optimizado - Ejemplos de Uso")

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

device = get_device('auto')
print(f"Dispositivo seleccionado: {device}")


# =============================================================================
# 1. Uso Basico
# =============================================================================

print_header("1. USO BASICO")

optimizer = QPSOTensorOptimized(
    cf=sphere,
    size=40,
    dim=10,
    bounds=[(-5.12, 5.12)] * 10,
    maxIters=1000
)

result = optimizer.optimize()
print_result(result, "QPSO Basico")


# =============================================================================
# 2. Configuracion de Boundary Handling
# =============================================================================

print_header("2. BOUNDARY HANDLING")

strategies = ['none', 'clamp', 'reflect', 'wrap', 'random']

for strategy in strategies:
    opt = QPSOTensorOptimized(
        cf=sphere,
        size=40,
        dim=10,
        bounds=[(-5.12, 5.12)] * 10,
        maxIters=500,
        boundary_strategy=strategy,
        seed=42  # Misma semilla para comparar
    )
    result = opt.optimize()
    print(f"  {strategy:8s}: {result.best_value:.6E} ({result.iterations} iters)")


# =============================================================================
# 3. Convergencia Temprana
# =============================================================================

print_header("3. CONVERGENCIA TEMPRANA")

# Sin convergencia temprana (patience muy alto)
opt_no_early = QPSOTensorOptimized(
    cf=sphere,
    size=40,
    dim=10,
    bounds=[(-5.12, 5.12)] * 10,
    maxIters=2000,
    patience=10000,  # Nunca convergera temprano
    seed=42
)
result_no_early = opt_no_early.optimize()
print(f"\nSin convergencia temprana:")
print(f"  Iteraciones: {result_no_early.iterations}")
print(f"  Razon: {result_no_early.convergence_reason}")

# Con convergencia temprana
opt_early = QPSOTensorOptimized(
    cf=sphere,
    size=40,
    dim=10,
    bounds=[(-5.12, 5.12)] * 10,
    maxIters=2000,
    tol=1e-15,
    patience=50,
    seed=42
)
result_early = opt_early.optimize()
print(f"\nCon convergencia temprana (tol=1e-15, patience=50):")
print(f"  Iteraciones: {result_early.iterations}")
print(f"  Razon: {result_early.convergence_reason}")
print(f"  Ahorro: {2000 - result_early.iterations} iteraciones")


# =============================================================================
# 4. Historial y Graficas
# =============================================================================

print_header("4. HISTORIAL DE OPTIMIZACION")

opt_history = QPSOTensorOptimized(
    cf=rastrigin,
    size=50,
    dim=10,
    bounds=[(-5.12, 5.12)] * 10,
    maxIters=1000,
    alpha=(1.0, 0.5),
    track_history=True,
    seed=42
)
result_history = opt_history.optimize()

print(f"\nHistorial disponible:")
print(f"  gbest_value:   {len(result_history.history['gbest_value'])} puntos")
print(f"  mean_fitness:  {len(result_history.history['mean_fitness'])} puntos")
print(f"  diversity:     {len(result_history.history['diversity'])} puntos")

# Intentar graficar si matplotlib esta disponible
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Convergencia
    axes[0].plot(result_history.history['gbest_value'])
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Iteracion')
    axes[0].set_ylabel('Mejor Valor')
    axes[0].set_title('Curva de Convergencia')
    axes[0].grid(True, alpha=0.3)

    # Diversidad
    axes[1].plot(result_history.history['diversity'])
    axes[1].set_xlabel('Iteracion')
    axes[1].set_ylabel('Diversidad')
    axes[1].set_title('Diversidad del Enjambre')
    axes[1].grid(True, alpha=0.3)

    # Desviacion estandar
    axes[2].plot(result_history.history['std_fitness'])
    axes[2].set_yscale('log')
    axes[2].set_xlabel('Iteracion')
    axes[2].set_ylabel('STD Fitness')
    axes[2].set_title('Variabilidad del Enjambre')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergencia_optimized.png', dpi=150)
    print("\nGrafica guardada en: convergencia_optimized.png")
    plt.close()

except ImportError:
    print("\nmatplotlib no disponible, omitiendo graficas")


# =============================================================================
# 5. Sistema de Callbacks
# =============================================================================

print_header("5. SISTEMA DE CALLBACKS")

# Contadores para callbacks
callback_counts = {'new_best': 0, 'iterations': 0}
best_improvements = []

def on_new_best(opt):
    callback_counts['new_best'] += 1
    best_improvements.append((opt.iters, opt.gbest_value))

def on_iteration_end(opt):
    callback_counts['iterations'] += 1

def on_convergence(opt):
    print(f"  [CALLBACK] Convergencia detectada en iteracion {opt.iters}")

def on_finish(opt):
    print(f"  [CALLBACK] Optimizacion finalizada")

opt_callbacks = QPSOTensorOptimized(
    cf=sphere,
    size=40,
    dim=10,
    bounds=[(-5.12, 5.12)] * 10,
    maxIters=500,
    tol=1e-20,
    patience=30,
    seed=42
)

# Registrar callbacks
opt_callbacks.callbacks.register(CallbackEvent.ON_NEW_BEST, on_new_best)
opt_callbacks.callbacks.register(CallbackEvent.ON_ITERATION_END, on_iteration_end)
opt_callbacks.callbacks.register(CallbackEvent.ON_CONVERGENCE, on_convergence)
opt_callbacks.callbacks.register(CallbackEvent.ON_FINISH, on_finish)

result_callbacks = opt_callbacks.optimize()

print(f"\nEstadisticas de callbacks:")
print(f"  Veces que mejoro gbest: {callback_counts['new_best']}")
print(f"  Iteraciones ejecutadas: {callback_counts['iterations']}")
print(f"  Primeras 5 mejoras: {best_improvements[:5]}")


# =============================================================================
# 6. Maximizacion
# =============================================================================

print_header("6. MAXIMIZACION")

# Minimizacion (default)
opt_min = QPSOTensorOptimized(
    cf=negative_sphere,  # -x^2, minimo en x=inf
    size=40,
    dim=10,
    bounds=[(-5.12, 5.12)] * 10,
    maxIters=500,
    minimize=True,
    seed=42
)
result_min = opt_min.optimize()
print(f"\nMinimizando -sphere (busca valores mas negativos):")
print(f"  Mejor valor: {result_min.best_value:.6E}")

# Maximizacion
opt_max = QPSOTensorOptimized(
    cf=negative_sphere,  # -x^2, maximo en x=0
    size=40,
    dim=10,
    bounds=[(-5.12, 5.12)] * 10,
    maxIters=500,
    minimize=False,  # MAXIMIZAR
    seed=42
)
result_max = opt_max.optimize()
print(f"\nMaximizando -sphere (busca x=0):")
print(f"  Mejor valor: {result_max.best_value:.6E} (debe ser ~0)")


# =============================================================================
# 7. Reproducibilidad con Semilla
# =============================================================================

print_header("7. REPRODUCIBILIDAD")

results_seed = []
for i in range(3):
    opt = QPSOTensorOptimized(
        cf=sphere,
        size=40,
        dim=10,
        bounds=[(-5.12, 5.12)] * 10,
        maxIters=500,
        seed=12345  # Misma semilla
    )
    result = opt.optimize()
    results_seed.append(result.best_value)
    print(f"  Ejecucion {i+1}: {result.best_value:.15E}")

print(f"\n  Todas iguales: {results_seed[0] == results_seed[1] == results_seed[2]}")


# =============================================================================
# 8. Context Manager
# =============================================================================

print_header("8. CONTEXT MANAGER")

print("\nUsando context manager para liberacion automatica de memoria:")

with QPSOTensorOptimized(
    cf=sphere,
    size=40,
    dim=10,
    bounds=[(-5.12, 5.12)] * 10,
    maxIters=500,
    device='auto'
) as opt:
    result = opt.optimize()
    print(f"  Dentro del context: {result.best_value:.6E}")

print("  Fuera del context: memoria liberada automaticamente")


# =============================================================================
# 9. Comparativa de Rendimiento
# =============================================================================

print_header("9. COMPARATIVA DE RENDIMIENTO")

# Configuracion para benchmark
config = {
    'size': 100,
    'dim': 50,
    'bounds': [(-5.12, 5.12)] * 50,
    'maxIters': 500,
    'seed': 42
}

# Importar version basica para comparar
try:
    from tensor_qpso.qpso_tensor import QPSOTensor

    # Version basica
    start = time.time()
    opt_basic = QPSOTensor(
        sphere, config['size'], config['dim'],
        config['bounds'], config['maxIters'],
        alpha=0.75, device='cpu'
    )
    opt_basic.update()
    time_basic = time.time() - start

    # Version optimizada
    start = time.time()
    opt_optimized = QPSOTensorOptimized(
        sphere, config['size'], config['dim'],
        config['bounds'], config['maxIters'],
        alpha=0.75, device='cpu', seed=config['seed']
    )
    result_optimized = opt_optimized.optimize()
    time_optimized = result_optimized.elapsed_time

    print(f"\nComparativa CPU (size={config['size']}, dim={config['dim']}):")
    print(f"  Version basica:     {time_basic:.3f}s, valor={opt_basic.gbest_value:.6E}")
    print(f"  Version optimizada: {time_optimized:.3f}s, valor={result_optimized.best_value:.6E}")
    print(f"  Speedup: {time_basic/time_optimized:.2f}x")

except ImportError:
    print("qpso_tensor.py no disponible para comparacion")


# =============================================================================
# 10. Factory Function
# =============================================================================

print_header("10. FACTORY FUNCTION")

algorithms = ['qpso', 'qdpso']
params = {
    'qpso': {'alpha': 0.75},
    'qdpso': {'g': 0.96}
}

for algo in algorithms:
    opt = create_optimizer(
        algorithm=algo,
        cf=sphere,
        size=40,
        dim=10,
        bounds=[(-5.12, 5.12)] * 10,
        maxIters=500,
        seed=42,
        **params[algo]
    )
    result = opt.optimize()
    print(f"  {algo.upper()}: {result.best_value:.6E}")


# =============================================================================
# 11. Ejemplo Completo: Problema Real
# =============================================================================

print_header("11. EJEMPLO COMPLETO - RASTRIGIN 20D")

print("\nOptimizando Rastrigin en 20 dimensiones (altamente multimodal)...")

with QPSOTensorOptimized(
    cf=rastrigin,
    size=100,                       # Mas particulas para problema dificil
    dim=20,
    bounds=[(-5.12, 5.12)] * 20,
    maxIters=3000,
    alpha=(1.0, 0.5),               # Alpha decreciente
    device='auto',
    dtype=torch.float32,
    seed=42,
    boundary_strategy='reflect',    # Rebotar en limites
    tol=1e-10,
    patience=200,
    track_history=True,
    minimize=True
) as optimizer:

    # Callback de progreso
    def progress(opt):
        if opt.iters % 500 == 0:
            print(f"    Iter {opt.iters:4d}: best = {opt.gbest_value:.6E}")

    result = optimizer.optimize(callback=progress, interval=500)

print(f"\n{result}")

# Resumen
print("\n" + "=" * 70)
print(" RESUMEN DE TODOS LOS EJEMPLOS COMPLETADOS")
print("=" * 70)
print("\nTodas las caracteristicas demostradas:")
print("  1. Uso basico con optimize()")
print("  2. Boundary handling (none, clamp, reflect, wrap, random)")
print("  3. Convergencia temprana con tol y patience")
print("  4. Historial de optimizacion con graficas")
print("  5. Sistema de callbacks por eventos")
print("  6. Soporte para maximizacion")
print("  7. Reproducibilidad con semilla")
print("  8. Context manager para recursos")
print("  9. Comparativa de rendimiento")
print("  10. Factory function")
print("  11. Ejemplo completo de problema real")
print("\nVer docs/docs_qpso_tensor_optimized.md para documentacion completa")

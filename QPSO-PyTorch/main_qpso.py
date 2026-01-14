"""
Ejemplos de uso: QPSO vs QDPSO

Este archivo demuestra el uso de ambas implementaciones:

1. QPSO (Original del Paper - Sun et al., 2004)
   - Usa mbest (mean best position) para calcular L
   - Punto atractor: c = phi * pbest + (1-phi) * gbest
   - Formula: L = alpha * |mbest - x|
   - Parametro: alpha (tipico: 0.75 o decreciente de 1.0 a 0.5)

2. QDPSO (Variante Delta - Implementacion PyPI original)
   - Usa distancia al punto atractor para calcular L
   - Punto atractor: c = (u1*pbest + u2*gbest) / (u1+u2)
   - Formula: L = (1/g) * |x - c|
   - Parametro: g (tipico: 0.96)
"""

import numpy as np
from tensor_qpso.qpso import QPSO, QDPSO


# =============================================================================
# Funciones de prueba (benchmark)
# =============================================================================

def sphere(args):
    """
    Funcion Esfera: f(x) = sum(x_i^2)
    Minimo global: f(0, 0, ..., 0) = 0
    Rango tipico: [-5.12, 5.12]
    """
    return sum([np.power(x, 2.) for x in args])


def rastrigin(args):
    """
    Funcion Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Minimo global: f(0, 0, ..., 0) = 0
    Rango tipico: [-5.12, 5.12]
    Caracteristica: Altamente multimodal
    """
    n = len(args)
    return 10 * n + sum([x**2 - 10 * np.cos(2 * np.pi * x) for x in args])


# =============================================================================
# Funcion de logging
# =============================================================================

def log(s, algorithm_name=""):
    """Muestra el progreso de la optimizacion."""
    best_values = [p.best_value for p in s.particles()]
    best_value_avg = np.mean(best_values)
    best_value_std = np.std(best_values)

    if s.iters == 0:
        print(f"\n{'='*60}")
        print(f" {algorithm_name}")
        print(f"{'='*60}")
        print("{0: >6}  {1: >12}  {2: >12}  {3: >12}".format(
            "Iter", "Best", "Mean", "STD"))
        print("-" * 60)

    print("{0: >6}  {1: >12.6E}  {2: >12.6E}  {3: >12.6E}".format(
        s.iters, s.gbest_value, best_value_avg, best_value_std))


# =============================================================================
# Configuracion comun
# =============================================================================

NParticle = 40          # Numero de particulas
MaxIters = 1000         # Iteraciones maximas
NDim = 10               # Dimensiones del problema
bounds = [(-5.12, 5.12) for _ in range(NDim)]  # Limites por dimension


# =============================================================================
# Ejemplo 1: QPSO Original (Paper de Sun et al., 2004)
# =============================================================================

print("\n" + "="*70)
print(" QPSO - Quantum Particle Swarm Optimization (Original Paper)")
print(" Usa: mbest (mean best), alpha, combinacion convexa")
print("="*70)

# Opcion A: alpha fijo
alpha_fijo = 0.75
qpso_fijo = QPSO(sphere, NParticle, NDim, bounds, MaxIters, alpha=alpha_fijo)
qpso_fijo.update(
    callback=lambda s: log(s, f"QPSO con alpha fijo = {alpha_fijo}"),
    interval=200
)
print(f"\nResultado QPSO (alpha={alpha_fijo}):")
print(f"  Mejor valor: {qpso_fijo.gbest_value:.6E}")
print(f"  Mejor posicion: [{', '.join([f'{x:.4f}' for x in qpso_fijo.gbest[:3]])}...]")

# Opcion B: alpha con decrecimiento lineal (recomendado en el paper)
print("\n" + "-"*70)
alpha_decrece = (1.0, 0.5)  # Decrece de 1.0 a 0.5
qpso_decrece = QPSO(sphere, NParticle, NDim, bounds, MaxIters, alpha=alpha_decrece)
qpso_decrece.update(
    callback=lambda s: log(s, f"QPSO con alpha decreciente {alpha_decrece}"),
    interval=200
)
print(f"\nResultado QPSO (alpha={alpha_decrece[0]}->{alpha_decrece[1]}):")
print(f"  Mejor valor: {qpso_decrece.gbest_value:.6E}")
print(f"  Mejor posicion: [{', '.join([f'{x:.4f}' for x in qpso_decrece.gbest[:3]])}...]")


# =============================================================================
# Ejemplo 2: QDPSO (Variante Delta - Implementacion PyPI)
# =============================================================================

print("\n" + "="*70)
print(" QDPSO - Quantum Delta PSO (Variante)")
print(" Usa: |x-c|, g, promedio ponderado estocastico")
print("="*70)

g = 0.96
qdpso = QDPSO(sphere, NParticle, NDim, bounds, MaxIters, g=g)
qdpso.update(
    callback=lambda s: log(s, f"QDPSO con g = {g}"),
    interval=200
)
print(f"\nResultado QDPSO (g={g}):")
print(f"  Mejor valor: {qdpso.gbest_value:.6E}")
print(f"  Mejor posicion: [{', '.join([f'{x:.4f}' for x in qdpso.gbest[:3]])}...]")


# =============================================================================
# Resumen comparativo
# =============================================================================

print("\n" + "="*70)
print(" RESUMEN COMPARATIVO")
print("="*70)
print(f"{'Algoritmo':<35} {'Mejor Valor':>15}")
print("-"*70)
print(f"{'QPSO (alpha fijo = 0.75)':<35} {qpso_fijo.gbest_value:>15.6E}")
print(f"{'QPSO (alpha 1.0 -> 0.5)':<35} {qpso_decrece.gbest_value:>15.6E}")
print(f"{'QDPSO (g = 0.96)':<35} {qdpso.gbest_value:>15.6E}")
print("="*70)
print("\nNota: Los resultados varian debido a la naturaleza estocastica.")
print("      Ejecute multiples veces para comparar rendimiento promedio.")

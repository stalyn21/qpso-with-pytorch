import numpy as np
import random


class Particle(object):
    def __init__(self, bounds):
        self._x = np.zeros(len(bounds))
        for idx, (lo, hi) in enumerate(bounds):
            self._x[idx] = random.uniform(lo, hi)

        self._best = self._x.copy()
        self._best_value = np.nan

    def __str__(self):
        return str(self._x)

    @property
    def best(self):
        return self._best

    def set_best(self, x):
        for i, v in enumerate(x):
            self._best[i] = v

    @property
    def best_value(self):
        return self._best_value

    def set_best_value(self, v):
        self._best_value = v

    def __getitem__(self, key):
        return self._x[key]

    def __setitem__(self, key, val):
        self._x[key] = val


class Swarm(object):
    def __init__(self, size, dim, bounds):
        self._particles = [Particle(bounds) for i in range(0, size)]
        self._dim = dim
        self._gbest_value = None
        self._gbest = None

    def size(self):
        return len(self._particles)

    def particles(self):
        return self._particles

    def mean_best(self):
        x = np.zeros(self._dim)
        for p in self._particles:
            x += p.best

        x /= self.size()

        return x

    @property
    def gbest(self):
        return self._gbest

    @property
    def gbest_value(self):
        return self._gbest_value

    def update_gbest(self):
        get_best_value = lambda x: x.best_value
        pg = min(self.particles(), key=get_best_value)
        if (not self._gbest_value) or self._gbest_value > pg.best_value:
            self._gbest = pg.best.copy()
            self._gbest_value = pg.best_value


class QPSOBase(Swarm):
    """
    Clase base para algoritmos QPSO.
    Contiene la logica comun de inicializacion, evaluacion y actualizacion.
    Las subclases deben implementar kernel_update() con su formula especifica.
    """
    def __init__(self, cf, size, dim, bounds, maxIters):
        super(QPSOBase, self).__init__(size, dim, bounds)
        self._cf = cf
        self._maxIters = maxIters
        self._iters = 0
        self.init_eval()

    def init_eval(self):
        for p in self.particles():
            pc = self._cf(p[:])
            p.set_best_value(pc)

        self.update_gbest()

    def update_best(self):
        for p in self.particles():
            pc = self._cf(p[:])
            if pc < p.best_value:
                p.set_best(p[:])
                p.set_best_value(pc)

        self.update_gbest()

    def kernel_update(self, **kwargs):
        raise NotImplementedError("Subclasses must implement kernel_update()")

    def update(self, callback=None, interval=None):
        while self._iters <= self._maxIters:
            self.kernel_update()
            self.update_best()
            if callback and (self._iters % interval == 0):
                callback(self)

            self._iters += 1

    @property
    def iters(self):
        return self._iters

    @property
    def maxIters(self):
        return self._maxIters


class QPSO(QPSOBase):
    """
    QPSO Original (Sun et al., 2004)

    Implementacion del Quantum Particle Swarm Optimization segun el paper original.
    Usa el Mean Best Position (mbest) para calcular la longitud caracteristica.

    Formula:
        mbest = (1/N) * sum(pbest_j) para j=1..N
        c = phi * pbest + (1-phi) * gbest, donde phi ~ U(0,1)
        L = alpha * |mbest - x|
        x_nuevo = c +/- L * ln(1/u)

    Parametros:
        cf: Funcion de costo a minimizar
        size: Numero de particulas
        dim: Dimensionalidad del problema
        bounds: Lista de tuplas (min, max) para cada dimension
        maxIters: Numero maximo de iteraciones
        alpha: Coeficiente de contraccion-expansion (default: 0.75)
               - Puede ser un valor fijo (float)
               - O una tupla (alpha_max, alpha_min) para decrecimiento lineal
    """
    def __init__(self, cf, size, dim, bounds, maxIters, alpha=0.75):
        self._alpha = alpha
        self._alpha_max = alpha[0] if isinstance(alpha, tuple) else alpha
        self._alpha_min = alpha[1] if isinstance(alpha, tuple) else alpha
        super(QPSO, self).__init__(cf, size, dim, bounds, maxIters)

    def _get_alpha(self):
        """Calcula alpha actual (fijo o con decrecimiento lineal)."""
        if isinstance(self._alpha, tuple):
            # Decrecimiento lineal: alpha = alpha_max - (alpha_max - alpha_min) * t / T
            t = self._iters
            T = self._maxIters
            return self._alpha_max - (self._alpha_max - self._alpha_min) * t / T
        return self._alpha

    def kernel_update(self, **kwargs):
        """
        Actualiza posiciones usando la formula QPSO original.

        Caracteristica distintiva: usa mbest (mean best) en el calculo de L.
        """
        mbest = self.mean_best()
        alpha = self._get_alpha()

        for p in self._particles:
            for i in range(0, self._dim):
                phi = random.uniform(0., 1.)
                u = random.uniform(0., 1.)
                rand_sign = 1 if random.random() > 0.5 else -1

                # Punto atractor local: combinacion convexa de pbest y gbest
                c = phi * p.best[i] + (1 - phi) * self._gbest[i]

                # Longitud caracteristica usando mbest (diferencia clave con QDPSO)
                L = alpha * abs(mbest[i] - p[i])

                # Nueva posicion con distribucion de Laplace
                p[i] = c + rand_sign * L * np.log(1. / u)


class QDPSO(QPSOBase):
    """
    QDPSO - Quantum Delta PSO (Variante)

    Variante del QPSO que usa una formula diferente para el punto atractor
    y la longitud caracteristica. Esta es la implementacion original del
    paquete PyPI tensor_qpso.

    Formula:
        c = (u1 * pbest + u2 * gbest) / (u1 + u2), donde u1, u2 ~ U(0,1)
        L = (1/g) * |x - c|
        x_nuevo = c +/- L * ln(1/u)

    Diferencias con QPSO original:
        - Punto atractor: usa promedio ponderado estocastico en lugar de
          combinacion convexa
        - Longitud L: usa |x - c| en lugar de |mbest - x|
        - Parametro: usa 'g' (tipico: 0.96) en lugar de 'alpha'

    Parametros:
        cf: Funcion de costo a minimizar
        size: Numero de particulas
        dim: Dimensionalidad del problema
        bounds: Lista de tuplas (min, max) para cada dimension
        maxIters: Numero maximo de iteraciones
        g: Coeficiente de contraccion-expansion (tipico: 0.96)
    """
    def __init__(self, cf, size, dim, bounds, maxIters, g=0.96):
        self._g = g
        super(QDPSO, self).__init__(cf, size, dim, bounds, maxIters)

    def kernel_update(self, **kwargs):
        """
        Actualiza posiciones usando la formula QDPSO (variante Delta).

        Caracteristica distintiva: usa |x - c| en lugar de |mbest - x|.
        """
        for p in self._particles:
            for i in range(0, self._dim):
                u1 = random.uniform(0., 1.)
                u2 = random.uniform(0., 1.)
                u3 = random.uniform(0., 1.)
                rand_sign = 1 if random.random() > 0.5 else -1

                # Punto atractor: promedio ponderado estocastico
                c = (u1 * p.best[i] + u2 * self._gbest[i]) / (u1 + u2)

                # Longitud caracteristica usando distancia a punto atractor
                L = (1 / self._g) * abs(p[i] - c)

                # Nueva posicion con distribucion de Laplace
                p[i] = c + rand_sign * L * np.log(1. / u3)
import torch

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Particle:
    def __init__(self, bounds):
        # Inicializa un tensor con valores aleatorios para todas las dimensiones
        lo, hi = zip(*bounds)
        lo = torch.tensor(lo, device=device)
        hi = torch.tensor(hi, device=device)
        
        # Genera valores aleatorios dentro de los límites especificados
        self._x = lo + (hi - lo) * torch.rand(len(bounds), device=device)
        
        self._best = self._x.clone()
        self._best_value = torch.tensor(float('inf'), device=device)

    def __str__(self):
        return str(self._x)

    @property
    def best(self):
        return self._best

    def set_best(self, x):
        self._best = x.clone()

    @property
    def best_value(self):
        return self._best_value

    def set_best_value(self, v):
        self._best_value = v.clone().detach()

    def __getitem__(self, key):
        return self._x[key]

    def __setitem__(self, key, val):
        self._x[key] = val

class Swarm:
    def __init__(self, size, dim, bounds):
        self._particles = [Particle(bounds) for _ in range(size)]
        self._dim = dim
        self._gbest_value = torch.tensor(float('inf'), device=device)
        self._gbest = torch.zeros(dim, device=device)

    def size(self):
        return len(self._particles)

    def particles(self):
        return self._particles

    def mean_best(self):
        # Recolecta todos los vectores de mejores posiciones en un tensor
        best_positions = torch.stack([p.best for p in self._particles], dim=0)
        
        # Calcula la media de los mejores vectores usando operaciones vectorizadas
        mean_best = torch.mean(best_positions, dim=0)
        return mean_best

    @property
    def gbest(self):
        return self._gbest

    @property
    def gbest_value(self):
        return self._gbest_value

    def update_gbest(self):
        # Recolecta todos los valores de best_value en un tensor
        best_values = torch.tensor([p.best_value for p in self._particles], device=device)
        # Encuentra el índice del mejor valor
        best_idx = torch.argmin(best_values)
        # Actualiza gbest y gbest_value usando el índice
        self._gbest = self._particles[best_idx].best.clone()
        self._gbest_value = best_values[best_idx].clone()

class QPSO(Swarm):
    def __init__(self, cf, size, dim, bounds, maxIters):
        super(QPSO, self).__init__(size, dim, bounds)
        self._cf = cf
        self._maxIters = maxIters
        self._iters = 0
        self.init_eval()

    def init_eval(self):
        # Evaluar todos los valores de función en uno solo en lugar de por partículas
        particle_positions = torch.stack([p.best for p in self.particles()], dim=0)  # Tensor de posiciones
        function_values = self._cf(particle_positions)  # Evaluamos para todos
        for i, p in enumerate(self.particles()):
            p.set_best_value(function_values[i])  # Set best values

        self.update_gbest()

    def update_best(self):
        # Obtener todos los valores de función para partículas en un solo paso
        particle_positions = torch.stack([p._x for p in self.particles()], dim=0)
        function_values = self._cf(particle_positions)  # Evaluamos la función para todas las partículas

        # Comparamos y actualizamos las mejores posiciones y valores
        for i, p in enumerate(self.particles()):
            if function_values[i] < p.best_value:
                p.set_best(particle_positions[i])
                p.set_best_value(function_values[i])  

        self.update_gbest()

    def kernel_update(self):
        # Generar números aleatorios en un solo paso
        u1 = torch.rand(len(self._particles), self._dim, device=device)
        u2 = torch.rand(len(self._particles), self._dim, device=device)
        u3 = torch.rand(len(self._particles), self._dim, device=device)

        # Calcular el valor de `c` en un solo paso usando broadcasting
        best_positions = torch.stack([p.best for p in self._particles], dim=0)
        c = (u1 * best_positions + u2 * self._gbest) / (u1 + u2)

        # Calcular las diferencias en un solo paso
        particle_positions = torch.stack([p._x for p in self._particles], dim=0)
        differences = particle_positions - c

        # Calcular las longitudes L usando broadcasting
        L = (1 / self._g) * torch.abs(differences)

        # Actualizar posiciones usando broadcasting
        rand_sign = torch.where(torch.rand(len(self._particles), self._dim, device=device) > 0.5, 1, -1)
        new_positions = c + rand_sign * L * torch.log(1. / u3)

        # Asignar las nuevas posiciones a las partículas
        for i, p in enumerate(self._particles):
            p._x = new_positions[i]

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

class QDPSO(QPSO):
    def __init__(self, cf, size, dim, bounds, maxIters, g):
        super(QDPSO, self).__init__(cf, size, dim, bounds, maxIters)
        self._g = g
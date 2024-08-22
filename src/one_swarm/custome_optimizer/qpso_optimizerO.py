import torch
from torch.optim import Optimizer
from tensor_qpso.qpsoO import QDPSO

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom QDPSO Optimizer Class for the ExtendedModel class
class QDPSOoOptimizer(Optimizer):
    def __init__(self, model, bounds, n_particles=20, max_iters=100, g=1.13, interval_parms_updated=10):
        if bounds is None:
            raise ValueError("Bounds must be provided")
        defaults = dict(n_particles=n_particles, max_iters=max_iters, g=g, bounds=bounds)
        super().__init__(model.parameters(), defaults)

        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iters = max_iters
        self.interval_parms_updated = interval_parms_updated
        self.g = g
        self.model = model
        self.params = list(model.parameters())
        self.dim = sum(p.numel() for p in self.params)
        self.optimizer = None
        self.best_params = None
        self.best_loss = float('inf')

    def _initialize_optimizer(self):
        self.optimizer = QDPSO(self._fitness_function, self.n_particles, self.dim, self.bounds, self.max_iters, self.g)

    def _fitness_function(self, flat_params_batch):
        losses = torch.empty(len(flat_params_batch), device=device)
        for i, flat_params in enumerate(flat_params_batch):
            self.model._set_params(flat_params)
            output = self.model(self.X_train)
            loss = self.model.lf.cross_entropy(self.y_train_one_hot, output)
            losses[i] = loss.item()
        return losses

    def _set_params(self, flat_params):
        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p).to(device))
            offset += numel

    def step(self):
        self._initialize_optimizer()
        self.optimizer.update(callback=self._log_callback, interval=self.interval_parms_updated)
        self._set_params(self.optimizer.gbest)

    def _log_callback(self, s):
        # best_value = torch.tensor([p.best_value for p in s.particles()], device=device)
        # best_value_avg = torch.mean(best_value).item()
        # best_value_std = torch.std(best_value).item()
        self._set_params(s.gbest)
        if s.gbest_value < self.best_loss:
            self.best_loss = s.gbest_value
            self.best_params = s.gbest.clone()

    def set_training_data(self, X_train, y_train_one_hot):
        self.X_train = X_train.to(device)
        self.y_train_one_hot = y_train_one_hot.to(device)
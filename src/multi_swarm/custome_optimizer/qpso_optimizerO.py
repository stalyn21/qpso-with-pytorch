import torch
import logging
from torch.optim import Optimizer
from tensor_qpso.qpsoO import QDPSO

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom QDPSOo Swarms Optimizer Class for the ExtendedModel class
class LayerQDPSOoOptimizer(Optimizer):
    def __init__(self, model, layer_idx, total_params, n_particles=20, max_iters=100, g=1.13, interval_parms_updated=10):
        defaults = dict(n_particles=n_particles, max_iters=max_iters, g=g)
        super().__init__(model.parameters(), defaults)
        self.layer_idx = layer_idx
        self.n_particles = n_particles
        self.max_iters = max_iters
        self.interval_parms_updated = interval_parms_updated
        self.g = g
        self.model = model
        self.total_params = total_params
        self.layer_params = model.get_flat_params_layer(layer_idx)
        self.bounds = [(-1, 1) for _ in range(len(self.layer_params))]
        self.dim = len(self.layer_params)
        self.optimizer = None
        self.best_params = None
        self.best_loss = float('inf')
        self.X_train = None
        self.y_train_one_hot = None

    def _initialize_optimizer(self):
        self.optimizer = QDPSO(self._fitness_function, self.n_particles, self.dim, self.bounds, self.max_iters, self.g)

    def _fitness_function(self, flat_params_batch):
        losses = torch.empty(len(flat_params_batch), device=device)
        for i, flat_params in enumerate(flat_params_batch):
            self.model.set_flat_params_layer(self.layer_idx, flat_params)
            output = self.model(self.X_train)
            loss = self.model.lf.cross_entropy(self.y_train_one_hot, output)
            losses[i] = loss.item()
        return losses

    def step(self):
        if self.optimizer is None:
            self._initialize_optimizer()
        self.optimizer.update(callback=self._log_callback, interval=self.interval_parms_updated)
        self.model.set_flat_params_layer(self.layer_idx, self.optimizer.gbest)

    def _log_callback(self, s):
        self.model.set_flat_params_layer(self.layer_idx, s.gbest)
        if s.gbest_value < self.best_loss:
            self.best_loss = s.gbest_value
            self.best_params = s.gbest.clone()

    def set_training_data(self, X_train, y_train_one_hot):
        self.X_train = X_train.to(device)
        self.y_train_one_hot = y_train_one_hot.to(device)

    def print_layer_info(self):
        logging.info(f"Optimizer initializing and Training layer {self.layer_idx} with particle dimension: {self.dim}")
        # logging.info(f"Current parameters: {self.layer_params}")
import torch
import logging

from torch.optim import Optimizer
from one_swarm.tensor_qpso.qpsoO import QDPSO

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom QDPSO Optimizer Class for the ExtendedModel class
class QDPSOoOptimizer(Optimizer):
    def __init__(self, model, bounds, n_particles, max_iters, g, interval_parms_updated):
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

        self.train_losses = []
        self.val_losses = []

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
        # Reiniciar las listas de pérdidas al inicio del step
        self.train_losses = []
        self.val_losses = []
        self._initialize_optimizer()
        self.best_val_loss = float('inf')
        self.epoch = 0
        self.optimizer.update(callback=self._log_callback, interval=self.interval_parms_updated)
        self._set_params(self.optimizer.gbest)

    def _log_callback(self, s):
        if self.epoch > self.max_iters:
            self.epoch = 0

        self._set_params(s.gbest)
        
        # Evaluate the model on the validation set
        with torch.no_grad():
            val_output = self.model(self.X_val)
            val_loss = self.model.lf.cross_entropy(self.y_val_one_hot, val_output)
        
        # Update the best parameters if the validation loss improves
        if val_loss.item() < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self.best_params = s.gbest.clone()
            logging.info(f'Epoch {self.epoch * self.interval_parms_updated}'
                         f' - Train Loss: {s.gbest_value:.4f}'
                         f' - Val Loss: {self.best_val_loss:.4f}'
                         f' - Best Val Loss: {self.best_val_loss:.4f}')
        else:
            logging.info(f'Epoch {self.epoch * self.interval_parms_updated}'
                         f' - Train Loss: {s.gbest_value:.4f}'
                         f' - Val Loss: {val_loss.item():.4f}'
                         f' - Best Val Loss: {self.best_val_loss:.4f}')
            
        # Guardar las pérdidas solo si no hemos superado max_iters
        if self.epoch < self.max_iters:
            self.train_losses.append(s.gbest_value)
            self.val_losses.append(val_loss.item())

        self.epoch = self.epoch + 1

    def set_training_data(self, X_train, y_train_one_hot, X_val, y_val_one_hot):
        self.X_train = X_train.to(device)
        self.y_train_one_hot = y_train_one_hot.to(device)
        self.X_val = X_val.to(device)
        self.y_val_one_hot = y_val_one_hot.to(device)
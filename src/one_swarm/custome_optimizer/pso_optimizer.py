import torch
import logging
from torch_pso import ParticleSwarmOptimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PSOOptimizer:
    def __init__(self, model, n_particles, max_iters, inertial_weight=0.5,
                 cognitive_coefficient=0.3, social_coefficient=0.9, min_param_value=None, max_param_value=None):
        self.model = model
        self.n_particles = n_particles
        self.max_iters = max_iters

        # Configuración base del optimizador
        optimizer_params = {
            'params': self.model.parameters(),
            'num_particles': n_particles,
            'inertial_weight': inertial_weight,
            'cognitive_coefficient': cognitive_coefficient,
            'social_coefficient': social_coefficient
        }

        # Añadir límites solo si se especifican
        if min_param_value is not None and max_param_value is not None:
            optimizer_params['min_param_value'] = min_param_value
            optimizer_params['max_param_value'] = max_param_value
            logging.info(f"PSO configurado con límites: [{min_param_value}, {max_param_value}]")
        else:
            logging.info("PSO configurado sin límites de parámetros")

        self.optimizer = ParticleSwarmOptimizer(**optimizer_params)
        self.best_params = None
        self.best_val_loss = float('inf')

        self.train_losses = []
        self.val_losses = []

    def _evaluate(self, X, y):
        outputs = self.model(X)
        return self.model.lf.cross_entropy(y, outputs)

    def step(self, X_train, y_train, X_val, y_val, callback_interval=1):
        self.epoch = 0

        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.max_iters):
            def closure():
                self.optimizer.zero_grad()
                loss = self._evaluate(X_train, y_train)
                return loss

            train_loss = self.optimizer.step(closure)

            # Evaluación en conjunto de validación
            with torch.no_grad():
                val_loss = self._evaluate(X_val, y_val)

            # Guardar las pérdidas
            self.train_losses.append(train_loss.item())
            self.val_losses.append(val_loss.item())

            # Actualizar mejores parámetros si mejora la pérdida de validación
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_params = {
                    name: param.clone().detach()
                    for name, param in self.model.named_parameters()
                }

            if epoch % callback_interval == 0:
                logging.info(f'Epoch {epoch}'
                           f' - Train Loss: {train_loss.item():.4f}'
                           f' - Val Loss: {val_loss.item():.4f}'
                           f' - Best Val Loss: {self.best_val_loss:.4f}')

            self.epoch += 1

        # Cargar los mejores parámetros encontrados
        if self.best_params is not None:
            for name, param in self.model.named_parameters():
                param.data.copy_(self.best_params[name])
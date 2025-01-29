import torch
import logging
from torch.optim import Optimizer
from torch_pso import ParticleSwarmOptimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerPSOOptimizer(Optimizer):
    """
    Custom Particle Swarm Optimization (PSO) Optimizer for individual layers of the ExtendedModel.

    Args:
        model (ExtendedModel): The neural network model to optimize.
        layer_idx (int): Index of the layer to optimize.
        total_params (torch.Tensor): Flattened tensor of all model parameters.
        n_particles (int): Number of particles in the swarm.
        max_iters (int): Maximum number of iterations for optimization.
        inertial_weight (float): Inertia weight for particle velocity update.
        cognitive_coefficient (float): Cognitive coefficient for particle velocity update.
        social_coefficient (float): Social coefficient for particle velocity update.
        min_param_value (float): Minimum parameter value for bounds.
        max_param_value (float): Maximum parameter value for bounds.
        interval_parms_updated (int): Interval at which parameters are updated and logged.
    """
    def __init__(self, model, layer_idx, total_params, n_particles=20, max_iters=100,
                 inertial_weight=0.5, cognitive_coefficient=0.3, social_coefficient=0.9,
                 min_param_value=-1, max_param_value=1, interval_parms_updated=10):
        defaults = dict(n_particles=n_particles, max_iters=max_iters)
        super().__init__(model.parameters(), defaults)

        self.layer_idx = layer_idx
        self.n_particles = n_particles
        self.max_iters = max_iters
        self.interval_parms_updated = interval_parms_updated
        self.model = model
        self.total_params = total_params
        self.layer_params = model.get_flat_params_layer(layer_idx)
        self.dim = len(self.layer_params)

        # Configuración PSO
        self.inertial_weight = inertial_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.min_param_value = min_param_value
        self.max_param_value = max_param_value

        self.optimizer = None
        self.best_params = None

        # Listas para almacenar las pérdidas
        self.train_losses = []
        self.val_losses = []
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.epoch = 0

    def _initialize_optimizer(self):
        """Initialize the PSO optimizer with the current layer parameters."""
        optimizer_params = {
            'params': [self.layer_params],
            'num_particles': self.n_particles,
            'inertial_weight': self.inertial_weight,
            'cognitive_coefficient': self.cognitive_coefficient,
            'social_coefficient': self.social_coefficient,
            'min_param_value': self.min_param_value,
            'max_param_value': self.max_param_value
        }
        self.optimizer = ParticleSwarmOptimizer(**optimizer_params)

    def step(self):
        """Perform optimization steps for the current layer."""
        # Reiniciar las listas de pérdidas al inicio del step
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')

        self._initialize_optimizer()

        for epoch in range(self.max_iters):
            def closure():
                self.optimizer.zero_grad()
                output = self.model(self.X_train)
                loss = self.model.lf.cross_entropy(self.y_train_one_hot, output)
                return loss

            train_loss = self.optimizer.step(closure)

            # Evaluación en conjunto de validación
            with torch.no_grad():
                val_output = self.model(self.X_val)
                val_loss = self.model.lf.cross_entropy(self.y_val_one_hot, val_output)

            # Guardar las pérdidas
            self.train_losses.append(train_loss.item())
            self.val_losses.append(val_loss.item())

            # Actualizar mejores parámetros si mejora la pérdida de validación
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss.item()
                self.best_train_loss = train_loss.item()
                self.best_params = self.layer_params.clone()

                if epoch % self.interval_parms_updated == 0:
                    logging.info(f'Layer {self.layer_idx} - Epoch {epoch}'
                               f' - Train Loss: {train_loss.item():.4f}'
                               f' - Val Loss: {val_loss.item():.4f}'
                               f' - Best Val Loss: {self.best_val_loss:.4f}')

            self.epoch += 1

        # Establecer los mejores parámetros encontrados
        if self.best_params is not None:
            self.model.set_flat_params_layer(self.layer_idx, self.best_params)

    def set_training_data(self, X_train, y_train_one_hot, X_val, y_val_one_hot):
        """
        Set the training and validation data for the optimizer.

        Args:
            X_train (torch.Tensor): Input training data
            y_train_one_hot (torch.Tensor): One-hot encoded target training data
            X_val (torch.Tensor): Input validation data
            y_val_one_hot (torch.Tensor): One-hot encoded target validation data
        """
        self.X_train = X_train.to(device)
        self.y_train_one_hot = y_train_one_hot.to(device)
        self.X_val = X_val.to(device)
        self.y_val_one_hot = y_val_one_hot.to(device)

    def print_layer_info(self):
        """Print information about the current layer being optimized."""
        logging.info(f"Training layer {self.layer_idx} with particle dimension: {self.dim}")

    def get_best_losses(self):
        """
        Returns the best training and validation losses achieved.

        Returns:
            tuple: (best_train_loss, best_val_loss)
        """
        return self.best_train_loss, self.best_val_loss

    def get_loss_history(self):
        """
        Returns the history of training and validation losses.

        Returns:
            tuple: (train_losses, val_losses)
        """
        return self.train_losses, self.val_losses
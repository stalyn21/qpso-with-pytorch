import torch
import logging
from torch.optim import Optimizer
from one_swarm.tensor_qpso.qpsoO import QDPSO

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerQDPSOoOptimizer(Optimizer):
    """
    Custom Quantum-behaved Particle Swarm Optimization (QPSO) Optimizer for individual layers of the ExtendedModel.

    Args:
        model (ExtendedModel): The neural network model to optimize.
        layer_idx (int): Index of the layer to optimize.
        total_params (torch.Tensor): Flattened tensor of all model parameters.
        n_particles (int): Number of particles in the swarm.
        max_iters (int): Maximum number of iterations for optimization.
        g (float): Contraction-expansion coefficient.
        interval_parms_updated (int): Interval at which parameters are updated and logged.
    """
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

        # Listas para almacenar las pérdidas
        self.train_losses = []
        self.val_losses = []
        self.best_train_loss = float('inf')

    def _initialize_optimizer(self):
        """Initialize the QDPSO optimizer."""
        self.optimizer = QDPSO(self._fitness_function, self.n_particles, self.dim, self.bounds, self.max_iters, self.g)

    def _fitness_function(self, flat_params_batch):
        """
        Compute the fitness (loss) for a batch of parameter sets.

        Args:
            flat_params_batch (list of torch.Tensor): Batch of flattened parameter tensors.

        Returns:
            torch.Tensor: Tensor of loss values for each parameter set.
        """
        losses = torch.empty(len(flat_params_batch), device=device)
        for i, flat_params in enumerate(flat_params_batch):
            self.model.set_flat_params_layer(self.layer_idx, flat_params)
            output = self.model(self.X_train)
            loss = self.model.lf.cross_entropy(self.y_train_one_hot, output)
            losses[i] = loss.item()
        return losses

    def step(self):
        """Perform one step of optimization."""
        # Reiniciar las listas de pérdidas al inicio del step
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')

        self._initialize_optimizer()
        self.epoch = 0
        self.optimizer.update(callback=self._log_callback, interval=self.interval_parms_updated)
        self.model.set_flat_params_layer(self.layer_idx, self.optimizer.gbest)

    def _log_callback(self, s):
        """
        Callback function for logging optimization progress.

        Args:
            s (QDPSO): The QDPSO optimizer instance.
        """
        if self.epoch > self.max_iters:
            self.epoch = 0

        self.model.set_flat_params_layer(self.layer_idx, s.gbest)

        # Evaluate the model on both training and validation sets
        with torch.no_grad():
            train_output = self.model(self.X_train)
            train_loss = self.model.lf.cross_entropy(self.y_train_one_hot, train_output)

            val_output = self.model(self.X_val)
            val_loss = self.model.lf.cross_entropy(self.y_val_one_hot, val_output)

        # Guardar las pérdidas
        if self.epoch < self.max_iters:
            self.train_losses.append(train_loss.item())
            self.val_losses.append(val_loss.item())

        # Update the best parameters if the validation loss improves
        if val_loss.item() < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self.best_train_loss = train_loss.item()
            self.best_params = s.gbest.clone()
            logging.info(f'Layer {self.layer_idx} - Epoch {self.epoch * self.interval_parms_updated}'
                        f' - Train Loss: {train_loss.item():.4f}'
                        f' - Val Loss: {self.best_val_loss:.4f}'
                        f' - Best Val Loss: {self.best_val_loss:.4f}')
        else:
            logging.info(f'Layer {self.layer_idx} - Epoch {self.epoch * self.interval_parms_updated}'
                        f' - Train Loss: {train_loss.item():.4f}'
                        f' - Val Loss: {val_loss.item():.4f}'
                        f' - Best Val Loss: {self.best_val_loss:.4f}')

        self.epoch += 1

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
        # logging.info(f"Current parameters: {self.layer_params}")

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
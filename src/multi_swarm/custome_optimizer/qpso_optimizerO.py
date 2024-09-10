import torch
import logging
from torch.optim import Optimizer
from tensor_qpso.qpsoO import QDPSO

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom QDPSOo Swarms Optimizer Class for the ExtendedModel class
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
        self.best_loss = float('inf')

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
            #self.model._set_params(flat_params)
            self.model.set_flat_params_layer(self.layer_idx, flat_params)
            output = self.model(self.X_train)
            loss = self.model.lf.cross_entropy(self.y_train_one_hot, output)
            losses[i] = loss.item()
        return losses

    def step(self):
        """Perform one step of optimization."""
        self._initialize_optimizer()
        self.optimizer.update(callback=self._log_callback, interval=self.interval_parms_updated)
        #self._set_params(self.optimizer.gbest)
        self.model.set_flat_params_layer(self.layer_idx, self.optimizer.gbest)

    def _log_callback(self, s):
        """
        Callback function for logging optimization progress.

        Args:
            s (QDPSO): The QDPSO optimizer instance.
        """
        # best_value = torch.tensor([p.best_value for p in s.particles()], device=device)
        # best_value_avg = torch.mean(best_value).item()
        # best_value_std = torch.std(best_value).item()
        #self._set_params(s.gbest)
        
        self.model.set_flat_params_layer(self.layer_idx, s.gbest)
        if s.gbest_value < self.best_loss:
            self.best_loss = s.gbest_value
            self.best_params = s.gbest.clone()

    def set_training_data(self, X_train, y_train_one_hot):
        """
        Set the training data for the optimizer.

        Args:
            X_train (torch.Tensor): Input training data.
            y_train_one_hot (torch.Tensor): One-hot encoded target training data.
        """
        self.X_train = X_train.to(device)
        self.y_train_one_hot = y_train_one_hot.to(device)

    def print_layer_info(self):
        """Print information about the current layer being optimized."""
        logging.info(f"Training layer {self.layer_idx} with particle dimension: {self.dim}")
        # logging.info(f"Current parameters: {self.layer_params}")
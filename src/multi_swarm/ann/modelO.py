import torch

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Activation function class
class ActivationFunction:
    """
    Class containing activation functions.
    """
    @staticmethod
    def softmax(z, dim=1):
        """
        Apply softmax activation function.

        Args:
            z (torch.Tensor): Input tensor.
            dim (int): Dimension along which softmax will be computed.

        Returns:
            torch.Tensor: Output after applying softmax.
        """
        return torch.nn.functional.softmax(z, dim=dim)

# Loss functions class
class LossFunction:
    """
    Class for computing loss functions with L2 regularization.

    Args:
        num_samples (int): Number of samples in the dataset.
        model (torch.nn.Module): The model being trained.
        beta (float): L2 regularization coefficient.
    """
    def __init__(self, num_samples, model, beta=5e-4):
        self.L2 = beta / num_samples
        self.model = model

    def cross_entropy(self, y, y_hat):
        """
        Compute cross entropy loss with L2 regularization.

        Args:
            y (torch.Tensor): True labels.
            y_hat (torch.Tensor): Predicted labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        cross_entropy_loss = torch.nn.functional.cross_entropy(y_hat, y, reduction='mean')
        l2_reg = sum(param.norm() for param in self.model.parameters())
        return cross_entropy_loss + self.L2 * l2_reg

# Extended Model Class
class ExtendedModel(torch.nn.Module):
    """
    Extended neural network model with custom forward pass and parameter management.

    Args:
        num_samples (int): Number of samples in the dataset.
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output (number of classes).
        hidden_layers (list): List of integers representing the number of neurons in each hidden layer.
    """
    def __init__(self, num_samples, input_dim, output_dim, hidden_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, hidden_layers[0])] +
            [torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]) for i in range(len(hidden_layers) - 1)] +
            [torch.nn.Linear(hidden_layers[-1], output_dim)]
        )
        self.af = ActivationFunction()
        self.lf = LossFunction(num_samples, self)

    def forward(self, X, params=None):
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): Input data.
            params (torch.Tensor, optional): Flattened model parameters.

        Returns:
            torch.Tensor: Model output after softmax activation.
        """
        if params is not None:
            self._set_params(params)
        for i in range(len(self.layers) - 1):
            X = torch.tanh(self.layers[i](X))
        y_hat = self.af.softmax(self.layers[-1](X))
        return y_hat

    def _set_params(self, flat_params):
        """
        Set model parameters from a flattened tensor.

        Args:
            flat_params (torch.Tensor): Flattened tensor of model parameters.
        """
        offset = 0
        for layer in self.layers:
            for p in layer.parameters():
                numel = p.numel()
                p.data = flat_params[offset:offset + numel].view_as(p.data).to(device)
                offset += numel

    def get_flat_params(self):
        """
        Get all model parameters as a flattened tensor.

        Returns:
            torch.Tensor: Flattened tensor of all model parameters.
        """
        return torch.cat([p.data.view(-1) for layer in self.layers for p in layer.parameters()]).to(device)

    def get_flat_params_layer(self, layer_idx):
        """
        Get parameters of a specific layer as a flattened tensor.

        Args:
            layer_idx (int): Index of the layer.

        Returns:
            torch.Tensor: Flattened tensor of the specified layer's parameters.
        """
        layer = self.layers[layer_idx]
        return torch.cat([p.data.view(-1) for p in layer.parameters()]).to(device)

    def set_flat_params_layer(self, layer_idx, flat_params):
        """
        Set parameters of a specific layer from a flattened tensor.

        Args:
            layer_idx (int): Index of the layer.
            flat_params (torch.Tensor): Flattened tensor of parameters for the specified layer.
        """
        offset = 0
        layer = self.layers[layer_idx]
        for p in layer.parameters():
            numel = p.numel()
            p.data = flat_params[offset:offset + numel].view_as(p.data).to(device)
            offset += numel

    def evaluate(self, X, y, params):
        """
        Evaluate the model on given data and parameters.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): True labels.
            params (torch.Tensor): Flattened model parameters.

        Returns:
            torch.Tensor: Computed loss.
        """
        y_hat = self.forward(X, params)
        loss = self.lf.cross_entropy(y, y_hat)
        return loss
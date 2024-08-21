import torch

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Activation function class
class ActivationFunction:
    @staticmethod
    def softmax(z):
        return torch.nn.functional.softmax(z, dim=1)

# Loss functions class
class LossFunction:
    def __init__(self, X_sample, model):
        self.n_sample_lf = X_sample
        self.beta = 5e-4
        self.L2 = self.beta / self.n_sample_lf
        self.model = model

    # Customize cross-entropy loss function with L2 regularization
    def cross_entropy(self, y, y_hat):
        cross_entropy_loss = torch.nn.functional.cross_entropy(y_hat, y, reduction='mean')
        l2_reg = sum(torch.norm(param) for param in self.model.parameters())
        return cross_entropy_loss + self.L2 * l2_reg

# Extended Model Class
class ExtendedModel(torch.nn.Module):
    def __init__(self, X_sample, input_dim, output_dim, hidden_layers):
        super(ExtendedModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.n_sample = X_sample

        # Input layer
        self.layers.append(torch.nn.Linear(input_dim, hidden_layers[0]))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        # Output layer
        self.layers.append(torch.nn.Linear(hidden_layers[-1], output_dim))

        self.af = ActivationFunction()
        self.lf = LossFunction(X_sample, self)

    def forward(self, X, params=None):
        if params is not None:
            self._set_params(params)

        for i in range(len(self.layers) - 1):
            X = torch.tanh(self.layers[i](X))
        y_hat = self.af.softmax(self.layers[-1](X))
        return y_hat

    def _set_params(self, flat_params):
        offset = 0
        for layer in self.layers:
            for p in layer.parameters():
                numel = p.numel()
                p.data = flat_params[offset:offset + numel].view_as(p.data).to(device)
                offset += numel

    def get_flat_params(self):
        return torch.cat([p.data.view(-1) for layer in self.layers for p in layer.parameters()]).to(device)

    def evaluate(self, X, y, params):
        y_hat = self.forward(X, params)
        loss = self.lf.cross_entropy(y, y_hat)
        return loss
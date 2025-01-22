import torch

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Activation function class
class ActivationFunction:
    @staticmethod
    def softmax(z):
        return torch.nn.functional.softmax(z, dim=1)

"""
# Standard Loss functions class 
# ---- L2 utiliza directamente la suma de los cuadrados ----
# -> Inicialización: La constante beta se pasa directamente y no se ajusta en función del tamaño del conjunto de datos.
# -> Regularización L2: Se calcula sumando los cuadrados de todos los elementos de los parámetros (param.pow(2.0).sum()). 
# Esto es una implementación más directa de la regularización L2, que es la suma de los cuadrados de los parámetros.

class LossFunction:
    def __init__(self, beta=5e-4):
        self.beta = beta

    def cross_entropy(self, y, y_hat, model):
        cross_entropy_loss = torch.nn.functional.cross_entropy(y_hat, y, reduction='mean')
        l2_reg = sum(param.pow(2.0).sum() for param in model.parameters())
        return cross_entropy_loss + self.beta * l2_reg
"""

# Loss functions class
# ---- L2 ajusta la regularización en función del tamaño del conjunto de datos ----
# -> Inicialización: La constante de regularización L2 se calcula dividiendo beta por X_sample, que es el número de muestras. 
# Esto implica que la regularización se ajusta en función del tamaño del conjunto de datos.
# -> Regularización L2: Se calcula usando la norma de cada parámetro del modelo (torch.norm(param)), 
# lo cual es una medida de magnitud que incluye tanto la suma de los cuadrados como la raíz cuadrada.

class LossFunction:
    def __init__(self, X_sample, model):
        self.n_sample_lf = X_sample
        self.beta = 5e-4
        self.L2 = self.beta / self.n_sample_lf
        self.model = model

    def cross_entropy(self, y, y_hat):
        cross_entropy_loss = torch.nn.functional.cross_entropy(y_hat, y, reduction='mean')
        l2_reg = sum(torch.norm(param) for param in self.model.parameters())
        return cross_entropy_loss + self.L2 * l2_reg

# Extended Model Class
class ExtendedModel(torch.nn.Module):
    # def __init__(self, input_dim, output_dim, hidden_layers):
    def __init__(self, X_sample, input_dim, output_dim, hidden_layers):
        super(ExtendedModel, self).__init__()
        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(torch.nn.Linear(input_dim, hidden_layers[0]))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        # Output layer
        self.layers.append(torch.nn.Linear(hidden_layers[-1], output_dim))

        self.af = ActivationFunction()
        # self.lf = LossFunction()
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
        loss = self.lf.cross_entropy(y, y_hat, self)
        return loss
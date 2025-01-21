import torch
import torch.nn as nn
import torch.nn.functional as F

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActivationFunction:
    @staticmethod
    def softmax(z, dim=1):
        return F.softmax(z, dim=dim)

class LossFunction:
    def __init__(self, num_samples, model, beta=5e-4):
        self.L2 = beta / num_samples
        self.model = model

    def cross_entropy(self, y, y_hat):
        cross_entropy_loss = F.cross_entropy(y_hat, y, reduction='mean')
        l2_reg = sum(param.norm() for param in self.model.parameters())
        return cross_entropy_loss + self.L2 * l2_reg

class ExtendedModelPSO(nn.Module):
    def __init__(self, num_samples, input_dim, output_dim, hidden_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_layers[0])] +
            [nn.Linear(hidden_layers[i], hidden_layers[i + 1]) for i in range(len(hidden_layers) - 1)] +
            [nn.Linear(hidden_layers[-1], output_dim)]
        )
        self.af = ActivationFunction()
        self.lf = LossFunction(num_samples, self)

    def forward(self, X):
        for i in range(len(self.layers) - 1):
            X = torch.tanh(self.layers[i](X))
        return self.af.softmax(self.layers[-1](X))

    def evaluate(self, X, y):
        y_hat = self.forward(X)
        return self.lf.cross_entropy(y, y_hat)
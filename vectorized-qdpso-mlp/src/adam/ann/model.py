import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActivationFunction:
    @staticmethod
    def softmax(z):
        return torch.nn.functional.softmax(z, dim=1)

class LossFunction:
    def __init__(self, X_sample, model):
        self.n_sample_lf = X_sample.shape[0]  # Asumimos que X_sample es un tensor
        self.beta = 5e-4
        self.L2 = self.beta / self.n_sample_lf
        self.model = model

    def crossentropy(self, y, y_hat):
        cross_entropy_loss = torch.nn.functional.cross_entropy(y_hat, y, reduction='mean')
        l2_reg = sum(torch.norm(param) for param in self.model.parameters())
        return cross_entropy_loss + self.L2 * l2_reg

class ExtendedModel(torch.nn.Module):
    def __init__(self, Xsample, inputdim, outputdim, hiddenlayers):
        super(ExtendedModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(inputdim, hiddenlayers[0]))
        
        for i in range(len(hiddenlayers) - 1):
            self.layers.append(torch.nn.Linear(hiddenlayers[i], hiddenlayers[i+1]))
        
        self.layers.append(torch.nn.Linear(hiddenlayers[-1], outputdim))
        self.af = ActivationFunction()
        self.lf = LossFunction(Xsample, self)
        
    def forward(self, X, params=None):
        if params is not None:
            self.setparams(params)
        
        for i in range(len(self.layers) - 1):
            X = torch.tanh(self.layers[i](X))
        
        yhat = self.af.softmax(self.layers[-1](X))
        return yhat
    
    def setparams(self, flatparams):
        offset = 0
        for layer in self.layers:
            for p in layer.parameters():
                numel = p.numel()
                p.data = flatparams[offset:offset + numel].view(p.data.shape).to(device)
                offset += numel
    
    def getflatparams(self):
        return torch.cat([p.data.view(-1) for layer in self.layers for p in layer.parameters()]).to(device)
    
    def evaluate(self, X, y, params):
        yhat = self.forward(X, params)
        loss = self.lf.crossentropy(y, yhat)
        return loss
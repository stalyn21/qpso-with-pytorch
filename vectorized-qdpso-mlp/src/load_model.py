import torch
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, make_circles
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from one_swarm.ann import ExtendedModel  # Asegúrate de que esta importación sea correcta

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data(dataset='iris'):
    dataset_dict = {
        'iris': load_iris,
        'breast_cancer': load_breast_cancer,
        'wine': load_wine,
    }
    if dataset in dataset_dict:
        data = dataset_dict[dataset]()
    elif dataset == 'circle':
        n = 500
        X, y = make_circles(n_samples=n, factor=0.5, noise=0.05)
        Dataset = namedtuple('Dataset', ['data', 'target', 'feature_names', 'target_names'])
        data = Dataset(
            data=X,
            target=y,
            feature_names=['X coordinate', 'Y coordinate'],
            target_names=['outer circle', 'inner circle']
        )
    else:
        raise ValueError("Unknown dataset")

    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=100)

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    config = checkpoint['config']
    model = ExtendedModel(
        num_samples=config['n_samples'],
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        hidden_layers=config['hidden_layers']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config, checkpoint['best_val_acc']

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_tensor).float().mean().item()
    return accuracy

def main():
    # load and preprocess the data accepcting the dataset name: iris, breast_cancer, wine, and circle 
    dataset_name = 'circle'
    # Cargar el modelo
    model_path = f"./models/{dataset_name}_best_model.pth"
    model, config, best_val_acc = load_model(model_path)
    print(f"Loaded model with best validation accuracy: {best_val_acc:.4f}")

    # Cargar y preprocesar los datos
    X_train_val, X_test, y_train_val, y_test = load_and_preprocess_data(dataset_name)

    # Evaluar el modelo en el conjunto de prueba
    test_accuracy = evaluate_model(model, X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Imprimir la configuración del modelo
    print("Model configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
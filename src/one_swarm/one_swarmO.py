import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split

# custome optimizer and model
from custome_optimizer.qpso_optimizerO import QDPSOoOptimizer
from ann.annO import ExtendedModel

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el conjunto de datos iris
data = load_iris()
X, y = data.data, data.target

# Escalar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir el conjunto de datos: 80% para entrenamiento y validación, 20% para pruebas
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# QPSOo Optimizer Parameters
n_particles = 20
max_iters = 60
interval_parms_updated = 10
g = 1.13

# K-Fold Cross-Validation: 4 pliegues en el conjunto de entrenamiento y validación
kf = KFold(n_splits=4, shuffle=True, random_state=100)

# Calcular dimensiones y crear el modelo
input_dim = X_train_val.shape[1]
output_dim = len(set(y_train_val))
hidden_layers = [ input_dim * 3 ]

# Initialize the model
model = ExtendedModel(num_samples=X_train_val.shape[0], input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers).to(device)

bounds = [(-1, 1) for _ in range(sum(p.numel() for p in model.parameters()))]
# Initialize the QPSOo optimizer
optimizer = QDPSOoOptimizer(model, bounds, n_particles=n_particles, max_iters=max_iters, g=g, interval_parms_updated=interval_parms_updated)

test_n_samples = []
train_n_samples = []
val_n_samples = []
test_results = []

fold = 0
print(f"=========================")

for train_index, val_index in kf.split(X_train_val):
    fold += 1
    print(f"======= Fold {fold} =======")
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    test_n_samples.append(X_test.shape[0])
    train_n_samples.append(X_train.shape[0])
    val_n_samples.append(X_val.shape[0])

    y_train_one_hot = torch.nn.functional.one_hot(y_train, num_classes=output_dim).float().to(device)
    y_val_one_hot = torch.nn.functional.one_hot(y_val, num_classes=output_dim).float().to(device)

    # Set the training data using the QPSOo optimizer
    optimizer.set_training_data(X_train, y_train_one_hot)

    for epoch in range(4):
        print(f"Starting epoch {epoch + 1} for fold {fold}")
        optimizer.step()

        # Calcular la pérdida de entrenamiento y validación
        output = model(X_train)
        loss = model.lf.cross_entropy(y_train_one_hot, output)

        # Validación
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = model.lf.cross_entropy(y_val_one_hot, val_output)

        print(f'Fold {fold}, Epoch {epoch + 1}, Loss: {loss.item()} - Validation Loss: {val_loss.item()}')
        print(f"Finished epoch {epoch + 1} for fold {fold}")

    # Evaluación final en el conjunto de validación
    model._set_params(optimizer.best_params)
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = model.lf.cross_entropy(y_val_one_hot, val_output)

    print(f'Final Validation Loss for fold {fold}: {val_loss.item()}')

    # Evaluación en el conjunto de prueba
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    y_pred = model(X_test_tensor).argmax(dim=1)
    accuracy = (y_pred == y_test_tensor).float().mean().item()
    print(f'Fold {fold}, Accuracy on iris test dataset: {accuracy:.4f}')
    print(f"=========================")

    test_results.append(accuracy)

# Imprimir la configuración del modelo
print("=============================================")
print("Model Setup:")
print(f"Test Sample Size: {test_n_samples}")
print(f"Training Sample Size: {train_n_samples}")
print(f"Validation Sample Size: {val_n_samples}")
print(f"Input Dimension: {input_dim}")
print(f"Output Dimension: {output_dim}")
print(f"The Number of Hidden Layers: {len(hidden_layers)}")
print(f"The Neurons of Hidden Layers: {hidden_layers}")
print(f"Number of Particles: {n_particles}")
print(f"Maximum Iterations: {max_iters}")
print(f"Inertia Weight: {g}")
print(f"Bounds: {len(bounds)}")
print("=============================================")

# Resultados finales
mean_accuracy = torch.tensor(test_results).mean().item()
std_accuracy = torch.tensor(test_results).std().item()
print(f'Mean accuracy on test dataset: {mean_accuracy:.4f}')
print(f'Standard deviation of accuracy on test dataset: {std_accuracy:.4f}')
print("=============================================")
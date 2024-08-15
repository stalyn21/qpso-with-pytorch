import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
#from collections import Counter

# custome optimizer and model
from custome_optimizer.qpso_optimizer import QDPSOptimizer
from ann.ann import ExtendedModel

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the iris dataset from scikit-learn
data = load_iris()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset: 80% for training and validation, 20% for test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

kf = KFold(n_splits=4, shuffle=True, random_state=100)

input_dim = X.shape[1]
output_dim = len(torch.unique(torch.tensor(y)))
hidden_layers = [input_dim * 3]

n_particles = 20
max_iters = 100
g = 1.13
interval_parms_update = 10

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

test_results = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train_val)):
    print(f"Fold {fold + 1}")

    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    model = ExtendedModel(input_dim, output_dim, hidden_layers).to(device)

    y_train_one_hot = torch.nn.functional.one_hot(y_train, num_classes=output_dim).float().to(device)
    y_val_one_hot = torch.nn.functional.one_hot(y_val, num_classes=output_dim).float().to(device)

    bounds = [(-1, 1) for _ in range(sum(p.numel() for p in model.parameters()))]
    optimizer = QDPSOptimizer(model, bounds, n_particles=n_particles, max_iters=max_iters, g=g, interval_parms_update=interval_parms_update)

    optimizer.set_training_data(X_train, y_train_one_hot)

    for epoch in range(4):
        print(f"Starting epoch {epoch + 1}")
        optimizer.step()
        output = model(X_train)
        loss = model.lf.cross_entropy(y_train_one_hot, output, model)

        with torch.no_grad():
            val_output = model(X_val)
            val_loss = model.lf.cross_entropy(y_val_one_hot, val_output, model)
        
        print(f'Fold {fold + 1}, Epoch {epoch + 1}, Loss: {loss.item()} - Validation Loss: {val_loss.item()}')
        print(f"Finished epoch {epoch + 1}")

    # Final evaluation on validation set
    model._set_params(optimizer.best_params)

    with torch.no_grad():
        val_output = model(X_val)
        val_loss = model.lf.cross_entropy(y_val_one_hot, val_output, model)
        print(f'Fold {fold + 1}, Final Validation Loss: {val_loss.item()}')

        # Evaluación en el conjunto de prueba
        y_pred = model(X_test).argmax(dim=1)
        accuracy = (y_pred == y_test).float().mean().item()
        print(f'Fold {fold + 1}, Accuracy on iris test dataset: {accuracy:.4f}')
        test_results.append(accuracy)

# Resultados finales
print(f'Mean accuracy on test dataset: {torch.tensor(test_results).mean().item():.4f}')
print(f'Standard deviation of accuracy on test dataset: {torch.tensor(test_results).std().item():.4f}')
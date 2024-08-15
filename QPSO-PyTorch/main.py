import torch
from tensor_qpso.qpso import QDPSO  # Asegúrate de importar correctamente

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definimos la función de costo usando PyTorch
def sphere(args):
    return torch.sum(args ** 2).item()

def log(s):
    # Extraemos los valores best_value de los partículas como tensores en GPU
    best_values = torch.tensor([p.best_value for p in s.particles()], device=device)
    
    # Calculamos la media y la desviación estándar usando PyTorch
    best_value_avg = torch.mean(best_values).item()
    best_value_std = torch.std(best_values).item()
    
    # Imprimimos los resultados
    print("{0: >5}  {1: >9}  {2: >9}  {3: >9}".format("Iters.", "Best", "Best(Mean)", "Best(STD)"))
    print("{0: >5}  {1: >9.3E}  {2: >9.3E}  {3: >9.3E}".format(s.iters, s.gbest_value.item(), best_value_avg, best_value_std))

NParticle = 40
MaxIters = 100
NDim = 10
bounds = [(-2.56, 5.12) for i in range(NDim)]
g = 0.96

# Inicializamos QDPSO con la función de costo, tamaño de partícula, dimensiones, límites y otros parámetros
s = QDPSO(sphere, NParticle, NDim, bounds, MaxIters, g)

# Ejecutamos la actualización del QDPSO con un callback para el log
s.update(callback=log, interval=10)
print("Found best position: {0}".format(s.gbest))

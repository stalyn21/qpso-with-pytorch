from tensor_qpso.qpso_tensor import get_device

# Uso
device = get_device('auto')
print(device)  # cuda o cpu segun disponibilidad

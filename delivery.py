import numpy as np

def step_function(x):
    return 1 if x >= 0 else -1

# Dados sobre as entregas
# [distância, pico, chuva]
X = np.array([
    [0, 0, 0],  # perto, sem pico, sem chuva
    [1, 1, 1],  # longe, pico, chuva
    [1, 0, 0],  # longe, sem pico, sem chuva
    [0, 1, 0],  # perto, pico
    [0, 0, 1],  # perto, chuva
    [1, 1, 0],  # longe, pico
])

# Saídas desejadas: 1 = rápido, -1 = lento
y = np.array([1, -1, -1, 1, 1, -1])

lr = 0.3
epochs = 10

# Pesos iniciais
pesos = np.array([0.2, -0.1, 0.3], dtype=float)
bias = 0.0

print("Pesos iniciais:", pesos, "Bias inicial:", bias)

# Treinamento
for ep in range(epochs):
    print(f"\nÉpoca {ep+1}")
    for x_i, y_i in zip(X, y):
        soma = np.dot(pesos, x_i) + bias
        y_pred = step_function(soma)
        erro = y_i - y_pred

        pesos += lr * erro * x_i
        bias += lr * erro

        print(f"Entrada: {x_i} | Esperado: {y_i} | Previsto: {y_pred} | Erro: {erro}")
        print("Pesos:", pesos, "| Bias:", bias)

print("\nPesos finais:", pesos, "Bias final:", bias)

# Testes
X_test = np.array([
    [0, 0, 0],  # condição ideal → rápido
    [1, 0, 1],  # longe + chuva → provavelmente lento
    [0, 1, 1],  # perto, mas chuva + pico → pode atrasar
    [1, 0, 0],  # longe → tende a ser mais lento
])

print("\n--- Testes de previsão ---")
for x in X_test:
    y_pred = step_function(np.dot(pesos, x) + bias)
    print(f"Entrada {x} → Previsto: {y_pred}")

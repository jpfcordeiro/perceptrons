import numpy as np

def step_function(x):
    return 1 if x >= 0 else -1

# Entradas da cozinha
# [complexidade, fila, ingrediente especial]
X = np.array([
    [0, 0, 0],  # prato simples, fila curta, sem especial
    [1, 1, 1],  # prato difícil, fila longa, ingrediente especial
    [1, 0, 0],  # prato difícil, fila curta
    [0, 1, 0],  # simples, mas fila longa
    [0, 0, 1],  # simples, ingrediente especial
    [1, 1, 0],  # difícil + fila longa
])

# Saída:  1 = rápido, -1 = demorado
y = np.array([1, -1, -1, -1, 1, -1])

lr = 0.3
epochs = 10

# Pesos iniciais
pesos = np.array([0.2, -0.3, 0.1], dtype=float)
bias = 0.1

print("Pesos iniciais:", pesos, "Bias inicial:", bias)

# -----------------------------
# Treinamento do Perceptron
# -----------------------------
for ep in range(epochs):
    print(f"\nÉpoca {ep+1}")
    for x_i, y_i in zip(X, y):
        soma = np.dot(pesos, x_i) + bias
        y_pred = step_function(soma)
        erro = y_i - y_pred

        # Atualização das variáveis
        pesos += lr * erro * x_i
        bias += lr * erro

        print(f"Entrada: {x_i} | Esperado: {y_i} | Prev: {y_pred} | Erro: {erro}")
        print("Pesos:", pesos, "| Bias:", bias)

print("\nPesos finais:", pesos, "Bias final:", bias)

# -----------------------------
# Teste com novos pedidos
# -----------------------------
X_test = np.array([
    [0, 0, 0],  # prato simples → rápido
    [1, 0, 1],  # difícil + especial → mais lento
    [0, 1, 1],  # simples mas com fila + especial
    [1, 1, 0],  # difícil e fila longa
])

print("\n--- Testes de previsão ---")
for x in X_test:
    y_pred = step_function(np.dot(pesos, x) + bias)
    print(f"Pedido {x} → Previsto: {y_pred}")

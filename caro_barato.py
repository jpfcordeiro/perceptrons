import numpy as np

def step_function(x):
    return 1 if x >= 0 else -1

# Dados
X = np.array([
    [3, 1, 1],  # combo grande premium
    [1, 0, 0],  # batata pequena
    [2, 1, 0],  # hambúrguer premium pequeno
    [0, 0, 0],  # água
    [1, 0, 1],  # refrigerante grande
])

y = np.array([1, -1, 1, -1, -1])

lr = 0.2
epochs = 12

pesos = np.array([0.1, 0.3, -0.4])
bias = 0.0

print("Pesos iniciais:", pesos, "Bias:", bias)

for ep in range(epochs):
    print(f"\nÉpoca {ep+1}")
    for x_i, y_i in zip(X, y):
        soma = np.dot(pesos, x_i) + bias
        y_pred = step_function(soma)
        erro = y_i - y_pred

        pesos += lr * erro * x_i
        bias += lr * erro

        print(f"{x_i} -> Previsto: {y_pred} | Esperado:{y_i} | Erro:{erro}")

print("\nPesos finais:", pesos, "Bias final:", bias)

# Teste
print("\n--- Testes ---")
X_test = np.array([
    [3, 1, 0],  # hambúrguer premium médio
    [1, 0, 1],  # refrigerante grande
    [0, 0, 0],  # água
])
for x in X_test:
    print(x, "=>", step_function(np.dot(pesos, x) + bias))

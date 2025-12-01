import numpy as np

def step_function(x):
    return 1 if x >= 0 else -1

# Dados de alimentos de fast food
X = np.array([
    [1, 1, 0],  # sorvete + batata
    [0, 1, 0],  # refrigerante
    [1, 0, 0],  # hambúrguer
    [0, 0, 1],  # salada fitness
    [0, 0, 1],  # wrap leve
])

y = np.array([-1, -1, -1, 1, 1])

lr = 0.3
epochs = 10

pesos = np.array([0.4, -0.3, 0.2])
bias = 0.1

print("Pesos iniciais:", pesos, "Bias inicial:", bias)

for ep in range(epochs):
    print(f"\nÉpoca {ep+1}")
    for x_i, y_i in zip(X, y):
        soma = np.dot(pesos, x_i) + bias
        y_pred = step_function(soma)
        erro = y_i - y_pred

        pesos += lr * erro * x_i
        bias += lr * erro

        print(f"Entrada: {x_i} | Esperado: {y_i} | Previsto: {y_pred} | Erro: {erro}")

print("\nPesos finais:", pesos, "Bias final:", bias)

# Teste
X_test = np.array([
    [1, 1, 0],  # milkshake + fritas -> não saudável
    [0, 0, 1],  # salada verde -> saudável
    [1, 0, 1],  # hambúrguer light -> saudável?
])

print("\n--- Testes ---")
for x in X_test:
    print(x, "->", step_function(np.dot(pesos, x) + bias))

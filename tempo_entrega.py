import numpy as np

def step_function(x):
    return 1 if x >= 0 else -1

# Entradas: [complexidade, quantidade_itens, tempo_pedido]
# onde tempo_pedido = 1 significa horário de pico (ex: almoço/jantar)
X = np.array([
    [0, 0, 0],  # simples, poucos itens, cedo -> rápido
    [1, 1, 1],  # complexo, muitos itens, pico -> demorado
    [1, 0, 0],  # complexo, poucos itens, cedo -> pode ser lento
    [0, 1, 1],  # simples, muitos itens, pico -> pode atrasar
    [0, 0, 1],  # simples, cedo, mas pico -> ainda rápido
    [1, 1, 0],  # complexo, muitos itens, cedo -> demorado
])

# Saída: 1 = rápido, -1 = demorado
y = np.array([1, -1, -1, -1, 1, -1])

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

        # Atualiza pesos e bias
        pesos += lr * erro * x_i
        bias += lr * erro

        print(f"Entrada: {x_i} | Esperado: {y_i} | Previsto: {y_pred} | Erro: {erro}")
        print(f"Pesos: {pesos} | Bias: {bias}")

print("\nPesos finais:", pesos, "Bias final:", bias)

# Testes com novos pedidos
X_test = np.array([
    [0, 1, 0],  # simples, muitos itens, cedo
    [1, 0, 1],  # complexo, poucos itens, pico
    [0, 0, 0],  # simples, cedo, poucos itens
    [1, 1, 1],  # complexo, muitos itens, pico
])

print("\n--- Testes de previsão ---")
for x in X_test:
    y_pred = step_function(np.dot(pesos, x) + bias)
    print(f"Pedido {x} → Previsto: {y_pred}")

import numpy as np

# Função de ativação (degrau bipolar: -1 ou +1)
def step_function(x):
    return 1 if x >= 0 else -1

# Dados do problema (Pratique 01)
X = np.array([
    
    [0, 0, 1],
    [1, 1, 0]
])
y = np.array([-1, 1])  # saídas esperadas

# Hiperparâmetros
lr = 0.4
n_epochs = 10

# Pesos iniciais (inclui bias separado)
pesos = np.array([0.4, -0.6, 0.6], dtype=float)
bias = 0.5

print("Pesos iniciais:", pesos, "Bias inicial:", bias)

# Treinamento
for epoca in range(n_epochs):
    print(f"\nÉpoca {epoca+1}")
    for x_i, y_i in zip(X, y):
        soma = np.dot(pesos, x_i) + bias
        y_pred = step_function(soma)
        erro = y_i - y_pred
        
        # Atualização dos pesos e bias
        pesos += lr * erro * x_i
        bias += lr * erro
        
        print(f"Entrada: {x_i}, Esperado: {y_i}, Previsto: {y_pred}, Erro: {erro}")
        print("Novos pesos:", pesos, "Novo bias:", bias)

print("\nPesos finais:", pesos, "Bias final:", bias)

# Predição em novos exemplos
X_teste = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 1]
])

Y_teste = np.array([-1, 1,1,-1])

print("\n--- Predições ---")
acertou = 0
exemplo_teste = 0
for x_i, y_i in zip(X_teste, Y_teste):
    soma = np.dot(pesos, x_i) + bias
    y_pred = step_function(soma)
    print(f"Entrada: {x_i} -> Saída prevista: {y_pred}")
    erro = y_i - y_pred
    if erro == 0:
        acertou +=1
    exemplo_teste+=1

acuracia = acertou/exemplo_teste
print(f"Acuracia: {acuracia}")
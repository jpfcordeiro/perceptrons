# Mini-tutorial: Implementando um Perceptron (básico)

Este mini-tutorial mostra os conceitos mínimos e um exemplo em Python para treinar um perceptron simples (classificador linear binário) usando NumPy. O exemplo é baseado no arquivo `caro_barato.py` deste repositório.

**O que é um perceptron?**
- Um perceptron é um classificador linear simples que faz uma combinação linear dos atributos de entrada, aplica uma função de ativação (no exemplo, a step function) e produz uma saída binária (aqui usamos +1 e -1).

**Elementos principais**
- Pesos (`w`) e bias (`b`).
- Função de ativação: `step(x) = +1 se x >= 0, caso contrário -1`.
- Regra de atualização (treinamento):
  - Para cada amostra x, com rótulo y em {+1, -1}:
    - y_pred = step(w . x + b)
    - erro = y - y_pred
    - w <- w + lr * erro * x
    - b <- b + lr * erro

Este é o algoritmo de perceptron clássico (perceptron learning rule).

## Código de exemplo

O repositório já contém um script chamado `caro_barato.py` com um exemplo completo. Abaixo segue uma versão comentada e organizada que implementa as mesmas ideias.

```python
import numpy as np

def step_function(x):
    return 1 if x >= 0 else -1

def train_perceptron(X, y, lr=0.2, epochs=12, pesos_init=None, bias_init=0.0):
    n_features = X.shape[1]
    if pesos_init is None:
        pesos = np.zeros(n_features)
    else:
        pesos = np.array(pesos_init, dtype=float)
    bias = float(bias_init)

    for ep in range(epochs):
        for x_i, y_i in zip(X, y):
            soma = np.dot(pesos, x_i) + bias
            y_pred = step_function(soma)
            erro = y_i - y_pred
            pesos += lr * erro * x_i
            bias += lr * erro
    return pesos, bias

def predict(X, pesos, bias):
    return np.array([step_function(np.dot(pesos, x) + bias) for x in X])

if __name__ == "__main__":
    # Dados (exemplo do repositório)
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
    pesos_init = [0.1, 0.3, -0.4]
    bias_init = 0.0

    pesos, bias = train_perceptron(X, y, lr=lr, epochs=epochs, pesos_init=pesos_init, bias_init=bias_init)

    print("Pesos finais:", pesos)
    print("Bias final:", bias)

    X_test = np.array([
        [3, 1, 0],
        [1, 0, 1],
        [0, 0, 0],
    ])
    print("Predições nos testes:", predict(X_test, pesos, bias))
```

## Como executar
- Certifique-se de ter o Python 3 e o NumPy instalados.

Instale NumPy (se necessário):

```bash
pip install numpy
```

Execute o script existente:

```bash
python3 caro_barato.py
```

Ou salve a versão acima como `perceptron_example.py` e execute:

```bash
python3 perceptron_example.py
```

## Interpretação dos resultados
- `pesos` e `bias` são ajustados para reduzir os erros nas amostras de treinamento.
- A função `step_function` produz apenas +1 ou -1; por isso perceptrons não são probabilísticos.
- Se os dados não são linearmente separáveis, o perceptron pode não convergir (os pesos podem oscilar).

## Dicas e variações
- Normalizar/escala dos atributos pode ajudar na convergência.
- Use diferentes taxas de aprendizado (`lr`) e número de épocas (`epochs`) para ver o efeito.
- Para problemas com mais de duas classes, considere "one-vs-rest" (treinar um perceptron por classe) ou usar modelos mais complexos (por ex., redes neurais multicamadas).
- Para saídas probabilísticas, use regressão logística (sigmoid + cross-entropy).

## Exercícios sugeridos
- Modifique o código para usar rótulos {0,1} em vez de {-1,+1} e adapte a regra de atualização.
- Adicione uma função de avaliação (acurácia) para acompanhar o progresso por época.
- Teste com ruído nos dados e observe o comportamento do perceptron.

---
Arquivo criado: `TUTORIAL_PERCEPTRON.md` (baseado em `caro_barato.py`).

## Template genérico (arquivo: `perceptron_template.py`)

Abaixo está um template Python completo e prático que segue os passos do mini passo a passo: preparação, split, normalização, treinamento, avaliação e salvamento do modelo. O arquivo `perceptron_template.py` também foi adicionado ao repositório.

```python
# Trechos principais (veja o arquivo `perceptron_template.py` no repositório):
import numpy as np
def step_function(x):
    return 1 if x >= 0 else -1

def preprocess_standard(X):
    # centraliza e escala por desvio padrão
    X = np.array(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    return Xs, {"method": "standard", "mean": mean.tolist(), "std": std.tolist()}

def split_train_test(X, y, test_size=0.2, random_state=None):
    # embaralha e divide
    n = len(X)
    idx = np.arange(n)
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        rng.shuffle(idx)
    else:
        np.random.shuffle(idx)
    split = int(n * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

def train_perceptron(X, y, lr=0.1, epochs=100, weights_init=None, bias_init=0.0):
    n_features = X.shape[1]
    w = np.zeros(n_features) if weights_init is None else np.array(weights_init, dtype=float)
    b = float(bias_init)
    for ep in range(epochs):
        errors = 0
        perm = np.arange(len(X))
        np.random.shuffle(perm)
        for i in perm:
            xi = X[i]
            yi = y[i]
            s = np.dot(w, xi) + b
            y_pred = 1 if s >= 0 else -1
            err = yi - y_pred
            if err != 0:
                w += lr * err * xi
                b += lr * err
                errors += 1
        if errors == 0:
            break
    return w, b

def predict(X, w, b):
    s = np.dot(X, w) + b
    return np.where(s >= 0, 1, -1)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

```

Como executar o template:

```bash
pip install numpy
python3 perceptron_template.py
```

O script executa um exemplo com os dados do repositório, treina um perceptron simples, imprime a acurácia no conjunto de teste e salva o modelo em `perceptron_model.npz`.

---

Arquivo `perceptron_template.py` criado no repositório.

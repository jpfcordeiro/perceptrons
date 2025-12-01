#!/usr/bin/env python3
"""
Template genérico de Perceptron (numpy-only).
Siga os passos: preparação, inicialização, treinamento, avaliação e salvamento.
"""

import numpy as np
import json


def step_function(x):
    return 1 if x >= 0 else -1


def preprocess_standard(X):
    X = np.array(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    metadata = {"method": "standard", "mean": mean.tolist(), "std": std.tolist()}
    return Xs, metadata


def split_train_test(X, y, test_size=0.2, random_state=None):
    n = len(X)
    idx = np.arange(n)
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        rng.shuffle(idx)
    else:
        np.random.shuffle(idx)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_perceptron(X, y, lr=0.1, epochs=100, weights_init=None, bias_init=0.0, verbose=False):
    n_features = X.shape[1]
    if weights_init is None:
        w = np.zeros(n_features, dtype=float)
    else:
        w = np.array(weights_init, dtype=float)
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
        if verbose:
            print(f"Epoch {ep+1}/{epochs} - errors: {errors}")
        if errors == 0:
            break
    return w, b


def predict(X, w, b):
    s = np.dot(X, w) + b
    return np.where(s >= 0, 1, -1)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def save_model(path, w, b, preproc_meta=None):
    np.savez(path, w=w, b=b, preproc=json.dumps(preproc_meta or {}))


def load_model(path):
    d = np.load(path, allow_pickle=True)
    w = d['w']
    b = float(d['b'])
    preproc = {}
    if 'preproc' in d:
        try:
            preproc = json.loads(d['preproc'].tolist())
        except Exception:
            preproc = {}
    return w, b, preproc


if __name__ == '__main__':
    # Exemplo com os mesmos dados do repositório
    X = np.array([
        [3, 1, 1],  # combo grande premium
        [1, 0, 0],  # batata pequena
        [2, 1, 0],  # hambúrguer premium pequeno
        [0, 0, 0],  # água
        [1, 0, 1],  # refrigerante grande
    ])
    y = np.array([1, -1, 1, -1, -1])

    # 1) Pré-processamento
    Xs, preproc_meta = preprocess_standard(X)

    # 2) Split treino/teste
    X_train, X_test, y_train, y_test = split_train_test(Xs, y, test_size=0.4, random_state=42)

    # 3) Treinamento
    w, b = train_perceptron(X_train, y_train, lr=0.2, epochs=50, verbose=True)

    # 4) Avaliação
    y_pred = predict(X_test, w, b)
    print("Test accuracy:", accuracy(y_test, y_pred))

    # 5) Salvar modelo
    save_model('perceptron_model.npz', w, b, preproc_meta)
    print('Modelo salvo em perceptron_model.npz')

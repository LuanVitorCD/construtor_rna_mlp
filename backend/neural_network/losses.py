import numpy as np
from backend.neural_network.network import NeuralNetwork


def mse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Erro Quadrático Médio (MSE) sobre amostras e saídas."""
    diff = np.asarray(Y_true, dtype=float) - np.asarray(Y_pred, dtype=float)
    return float(np.mean(diff ** 2))


def evaluate_mse(network: NeuralNetwork, X: np.ndarray, Y: np.ndarray) -> float:
    """Avalia o MSE da rede em um conjunto de dados sem alterar seus pesos."""
    preds = np.array([network.forward(x) for x in X], dtype=float)
    return mse(Y, preds)
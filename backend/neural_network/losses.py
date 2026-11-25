"""
Funções de custo (perda) para avaliação quantitativa do desempenho da rede neural.

Implementa o Erro Quadrático Médio (MSE) utilizado no monitoramento de convergência
durante o treinamento e na avaliação final dos modelos.
"""

import numpy as np
from backend.neural_network.network import NeuralNetwork


def mse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Calcula o Erro Quadrático Médio (MSE): (1 / (N * K)) * sum((y - y_hat)^2)."""
    diff = np.asarray(Y_true, dtype=float) - np.asarray(Y_pred, dtype=float)
    return float(np.mean(diff ** 2))


def evaluate_mse(network: NeuralNetwork, X: np.ndarray, Y: np.ndarray) -> float:
    """Executa inferência sobre X e calcula o MSE em relação aos alvos Y sem modificar pesos."""
    preds = np.array([network.forward(x) for x in X], dtype=float)
    return mse(Y, preds)
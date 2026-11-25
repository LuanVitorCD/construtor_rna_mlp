"""
Implementação da unidade computacional básica (neurônio artificial) do modelo MLP.

Armazena pesos sinápticos, viés (bias), potencial de ativação (z), sinal de saída
e o gradiente local (delta) utilizado durante a retropropagação do erro.
"""

import numpy as np
from backend.neural_network.activations import Activation


class Neuron:
    """Representa um neurônio artificial com vetor de pesos, viés e estado interno do último passo."""

    def __init__(self, weights: np.ndarray, bias: float, activation: Activation):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)
        self.activation = activation

        # Estados internos armazenados durante o forward para uso no cálculo de gradientes (backward)
        self.z: float = 0.0       # Potencial de ativação (soma ponderada): z = w^T x + b
        self.output: float = 0.0  # Sinal funcional de saída: a = f(z)
        self.delta: float = 0.0   # Gradiente local (sensibilidade): delta = dE / dz

    def forward(self, inputs: np.ndarray) -> float:
        """Calcula o potencial de ativação z e aplica a função de ativação f(z)."""
        self.z = float(np.dot(self.weights, inputs) + self.bias)
        self.output = float(self.activation.fn(self.z))
        return self.output
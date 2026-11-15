import numpy as np
from backend.activations import Activation


class Neuron:
    """Representa um neurônio artificial com pesos, viés e estado do último passo."""

    def __init__(self, weights: np.ndarray, bias: float, activation: Activation):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)
        self.activation = activation

        # Valores intermediários mantidos para o cálculo dos gradientes no backward
        self.z: float = 0.0
        self.output: float = 0.0
        self.delta: float = 0.0  # Sensibilidade local: dE / dz

    def forward(self, inputs: np.ndarray) -> float:
        # Soma ponderada z = w · x + b seguida da não-linearidade a = f(z)
        self.z = float(np.dot(self.weights, inputs) + self.bias)
        self.output = float(self.activation.fn(self.z))
        return self.output
"""
Implementação de uma camada totalmente conectada (densa) da rede MLP.

Agrupa um conjunto de neurônios que compartilham o mesmo vetor de entrada e
a mesma função de ativação, gerenciando sua propagação direta.
"""

import numpy as np
from numpy.random import Generator
from backend.neural_network.activations import get_activation
from backend.errors import ConfigError
from backend.neural_network.initializers import init_layer_params
from backend.neural_network.neuron import Neuron


class Layer:
    """Camada densa composta por múltiplos neurônios conectados a todas as entradas anteriores."""

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        activation_name: str,
        rng: Generator,
    ):
        if n_inputs < 1:
            raise ConfigError("A camada precisa receber ao menos 1 entrada.")
        if n_neurons < 1:
            raise ConfigError("A camada precisa ter ao menos 1 neurônio.")

        self.n_inputs = int(n_inputs)
        self.n_neurons = int(n_neurons)
        self.activation_name = activation_name

        activation = get_activation(activation_name)
        W, b = init_layer_params(self.n_inputs, self.n_neurons, activation_name, rng)

        self.neurons = [
            Neuron(W[j].copy(), float(b[j]), activation)
            for j in range(self.n_neurons)
        ]
        self.last_inputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Executa o passo forward em todos os neurônios da camada e retém o vetor de entrada."""
        # Retém o vetor de entrada x^(l-1) para o cálculo do gradiente dE/dw_ji = delta_j * x_i no backward
        self.last_inputs = np.asarray(inputs, dtype=float)
        return np.array([n.forward(self.last_inputs) for n in self.neurons], dtype=float)

    def weight_matrix(self) -> np.ndarray:
        """Retorna a matriz sináptica W de dimensão (n_neurons, n_inputs) com os pesos atuais."""
        return np.array([n.weights for n in self.neurons], dtype=float)

    def set_parameters(self, weights: list | np.ndarray, biases: list | np.ndarray) -> None:
        """Sobrescreve pesos e vieses dos neurônios (utilizado na desserialização de modelos salvos)."""
        for neuron, w, b in zip(self.neurons, weights, biases):
            neuron.weights = np.array(w, dtype=float)
            neuron.bias = float(b)
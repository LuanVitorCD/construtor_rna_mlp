import numpy as np
from numpy.random import Generator
from backend.neural_network.activations import get_activation
from backend.errors import ConfigError
from backend.neural_network.initializers import init_layer_params
from backend.neural_network.neuron import Neuron


class Layer:
    """Agrupa neurônios que compartilham o mesmo vetor de entrada e função de ativação."""

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
        # Armazena as entradas desta camada para multiplicar pelo delta na atualização de pesos
        self.last_inputs = np.asarray(inputs, dtype=float)
        return np.array([n.forward(self.last_inputs) for n in self.neurons], dtype=float)

    def weight_matrix(self) -> np.ndarray:
        """Retorna matriz W de formato (n_neurons, n_inputs) com os pesos atuais."""
        return np.array([n.weights for n in self.neurons], dtype=float)

    def set_parameters(self, weights: list | np.ndarray, biases: list | np.ndarray) -> None:
        """Sobrescreve pesos e vieses (usado ao restaurar modelos ou reproduzir exemplos)."""
        for neuron, w, b in zip(self.neurons, weights, biases):
            neuron.weights = np.array(w, dtype=float)
            neuron.bias = float(b)
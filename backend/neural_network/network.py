"""
Núcleo matemático da arquitetura Multilayer Perceptron (MLP).

Implementa a composição dinâmica de camadas densas, a propagação direta (forward)
e o algoritmo de retropropagação do erro (backpropagation) com atualização SGD.
"""

import numpy as np
from backend.errors import ConfigError, DatasetError
from backend.neural_network.layer import Layer


class NeuralNetwork:
    """Rede Multilayer Perceptron (MLP) com número configurável de camadas ocultas e neurônios."""

    def __init__(self, n_inputs: int, seed: int | None = None):
        if n_inputs < 1:
            raise ConfigError("A rede deve possuir ao menos 1 entrada.")
        self.n_inputs = int(n_inputs)
        self.layers: list[Layer] = []
        self.seed = seed
        # Gerador único garante reprodutibilidade determinística na inicialização entre camadas
        self.rng = np.random.default_rng(seed)

    def add_layer(self, n_neurons: int, activation_name: str) -> None:
        """Adiciona uma nova camada densa conectada à saída da camada anterior."""
        n_in = self.n_inputs if not self.layers else self.layers[-1].n_neurons
        self.layers.append(Layer(n_in, n_neurons, activation_name, self.rng))

    @classmethod
    def from_config(
        cls,
        n_inputs: int,
        layer_specs: list[dict],
        seed: int | None = None,
    ) -> "NeuralNetwork":
        """Constrói a topologia completa da rede a partir de uma lista de especificações de camadas."""
        if not layer_specs:
            raise ConfigError("A rede precisa de ao menos 1 camada.")
        net = cls(n_inputs, seed=seed)
        for spec in layer_specs:
            net.add_layer(spec["neurons"], spec["activation"])
        return net

    @property
    def n_outputs(self) -> int:
        if not self.layers:
            raise ConfigError("A rede ainda não possui camadas.")
        return self.layers[-1].n_neurons

    def forward(self, x: list[float] | np.ndarray) -> np.ndarray:
        """Executa a propagação direta (forward pass) do vetor de entrada x até a camada de saída."""
        if not self.layers:
            raise ConfigError("A rede não possui camadas para executar forward.")
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim != 1 or len(x_arr) != self.n_inputs:
            raise DatasetError(
                f"Entrada incompatível: a rede espera {self.n_inputs} valores, recebeu {len(x_arr)}."
            )
        current = x_arr
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def backward(self, y_true: list[float] | np.ndarray, lr: float) -> None:
        """Calcula os gradientes locais via regra da cadeia e atualiza pesos e vieses (1 passo SGD)."""
        y_arr = np.asarray(y_true, dtype=float)
        out_layer = self.layers[-1]
        if len(y_arr) != out_layer.n_neurons:
            raise DatasetError(
                f"Saída esperada incompatível: esperava {out_layer.n_neurons} valores, recebeu {len(y_arr)}."
            )

        # Etapa 1: Gradiente local (delta) na camada de saída (L)
        # Perda quadrática do exemplo: E = (1/K) * sum((y_k - y_hat_k)^2)
        # Derivada em relação à saída ativada: dE/da_k = (2/K) * (y_hat_k - y_k)
        # Gradiente local: delta_k^(L) = (dE/da_k) * f'(z_k^(L))
        n_out = out_layer.n_neurons
        for k, neuron in enumerate(out_layer.neurons):
            dE_da = (2.0 / n_out) * (neuron.output - y_arr[k])
            neuron.delta = dE_da * neuron.activation.dfn(neuron.z, neuron.output)

        # Etapa 2: Retropropagação dos deltas para as camadas ocultas (l = L-1 até 1)
        # Utiliza os pesos ANTES da atualização: delta_j^(l) = (sum_k w_kj^(l+1) * delta_k^(l+1)) * f'(z_j^(l))
        for l in reversed(range(len(self.layers) - 1)):
            next_layer = self.layers[l + 1]
            W_next = next_layer.weight_matrix()
            deltas_next = np.array([m.delta for m in next_layer.neurons], dtype=float)

            for j, neuron in enumerate(self.layers[l].neurons):
                propagated_error = float(np.dot(W_next[:, j], deltas_next))
                neuron.delta = propagated_error * neuron.activation.dfn(neuron.z, neuron.output)

        # Etapa 3: Atualização dos parâmetros via Descida do Gradiente Estocástica (SGD)
        # Gradientes: dE/dw_ji = delta_j * x_i  e  dE/db_j = delta_j
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.weights -= lr * neuron.delta * layer.last_inputs
                neuron.bias -= lr * neuron.delta
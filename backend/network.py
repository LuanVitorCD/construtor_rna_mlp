import numpy as np
from backend.errors import ConfigError, DatasetError
from backend.layer import Layer


class NeuralNetwork:
    """Rede Multilayer Perceptron (MLP) com composição dinâmica de camadas."""

    def __init__(self, n_inputs: int, seed: int | None = None):
        if n_inputs < 1:
            raise ConfigError("A rede deve possuir ao menos 1 entrada.")
        self.n_inputs = int(n_inputs)
        self.layers: list[Layer] = []
        self.seed = seed
        # Um único gerador garante sequência determinística entre camadas sucessivas
        self.rng = np.random.default_rng(seed)

    def add_layer(self, n_neurons: int, activation_name: str) -> None:
        n_in = self.n_inputs if not self.layers else self.layers[-1].n_neurons
        self.layers.append(Layer(n_in, n_neurons, activation_name, self.rng))

    @classmethod
    def from_config(
        cls,
        n_inputs: int,
        layer_specs: list[dict],
        seed: int | None = None,
    ) -> "NeuralNetwork":
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
        """Propaga o vetor de entrada x camada a camada até a saída."""
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
        """Calcula deltas via regra da cadeia e atualiza parâmetros por SGD (1 exemplo)."""
        y_arr = np.asarray(y_true, dtype=float)
        out_layer = self.layers[-1]
        if len(y_arr) != out_layer.n_neurons:
            raise DatasetError(
                f"Saída esperada incompatível: esperava {out_layer.n_neurons} valores, recebeu {len(y_arr)}."
            )

        # 1) Delta da camada de saída:
        # Perda do exemplo E = (1/K) * sum((y_k - y_hat_k)^2) -> dE/dy_hat_k = (2/K) * (y_hat_k - y_k)
        n_out = out_layer.n_neurons
        for k, neuron in enumerate(out_layer.neurons):
            dE_da = (2.0 / n_out) * (neuron.output - y_arr[k])
            neuron.delta = dE_da * neuron.activation.dfn(neuron.z, neuron.output)

        # 2) Deltas das camadas ocultas (de trás para frente, usando pesos ANTES da atualização):
        # delta_j^(l) = (sum_k w_kj^(l+1) * delta_k^(l+1)) * f'(z_j^(l))
        for l in reversed(range(len(self.layers) - 1)):
            next_layer = self.layers[l + 1]
            W_next = next_layer.weight_matrix()
            deltas_next = np.array([m.delta for m in next_layer.neurons], dtype=float)

            for j, neuron in enumerate(self.layers[l].neurons):
                propagated_error = float(np.dot(W_next[:, j], deltas_next))
                neuron.delta = propagated_error * neuron.activation.dfn(neuron.z, neuron.output)

        # 3) Atualização dos parâmetros via descida do gradiente:
        # dE/dw_ji = delta_j * x_i  e  dE/db_j = delta_j
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.weights -= lr * neuron.delta * layer.last_inputs
                neuron.bias -= lr * neuron.delta
from typing import Callable
import numpy as np
from backend.neural_network.network import NeuralNetwork


def snapshot_params(network: NeuralNetwork) -> list[list[tuple[np.ndarray, float]]]:
    """Cópia profunda de todos os pesos e vieses da rede."""
    return [
        [(n.weights.copy(), float(n.bias)) for n in layer.neurons]
        for layer in network.layers
    ]


def params_iguais(snap_a: list, snap_b: list) -> bool:
    """Verifica igualdade exata entre dois snapshots de parâmetros."""
    if len(snap_a) != len(snap_b):
        return False
    for layer_a, layer_b in zip(snap_a, snap_b):
        if len(layer_a) != len(layer_b):
            return False
        for (w_a, b_a), (w_b, b_b) in zip(layer_a, layer_b):
            if not np.array_equal(w_a, w_b) or b_a != b_b:
                return False
    return True


def numeric_gradients(
    network: NeuralNetwork,
    loss_fn: Callable[[], float],
    eps: float = 1e-6,
) -> dict[tuple[int, int], tuple[np.ndarray, float]]:
    """Calcula gradientes por diferenças finitas centrais: (L(p+eps) - L(p-eps)) / 2eps."""
    grads: dict[tuple[int, int], tuple[np.ndarray, float]] = {}

    for l, layer in enumerate(network.layers):
        for j, neuron in enumerate(layer.neurons):
            gw = np.zeros_like(neuron.weights, dtype=float)
            for i in range(len(neuron.weights)):
                original = float(neuron.weights[i])
                neuron.weights[i] = original + eps
                lp = loss_fn()
                neuron.weights[i] = original - eps
                lm = loss_fn()
                neuron.weights[i] = original
                gw[i] = (lp - lm) / (2.0 * eps)

            orig_bias = float(neuron.bias)
            neuron.bias = orig_bias + eps
            lp_b = loss_fn()
            neuron.bias = orig_bias - eps
            lm_b = loss_fn()
            neuron.bias = orig_bias
            gb = float((lp_b - lm_b) / (2.0 * eps))

            grads[(l, j)] = (gw, gb)

    return grads
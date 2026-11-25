import numpy as np
from numpy.random import Generator


def init_layer_params(
    n_in: int,
    n_out: int,
    activation_name: str,
    rng: Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Inicializa pesos com He (ReLU) ou Xavier/Glorot (demais ativações) e vieses em zero."""
    if activation_name == "relu":
        # He Normal: desvio padrão sqrt(2 / n_in) preserva variância em camadas ReLU
        std = float(np.sqrt(2.0 / n_in))
        weights = rng.normal(0.0, std, size=(n_out, n_in))
    else:
        # Xavier/Glorot Uniforme: intervalo [-sqrt(6 / (n_in + n_out)), +sqrt(6 / (n_in + n_out))]
        limit = float(np.sqrt(6.0 / (n_in + n_out)))
        weights = rng.uniform(-limit, limit, size=(n_out, n_in))

    biases = np.zeros(n_out, dtype=float)
    return weights, biases
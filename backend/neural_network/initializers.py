"""
Estratégias de inicialização de pesos sinápticos e vieses para camadas densas.

Implementa as inicializações de He (Kaiming) e Xavier (Glorot) para estabilizar
a variância das ativações e gradientes entre camadas sucessivas.
"""

import numpy as np
from numpy.random import Generator


def init_layer_params(
    n_in: int,
    n_out: int,
    activation_name: str,
    rng: Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Inicializa matriz de pesos com He (ReLU) ou Xavier/Glorot (demais) e vieses nulos."""
    if activation_name == "relu":
        # Inicialização He Normal: N(0, sqrt(2 / n_in)) compensa a nulidade de metade dos sinais na ReLU
        std = float(np.sqrt(2.0 / n_in))
        weights = rng.normal(0.0, std, size=(n_out, n_in))
    else:
        # Inicialização Xavier/Glorot Uniforme: U[-sqrt(6 / (n_in + n_out)), +sqrt(6 / (n_in + n_out))]
        limit = float(np.sqrt(6.0 / (n_in + n_out)))
        weights = rng.uniform(-limit, limit, size=(n_out, n_in))

    # Vieses inicializados em zero para quebra de simetria guiada pelos pesos aleatórios
    biases = np.zeros(n_out, dtype=float)
    return weights, biases
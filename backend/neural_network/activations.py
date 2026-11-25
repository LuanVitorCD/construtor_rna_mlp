"""
Catálogo de funções de ativação diferenciáveis e suas respectivas derivadas analíticas.

Fornece as não-linearidades necessárias para a capacidade de aproximação universal
das camadas ocultas e o mapeamento adequado da camada de saída.
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np
from backend.errors import ConfigError


@dataclass(frozen=True)
class Activation:
    """Encapsula uma função de ativação f(z) e sua derivada f'(z) expressa em função de z e a = f(z)."""
    name: str
    fn: Callable[[float], float]
    dfn: Callable[[float, float], float]


def _sigmoid(z: float) -> float:
    """Calcula a função logística sigmoide com limitação de domínio para evitar overflow numérico."""
    z_safe = float(np.clip(z, -500.0, 500.0))
    return float(1.0 / (1.0 + np.exp(-z_safe)))


ACTIVATIONS: dict[str, Activation] = {
    "linear": Activation(
        name="linear",
        fn=lambda z: float(z),
        # Derivada constante unitária: f'(z) = 1
        dfn=lambda z, a: 1.0,
    ),
    "sigmoid": Activation(
        name="sigmoid",
        fn=_sigmoid,
        # Derivada analítica da sigmoide em função da própria saída: f'(z) = a * (1 - a)
        dfn=lambda z, a: float(a * (1.0 - a)),
    ),
    "tanh": Activation(
        name="tanh",
        fn=lambda z: float(np.tanh(z)),
        # Derivada analítica da tangente hiperbólica: f'(z) = 1 - a^2
        dfn=lambda z, a: float(1.0 - a * a),
    ),
    "relu": Activation(
        name="relu",
        fn=lambda z: float(max(0.0, z)),
        # Subgradiente da unidade linear retificada: 1 para z > 0, 0 caso contrário
        dfn=lambda z, a: 1.0 if z > 0.0 else 0.0,
    ),
}


def get_activation(name: str) -> Activation:
    """Recupera o par (função, derivada) registrado pelo identificador textual."""
    if name not in ACTIVATIONS:
        raise ConfigError(f"Função de ativação desconhecida: '{name}'.")
    return ACTIVATIONS[name]


def list_activations() -> list[str]:
    """Retorna a lista de identificadores de funções de ativação suportadas."""
    return list(ACTIVATIONS.keys())
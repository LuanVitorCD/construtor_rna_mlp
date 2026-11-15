from dataclasses import dataclass
from typing import Callable
import numpy as np
from backend.errors import ConfigError


@dataclass(frozen=True)
class Activation:
    name: str
    fn: Callable[[float], float]
    # dfn recebe z (soma ponderada) e a = f(z) (saída já calculada no forward)
    dfn: Callable[[float, float], float]


def _sigmoid(z: float) -> float:
    # Clip evita overflow numérico em np.exp para valores extremos de z
    z_safe = float(np.clip(z, -500.0, 500.0))
    return float(1.0 / (1.0 + np.exp(-z_safe)))


ACTIVATIONS: dict[str, Activation] = {
    "linear": Activation(
        name="linear",
        fn=lambda z: float(z),
        dfn=lambda z, a: 1.0,
    ),
    "sigmoid": Activation(
        name="sigmoid",
        fn=_sigmoid,
        # Derivada da sigmoide em função da própria saída: f'(z) = a * (1 - a)
        dfn=lambda z, a: float(a * (1.0 - a)),
    ),
    "tanh": Activation(
        name="tanh",
        fn=lambda z: float(np.tanh(z)),
        # Derivada da tangente hiperbólica: f'(z) = 1 - a²
        dfn=lambda z, a: float(1.0 - a * a),
    ),
    "relu": Activation(
        name="relu",
        fn=lambda z: float(max(0.0, z)),
        dfn=lambda z, a: 1.0 if z > 0.0 else 0.0,
    ),
}


def get_activation(name: str) -> Activation:
    if name not in ACTIVATIONS:
        raise ConfigError(f"Função de ativação desconhecida: '{name}'.")
    return ACTIVATIONS[name]


def list_activations() -> list[str]:
    return list(ACTIVATIONS.keys())
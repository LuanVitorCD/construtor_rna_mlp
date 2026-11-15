from dataclasses import dataclass
import json
from pathlib import Path
from backend.errors import ConfigError, PersistenceError
from backend.network import NeuralNetwork
from backend.normalization import Normalizer

FORMAT_VERSION = 1


@dataclass
class LoadedModel:
    network: NeuralNetwork
    normalizer: Normalizer | None
    task: str
    meta: dict


def save_model(
    path: str | Path,
    network: NeuralNetwork,
    normalizer: Normalizer | None,
    task: str = "regression",
    meta: dict | None = None,
) -> None:
    """Serializa arquitetura, pesos, vieses e parâmetros de normalização em JSON."""
    payload = {
        "format_version": FORMAT_VERSION,
        "task": task,
        "n_inputs": network.n_inputs,
        "layers": [
            {
                "n_neurons": layer.n_neurons,
                "activation": layer.activation_name,
                "neurons": [
                    {
                        "weights": neuron.weights.tolist(),
                        "bias": float(neuron.bias),
                    }
                    for neuron in layer.neurons
                ],
            }
            for layer in network.layers
        ],
        "normalization": normalizer.to_dict() if normalizer is not None else None,
        "meta": meta or {},
    }

    file_path = Path(path)
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise PersistenceError(f"Falha ao gravar arquivo do modelo: {exc}") from exc


def load_model(path: str | Path) -> LoadedModel:
    """Reconstrói a rede neural e o normalizador a partir de um arquivo JSON validado."""
    file_path = Path(path)
    if not file_path.exists():
        raise PersistenceError(f"Arquivo de modelo não encontrado: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistenceError(f"Arquivo JSON inválido ou corrompido: {exc}") from exc

    if not isinstance(data, dict):
        raise PersistenceError("Estrutura raiz do JSON inválida.")

    if data.get("format_version") != FORMAT_VERSION:
        raise PersistenceError(f"Versão de formato não suportada: {data.get('format_version')}")

    required_keys = {"n_inputs", "layers", "normalization"}
    if not required_keys.issubset(data.keys()):
        raise PersistenceError(f"Campos obrigatórios ausentes no JSON: {required_keys - set(data.keys())}")

    if not isinstance(data["layers"], list) or len(data["layers"]) == 0:
        raise PersistenceError("O modelo deve conter ao menos uma camada válida.")

    try:
        net = NeuralNetwork(int(data["n_inputs"]))
        expected_inputs = net.n_inputs

        for layer_data in data["layers"]:
            n_neurons = int(layer_data["n_neurons"])
            activation = layer_data["activation"]
            neurons_data = layer_data["neurons"]

            if len(neurons_data) != n_neurons:
                raise PersistenceError("Contagem de neurônios inconsistente com 'n_neurons'.")

            for n_data in neurons_data:
                if len(n_data["weights"]) != expected_inputs:
                    raise PersistenceError(
                        f"Dimensão de pesos inválida: esperava {expected_inputs}, "
                        f"encontrou {len(n_data['weights'])}."
                    )

            net.add_layer(n_neurons, activation)
            net.layers[-1].set_parameters(
                weights=[n["weights"] for n in neurons_data],
                biases=[n["bias"] for n in neurons_data],
            )
            expected_inputs = n_neurons

        normalizer = (
            Normalizer.from_dict(data["normalization"])
            if data["normalization"] is not None
            else None
        )
    except (KeyError, TypeError, ValueError, ConfigError) as exc:
        raise PersistenceError(f"Erro ao reconstruir modelo a partir do JSON: {exc}") from exc

    return LoadedModel(
        network=net,
        normalizer=normalizer,
        task=data.get("task", "regression"),
        meta=data.get("meta") or {},
    )
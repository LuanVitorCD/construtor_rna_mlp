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
    y_normalizer: Normalizer | None = None


def save_model(
    path: str | Path,
    network: NeuralNetwork,
    normalizer: Normalizer | None,
    task: str = "regression",
    meta: dict | None = None,
    y_normalizer: Normalizer | None = None,
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
        "y_normalization": y_normalizer.to_dict() if y_normalizer is not None else None,
        "meta": meta or {},
    }

    file_path = Path(path)
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise PersistenceError(f"Falha ao gravar arquivo do modelo: {exc}") from exc


def load_model(path: str | Path) -> LoadedModel:
    """Reconstrói a rede neural e os normalizadores a partir de um arquivo JSON validado."""
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
        n_inputs = int(data["n_inputs"])
        if n_inputs < 1:
            raise PersistenceError("O número de entradas do modelo deve ser >= 1.")

        net = NeuralNetwork(n_inputs)
        expected_inputs = net.n_inputs

        for layer_data in data["layers"]:
            n_neurons = int(layer_data["n_neurons"])
            activation = layer_data["activation"]
            neurons_data = layer_data["neurons"]

            if len(neurons_data) != n_neurons:
                raise PersistenceError("Contagem de neurônios inconsistente com 'n_neurons'.")

            weights_list = []
            biases_list = []
            for n_data in neurons_data:
                w = [float(val) for val in n_data["weights"]]
                b = float(n_data["bias"])
                if len(w) != expected_inputs:
                    raise PersistenceError(
                        f"Dimensão de pesos inválida: esperava {expected_inputs}, "
                        f"encontrou {len(w)}."
                    )
                if not all(val == val and abs(val) != float("inf") for val in w + [b]):
                    raise PersistenceError("Pesos e vieses do modelo devem ser numéricos finitos.")
                weights_list.append(w)
                biases_list.append(b)

            net.add_layer(n_neurons, activation)
            net.layers[-1].set_parameters(weights=weights_list, biases=biases_list)
            expected_inputs = n_neurons

        normalizer = (
            Normalizer.from_dict(data["normalization"])
            if data["normalization"] is not None
            else None
        )
        y_normalizer = (
            Normalizer.from_dict(data["y_normalization"])
            if data.get("y_normalization") is not None
            else None
        )

        if normalizer is not None:
            for param_arr in normalizer.params.values():
                if param_arr.shape != (net.n_inputs,) or not all(
                    float(v) == float(v) and abs(float(v)) != float("inf") for v in param_arr
                ):
                    raise PersistenceError("Parâmetros de normalização de entrada incompatíveis.")

        if y_normalizer is not None:
            for param_arr in y_normalizer.params.values():
                if param_arr.shape != (net.n_outputs,) or not all(
                    float(v) == float(v) and abs(float(v)) != float("inf") for v in param_arr
                ):
                    raise PersistenceError("Parâmetros de normalização de saída incompatíveis.")
    except PersistenceError:
        raise
    except (KeyError, TypeError, ValueError, ConfigError) as exc:
        raise PersistenceError(f"Erro ao reconstruir modelo a partir do JSON: {exc}") from exc

    return LoadedModel(
        network=net,
        normalizer=normalizer,
        task=data.get("task", "regression"),
        meta=data.get("meta") or {},
        y_normalizer=y_normalizer,
    )
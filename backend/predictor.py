import numpy as np
from backend.classification import LabelEncoder
from backend.errors import DatasetError
from backend.persistence import LoadedModel


def predict(model: LoadedModel, values: list[float] | np.ndarray) -> dict:
    """Executa inferência (apenas forward), aplicando a normalização do treino se existir."""
    if len(values) != model.network.n_inputs:
        raise DatasetError(
            f"Quantidade de entradas inválida: a rede espera {model.network.n_inputs}, "
            f"mas recebeu {len(values)}."
        )

    x = np.asarray(values, dtype=float)
    if model.normalizer is not None:
        # Normalizer espera matriz 2D (1, n_inputs)
        x = model.normalizer.transform(x[None, :])[0]

    outputs = model.network.forward(x)
    result: dict = {"outputs": [float(v) for v in outputs]}

    if model.task == "classification" and model.meta.get("class_labels"):
        encoder = LabelEncoder.from_classes(model.meta["class_labels"])
        result["label"] = encoder.decode(outputs)

    return result
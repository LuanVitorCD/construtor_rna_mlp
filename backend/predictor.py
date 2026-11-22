import numpy as np
from backend.classification import LabelEncoder
from backend.errors import DatasetError
from backend.persistence import LoadedModel


def predict(model: LoadedModel, values: list[float] | np.ndarray) -> dict:
    """Executa inferência (apenas forward), normalizando entradas e desnormalizando saídas."""
    try:
        x = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DatasetError("Os valores de entrada devem ser numéricos.") from exc

    if x.ndim != 1 or len(x) != model.network.n_inputs:
        raise DatasetError(
            f"Quantidade de entradas inválida: a rede espera {model.network.n_inputs}, "
            f"mas recebeu {x.shape[0] if x.ndim == 1 else x.shape}."
        )

    if not np.isfinite(x).all():
        raise DatasetError("Os valores de entrada devem ser números finitos (sem NaN ou Inf).")

    if model.normalizer is not None:
        # Normalizer espera matriz 2D (1, n_inputs)
        x = model.normalizer.transform(x[None, :])[0]

    outputs = model.network.forward(x)

    if model.y_normalizer is not None:
        outputs = model.y_normalizer.inverse_transform(outputs[None, :])[0]

    result: dict = {"outputs": [float(v) for v in outputs]}

    if model.task == "classification" and model.meta.get("class_labels"):
        encoder = LabelEncoder.from_classes(model.meta["class_labels"])
        result["label"] = encoder.decode(outputs)

    return result
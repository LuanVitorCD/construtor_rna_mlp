import numpy as np
from backend.dataset import Dataset, DataSplit
from backend.errors import ConfigError


def _safe_denominator(values: np.ndarray) -> np.ndarray:
    """Substitui zeros por 1.0 para evitar NaN/Inf em colunas constantes."""
    return np.where(values == 0.0, 1.0, values)


class Normalizer:
    """Normaliza e desnormaliza dados usando estatísticas extraídas apenas do treino."""

    def __init__(self, kind: str):
        if kind not in ("minmax", "zscore"):
            raise ConfigError(f"Tipo de normalização desconhecido: '{kind}'.")
        self.kind = kind
        self.params: dict[str, np.ndarray] = {}

    def fit(self, data: np.ndarray) -> "Normalizer":
        arr = np.asarray(data, dtype=float)
        if self.kind == "minmax":
            self.params = {
                "min": arr.min(axis=0),
                "max": arr.max(axis=0),
            }
        elif self.kind == "zscore":
            self.params = {
                "mean": arr.mean(axis=0),
                "std": arr.std(axis=0),
            }
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if self.kind == "minmax":
            denom = _safe_denominator(self.params["max"] - self.params["min"])
            return (arr - self.params["min"]) / denom
        # zscore: (x - média) / desvio_padrão
        denom = _safe_denominator(self.params["std"])
        return (arr - self.params["mean"]) / denom

    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """Reverte a normalização para retornar os valores à escala original."""
        arr = np.asarray(normalized_data, dtype=float)
        if self.kind == "minmax":
            denom = _safe_denominator(self.params["max"] - self.params["min"])
            return arr * denom + self.params["min"]
        # zscore: x_norm * desvio_padrão + média
        denom = _safe_denominator(self.params["std"])
        return arr * denom + self.params["mean"]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "params": {k: v.tolist() for k, v in self.params.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Normalizer":
        norm = cls(data["kind"])
        norm.params = {k: np.array(v, dtype=float) for k, v in data["params"].items()}
        return norm


def _transform_subset(
    dataset: Dataset,
    x_norm: Normalizer,
    y_norm: Normalizer | None,
) -> Dataset:
    return Dataset(
        X=x_norm.transform(dataset.X),
        Y=y_norm.transform(dataset.Y) if y_norm is not None else dataset.Y.copy(),
        input_cols=list(dataset.input_cols),
        output_cols=list(dataset.output_cols),
    )


def apply_normalization(
    data_split: DataSplit,
    kind: str | None,
    normalize_y: bool = False,
) -> tuple[DataSplit, Normalizer | None, Normalizer | None]:
    """Ajusta os normalizadores apenas no treino e aplica nos três subconjuntos."""
    if kind is None:
        return data_split, None, None

    x_norm = Normalizer(kind).fit(data_split.train.X)
    y_norm = Normalizer(kind).fit(data_split.train.Y) if normalize_y else None

    normalized_split = DataSplit(
        train=_transform_subset(data_split.train, x_norm, y_norm),
        val=_transform_subset(data_split.val, x_norm, y_norm),
        test=_transform_subset(data_split.test, x_norm, y_norm),
    )
    return normalized_split, x_norm, y_norm
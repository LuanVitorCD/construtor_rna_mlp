import numpy as np
from backend.dataset import Dataset, DataSplit
from backend.errors import ConfigError


def _safe_denominator(values: np.ndarray) -> np.ndarray:
    """Substitui zeros por 1.0 para evitar NaN/Inf em colunas constantes."""
    return np.where(values == 0.0, 1.0, values)


class Normalizer:
    """Normaliza atributos de entrada usando estatísticas extraídas apenas do treino."""

    def __init__(self, kind: str):
        if kind not in ("minmax", "zscore"):
            raise ConfigError(f"Tipo de normalização desconhecido: '{kind}'.")
        self.kind = kind
        self.params: dict[str, np.ndarray] = {}

    def fit(self, X_train: np.ndarray) -> "Normalizer":
        X_arr = np.asarray(X_train, dtype=float)
        if self.kind == "minmax":
            self.params = {
                "min": X_arr.min(axis=0),
                "max": X_arr.max(axis=0),
            }
        elif self.kind == "zscore":
            self.params = {
                "mean": X_arr.mean(axis=0),
                "std": X_arr.std(axis=0),
            }
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if self.kind == "minmax":
            denom = _safe_denominator(self.params["max"] - self.params["min"])
            return (X_arr - self.params["min"]) / denom
        # zscore: (x - média) / desvio_padrão
        denom = _safe_denominator(self.params["std"])
        return (X_arr - self.params["mean"]) / denom

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


def _with_X(dataset: Dataset, new_X: np.ndarray) -> Dataset:
    return Dataset(
        X=new_X,
        Y=dataset.Y.copy(),
        input_cols=list(dataset.input_cols),
        output_cols=list(dataset.output_cols),
    )


def apply_normalization(
    data_split: DataSplit,
    kind: str | None,
) -> tuple[DataSplit, Normalizer | None]:
    """Ajusta o normalizador apenas no treino e aplica nos três subconjuntos (Y intacto)."""
    if kind is None:
        return data_split, None

    norm = Normalizer(kind).fit(data_split.train.X)
    normalized_split = DataSplit(
        train=_with_X(data_split.train, norm.transform(data_split.train.X)),
        val=_with_X(data_split.val, norm.transform(data_split.val.X)),
        test=_with_X(data_split.test, norm.transform(data_split.test.X)),
    )
    return normalized_split, norm
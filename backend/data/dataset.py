"""
Módulo de leitura, validação dimensional e particionamento de conjuntos de dados CSV.

Gerencia a extração de tensores de entrada (X) e saída desejada (Y), além da
divisão estratificada/aleatória em subconjuntos disjuntos de treino, validação e teste.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from backend.errors import DatasetError
from backend.neural_network.network import NeuralNetwork


@dataclass
class Dataset:
    """Encapsula matrizes de atributos (X) e alvos (Y) com os metadados de suas colunas."""
    X: np.ndarray
    Y: np.ndarray
    input_cols: list[str]
    output_cols: list[str]

    def __len__(self) -> int:
        return len(self.X)


@dataclass
class DataSplit:
    """Agrupa as três partições disjuntas exigidas para treinamento e avaliação da MLP."""
    train: Dataset
    val: Dataset
    test: Dataset


def read_columns(path: str | Path) -> list[str]:
    """Lê apenas o cabeçalho do arquivo CSV para mapeamento das colunas na interface."""
    file_path = Path(path)
    if not file_path.exists():
        raise DatasetError(f"Arquivo CSV não encontrado: {file_path}")
    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if not header:
        raise DatasetError("O arquivo CSV está vazio.")
    return [col.strip() for col in header]


def load_csv(
    path: str | Path,
    input_cols: list[str],
    output_cols: list[str],
    raw_outputs: bool = False,
) -> Dataset:
    """Carrega matrizes de entrada (X) e saída desejada (Y) a partir de um arquivo CSV."""
    if not input_cols or not output_cols:
        raise DatasetError("Selecione ao menos uma coluna de entrada e uma de saída.")

    overlap = set(input_cols) & set(output_cols)
    if overlap:
        raise DatasetError(f"Colunas não podem ser entrada e saída ao mesmo tempo: {sorted(overlap)}")

    file_path = Path(path)
    if not file_path.exists():
        raise DatasetError(f"Arquivo CSV não encontrado: {file_path}")

    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise DatasetError("O arquivo CSV não possui cabeçalho.")
        available = set(reader.fieldnames)

        missing = [c for c in input_cols + output_cols if c not in available]
        if missing:
            raise DatasetError(f"Coluna(s) inexistente(s) no CSV: {missing}")

        rows = list(reader)

    if not rows:
        raise DatasetError("O arquivo CSV não contém linhas de dados.")

    try:
        X = np.array([[float(row[c]) for c in input_cols] for row in rows], dtype=float)
    except ValueError as exc:
        raise DatasetError("As colunas de entrada devem conter apenas valores numéricos.") from exc

    # Para classificação, preserva rótulos textuais (strings) para codificação posterior
    if raw_outputs:
        Y = np.array([[row[c].strip() for c in output_cols] for row in rows], dtype=object)
    else:
        try:
            Y = np.array([[float(row[c]) for c in output_cols] for row in rows], dtype=float)
        except ValueError as exc:
            raise DatasetError("As colunas de saída devem conter valores numéricos para regressão.") from exc

    return Dataset(X=X, Y=Y, input_cols=list(input_cols), output_cols=list(output_cols))


def check_compatibility(dataset: Dataset, network: NeuralNetwork) -> None:
    """Valida se as dimensões de entrada e saída do dataset coincidem com a topologia da rede."""
    if dataset.X.shape[1] != network.n_inputs:
        raise DatasetError(
            f"Incompatibilidade de entradas: dataset tem {dataset.X.shape[1]} colunas, "
            f"mas a rede espera {network.n_inputs}."
        )
    if dataset.Y.shape[1] != network.n_outputs:
        raise DatasetError(
            f"Incompatibilidade de saídas: dataset exige {dataset.Y.shape[1]} saídas, "
            f"mas a camada de saída da rede tem {network.n_outputs} neurônios."
        )


def split(
    dataset: Dataset,
    train_pct: float,
    val_pct: float,
    test_pct: float,
    seed: int | None = None,
) -> DataSplit:
    """Particiona aleatoriamente o dataset em treino, validação e teste sem sobreposição de amostras."""
    if train_pct < 0 or val_pct < 0 or test_pct < 0:
        raise DatasetError("Os percentuais de divisão não podem ser negativos.")
    if abs((train_pct + val_pct + test_pct) - 100.0) > 1e-6:
        raise DatasetError("A soma dos percentuais de treino, validação e teste deve ser 100%.")

    n_total = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)

    n_train = round(n_total * (train_pct / 100.0))
    n_val = round(n_total * (val_pct / 100.0))
    # O restante é atribuído ao conjunto de teste para evitar perda de amostras por arredondamento
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise DatasetError("Os subconjuntos de treino, validação e teste não podem ficar vazios.")

    def _subset(idx: np.ndarray) -> Dataset:
        return Dataset(
            X=dataset.X[idx].copy(),
            Y=dataset.Y[idx].copy(),
            input_cols=list(dataset.input_cols),
            output_cols=list(dataset.output_cols),
        )

    return DataSplit(
        train=_subset(train_idx),
        val=_subset(val_idx),
        test=_subset(test_idx),
    )
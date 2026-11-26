"""
Utilitários de codificação de rótulos e métricas de avaliação para tarefas de classificação.

Suporta problemas binários (saída escalar com limiar 0.5) e multiclasse
(codificação One-Hot com decisão via argmax), além de acurácia e matriz de confusão.
"""

import numpy as np
from backend.errors import DatasetError


class LabelEncoder:
    """Codifica rótulos categóricos em 0/1 (binário) ou vetores One-Hot (multiclasse)."""

    def __init__(self):
        self.classes_: list[str] = []

    def fit(self, labels: list | np.ndarray) -> "LabelEncoder":
        """Identifica e ordena o vocabulário único de classes presentes no conjunto de dados."""
        flat = [str(lbl) for lbl in np.asarray(labels).ravel()]
        self.classes_ = sorted(set(flat))
        if len(self.classes_) < 2:
            raise DatasetError("A classificação requer ao menos 2 classes distintas.")
        return self

    @classmethod
    def from_classes(cls, classes: list[str]) -> "LabelEncoder":
        """Reconstrói o codificador a partir da lista ordenada de classes salva nos metadados."""
        enc = cls()
        enc.classes_ = [str(c) for c in classes]
        return enc

    @property
    def n_outputs(self) -> int:
        # Classificação binária utiliza 1 neurônio de saída; multiclasse utiliza C neurônios (One-Hot)
        return 1 if len(self.classes_) == 2 else len(self.classes_)

    def encode(self, labels: list | np.ndarray) -> np.ndarray:
        """Converte rótulos textuais em alvos numéricos compatíveis com a camada de saída da MLP."""
        flat = [str(lbl) for lbl in np.asarray(labels).ravel()]
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for lbl in flat:
            if lbl not in class_to_idx:
                raise DatasetError(f"Rótulo desconhecido encontrado: '{lbl}'.")

        if len(self.classes_) == 2:
            return np.array([[float(class_to_idx[lbl])] for lbl in flat], dtype=float)

        encoded = np.zeros((len(flat), len(self.classes_)), dtype=float)
        for row_idx, lbl in enumerate(flat):
            encoded[row_idx, class_to_idx[lbl]] = 1.0
        return encoded

    def decode(self, outputs: list[float] | np.ndarray) -> str:
        """Mapeia o vetor contínuo de saída da rede de volta para o rótulo categórico previsto."""
        arr = np.asarray(outputs, dtype=float).ravel()
        if len(self.classes_) == 2:
            return self.classes_[1] if arr[0] >= 0.5 else self.classes_[0]
        return self.classes_[int(np.argmax(arr))]


def accuracy(y_true_labels: list[str], y_pred_labels: list[str]) -> float:
    """Calcula a taxa de acerto global (acurácia) entre rótulos reais e previstos."""
    if len(y_true_labels) != len(y_pred_labels):
        raise ValueError("A quantidade de rótulos verdadeiros e previstos deve ser igual.")
    if not y_true_labels:
        return 0.0
    hits = sum(1 for yt, yp in zip(y_true_labels, y_pred_labels) if yt == yp)
    return float(hits / len(y_true_labels))


def confusion_matrix(
    y_true_labels: list[str],
    y_pred_labels: list[str],
    classes: list[str],
) -> np.ndarray:
    """Constrói a matriz de confusão C x C (linhas = classe real, colunas = classe prevista)."""
    if len(y_true_labels) != len(y_pred_labels):
        raise ValueError("A quantidade de rótulos verdadeiros e previstos deve ser igual.")
    idx_map = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for yt, yp in zip(y_true_labels, y_pred_labels):
        cm[idx_map[yt], idx_map[yp]] += 1
    return cm
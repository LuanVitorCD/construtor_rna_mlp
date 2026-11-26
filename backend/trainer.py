"""
Controlador do loop de treinamento via Descida do Gradiente Estocástica (SGD).

Gerencia o embaralhamento das amostras por época, a atualização online dos pesos,
o cálculo do MSE de treino/validação e a interrupção segura via sinalização de thread.
"""

from dataclasses import dataclass
import threading
from typing import Callable
import numpy as np
from backend.data.dataset import Dataset
from backend.errors import ConfigError, DatasetError
from backend.neural_network.losses import mse
from backend.neural_network.network import NeuralNetwork
from backend.data.normalization import Normalizer


@dataclass
class EpochResult:
    """Registro das métricas de perda (MSE) consolidadas ao final de uma época de treinamento."""
    epoch: int
    train_mse: float
    val_mse: float


class Trainer:
    """Executa o treinamento da rede MLP via Stochastic Gradient Descent (atualização exemplo a exemplo)."""

    def __init__(
        self,
        network: NeuralNetwork,
        train: Dataset,
        val: Dataset,
        lr: float,
        epochs: int,
        shuffle_seed: int | None = None,
        y_normalizer: Normalizer | None = None,
    ):
        if lr <= 0.0 or epochs < 1:
            raise ConfigError("Taxa de aprendizado deve ser > 0 e épocas >= 1.")
        if len(train) == 0 or len(val) == 0:
            raise DatasetError("Os conjuntos de treino e validação não podem ser vazios.")

        self.network = network
        self.train = train
        self.val = val
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.shuffle_seed = shuffle_seed
        self.y_normalizer = y_normalizer
        self._stop_event = threading.Event()
        self.history: list[EpochResult] = []

    def stop(self) -> None:
        """Sinaliza interrupção cooperativa e segura do treinamento entre iterações."""
        self._stop_event.set()

    def _compute_display_mse(self, dataset: Dataset) -> float:
        """Calcula o MSE sobre um subconjunto, revertendo a escala de Y caso esteja normalizada."""
        preds = np.array([self.network.forward(x) for x in dataset.X], dtype=float)
        if self.y_normalizer is not None:
            y_true = self.y_normalizer.inverse_transform(dataset.Y)
            y_pred = self.y_normalizer.inverse_transform(preds)
            return mse(y_true, y_pred)
        return mse(dataset.Y, preds)

    def run(self, on_epoch: Callable[[EpochResult], None] | None = None) -> list[EpochResult]:
        """Executa o ciclo completo de épocas, emitindo o histórico para atualização da interface."""
        rng = np.random.default_rng(self.shuffle_seed)
        n_samples = len(self.train)

        for epoch in range(1, self.epochs + 1):
            if self._stop_event.is_set():
                break

            # Embaralha a ordem de apresentação dos padrões a cada época para evitar ciclos no SGD
            for idx in rng.permutation(n_samples):
                if self._stop_event.is_set():
                    break
                self.network.forward(self.train.X[idx])
                self.network.backward(self.train.Y[idx], self.lr)

            if self._stop_event.is_set():
                break

            # Avalia o erro quadrático médio (MSE) de treino e validação ao término da época
            train_mse = self._compute_display_mse(self.train)
            val_mse = self._compute_display_mse(self.val)
            result = EpochResult(epoch=epoch, train_mse=train_mse, val_mse=val_mse)
            self.history.append(result)

            if on_epoch is not None:
                on_epoch(result)

        return self.history
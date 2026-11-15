from dataclasses import asdict
from pathlib import Path
import threading
from typing import Callable
import numpy as np
from backend.activations import list_activations
from backend.classification import LabelEncoder, accuracy, confusion_matrix
from backend.dataset import Dataset, DataSplit, check_compatibility, load_csv, read_columns, split
from backend.errors import ConfigError, DatasetError, StateError
from backend.losses import evaluate_mse
from backend.network import NeuralNetwork
from backend.normalization import Normalizer, apply_normalization
from backend.persistence import LoadedModel, load_model, save_model
from backend.predictor import predict
from backend.trainer import EpochResult, Trainer


class MLPService:
    """Fachada única consumida pela UI (Flet). Retorna apenas tipos nativos serializáveis."""

    def __init__(self):
        self.network: NeuralNetwork | None = None
        self.normalizer: Normalizer | None = None
        self.norm_kind: str | None = None
        self.task: str = "regression"

        self.raw_dataset: Dataset | None = None
        self.split_raw: DataSplit | None = None
        self.split: DataSplit | None = None

        self.encoder: LabelEncoder | None = None
        self.trainer: Trainer | None = None
        self._train_thread: threading.Thread | None = None
        self.history: list[dict] = []
        self.meta: dict = {}

    # ---------- Arquitetura ----------

    def list_activations(self) -> list[str]:
        return list_activations()

    def create_network(
        self,
        n_inputs: int,
        layers: list[dict],
        seed: int | None = None,
        task: str = "regression",
    ) -> dict:
        if task not in ("regression", "classification"):
            raise ConfigError(f"Tipo de tarefa inválido: '{task}'.")

        self.network = NeuralNetwork.from_config(n_inputs, layers, seed=seed)
        self.task = task
        # Reinicia estado de dados/treino ao criar uma nova arquitetura
        self.raw_dataset = None
        self.split_raw = None
        self.split = None
        self.normalizer = None
        self.norm_kind = None
        self.encoder = None
        self.history = []
        self.meta = {}
        return self.describe_network()

    def describe_network(self) -> dict:
        if self.network is None:
            raise StateError("Nenhuma rede neural foi criada ou carregada.")

        layers_info = []
        total_params = 0
        for idx, layer in enumerate(self.network.layers):
            params = layer.n_neurons * (layer.n_inputs + 1)  # pesos + 1 viés por neurônio
            total_params += params
            layers_info.append({
                "index": idx,
                "n_inputs": int(layer.n_inputs),
                "n_neurons": int(layer.n_neurons),
                "activation": layer.activation_name,
                "parameters": int(params),
            })

        return {
            "task": self.task,
            "n_inputs": int(self.network.n_inputs),
            "n_outputs": int(self.network.n_outputs),
            "total_parameters": int(total_params),
            "layers": layers_info,
        }

    # ---------- Dados e Pré-processamento ----------

    def get_csv_columns(self, path: str | Path) -> list[str]:
        return read_columns(path)

    def load_dataset(
        self,
        path: str | Path,
        input_cols: list[str],
        output_cols: list[str],
    ) -> dict:
        if self.network is None:
            raise StateError("Crie ou carregue uma rede neural antes de carregar o dataset.")

        is_classification = self.task == "classification"
        if is_classification and len(output_cols) != 1:
            raise DatasetError("Para classificação, selecione exatamente 1 coluna de rótulo de saída.")

        ds = load_csv(path, input_cols, output_cols, raw_outputs=is_classification)

        if is_classification:
            self.encoder = LabelEncoder().fit(ds.Y)
            encoded_Y = self.encoder.encode(ds.Y)
            ds = Dataset(X=ds.X, Y=encoded_Y, input_cols=ds.input_cols, output_cols=ds.output_cols)
            self.meta["class_labels"] = list(self.encoder.classes_)
        else:
            self.encoder = None
            self.meta["class_labels"] = None

        check_compatibility(ds, self.network)

        self.raw_dataset = ds
        self.split_raw = None
        self.split = None
        self.normalizer = None
        self.norm_kind = None
        self.history = []
        self.meta["input_cols"] = list(input_cols)
        self.meta["output_cols"] = list(output_cols)

        preview_rows = min(5, len(ds))
        return {
            "rows": int(len(ds)),
            "inputs": list(input_cols),
            "outputs": list(output_cols),
            "preview": [
                {
                    "x": [float(v) for v in ds.X[i]],
                    "y": [float(v) for v in ds.Y[i]],
                }
                for i in range(preview_rows)
            ],
        }

    def split_dataset(
        self,
        train_pct: float,
        val_pct: float,
        test_pct: float,
        seed: int | None = None,
    ) -> dict:
        if self.raw_dataset is None:
            raise StateError("Carregue um dataset antes de realizar a divisão.")

        self.split_raw = split(self.raw_dataset, train_pct, val_pct, test_pct, seed=seed)
        # Reaplica normalização caso já estivesse configurada
        self.split, self.normalizer = apply_normalization(self.split_raw, self.norm_kind)

        return {
            "train": int(len(self.split.train)),
            "val": int(len(self.split.val)),
            "test": int(len(self.split.test)),
        }

    def set_normalization(self, kind: str | None) -> dict:
        if self.split_raw is None:
            raise StateError("Divida o dataset antes de configurar a normalização.")

        # Aplica sempre a partir de split_raw para evitar normalizar dados já normalizados
        self.split, self.normalizer = apply_normalization(self.split_raw, kind)
        self.norm_kind = kind
        return {"normalization": kind}

    # ---------- Treinamento ----------

    def start_training(
        self,
        epochs: int,
        lr: float,
        shuffle_seed: int | None = None,
        on_epoch: Callable[[dict], None] | None = None,
        on_finish: Callable[[], None] | None = None,
    ) -> None:
        if self.network is None:
            raise StateError("Nenhuma rede neural configurada para treinar.")
        if self.split is None:
            raise StateError("Carregue e divida o dataset antes de iniciar o treinamento.")
        if self._train_thread is not None and self._train_thread.is_alive():
            raise StateError("Já existe um treinamento em andamento.")

        self.history = []
        self.trainer = Trainer(
            network=self.network,
            train=self.split.train,
            val=self.split.val,
            lr=lr,
            epochs=epochs,
            shuffle_seed=shuffle_seed,
        )

        def _handle_epoch(res: EpochResult) -> None:
            record = asdict(res)
            self.history.append(record)
            if on_epoch is not None:
                on_epoch(record)

        def _worker() -> None:
            try:
                self.trainer.run(on_epoch=_handle_epoch)
            finally:
                if on_finish is not None:
                    on_finish()

        self._train_thread = threading.Thread(target=_worker, daemon=True)
        self._train_thread.start()

    def stop_training(self) -> None:
        if self.trainer is not None:
            self.trainer.stop()

    def get_history(self) -> list[dict]:
        return list(self.history)

    # ---------- Avaliação e Previsão ----------

    def evaluate(self, subset: str = "test") -> dict:
        if self.network is None:
            raise StateError("Nenhuma rede neural disponível para avaliação.")
        if self.split is None:
            raise StateError("Divida o dataset antes de avaliar.")

        subsets = {"train": self.split.train, "val": self.split.val, "test": self.split.test}
        if subset not in subsets:
            raise ConfigError(f"Subconjunto inválido: '{subset}'. Use 'train', 'val' ou 'test'.")

        target_ds = subsets[subset]
        if len(target_ds) == 0:
            raise DatasetError(f"O subconjunto '{subset}' está vazio.")

        mse_val = evaluate_mse(self.network, target_ds.X, target_ds.Y)
        result: dict = {"subset": subset, "mse": float(mse_val)}

        if self.task == "classification" and self.encoder is not None:
            y_true_labels = [self.encoder.decode(y) for y in target_ds.Y]
            y_pred_labels = [self.encoder.decode(self.network.forward(x)) for x in target_ds.X]
            cm = confusion_matrix(y_true_labels, y_pred_labels, self.encoder.classes_)
            result["accuracy"] = float(accuracy(y_true_labels, y_pred_labels))
            result["classes"] = list(self.encoder.classes_)
            result["confusion_matrix"] = cm.tolist()

        return result

    def predict(self, values: list[float]) -> dict:
        if self.network is None:
            raise StateError("Crie ou carregue uma rede neural antes de realizar previsões.")

        model = LoadedModel(
            network=self.network,
            normalizer=self.normalizer,
            task=self.task,
            meta=self.meta,
        )
        return predict(model, values)

    # ---------- Persistência ----------

    def save(self, path: str | Path) -> None:
        if self.network is None:
            raise StateError("Nenhuma rede neural para salvar.")
        save_model(
            path=path,
            network=self.network,
            normalizer=self.normalizer,
            task=self.task,
            meta=self.meta,
        )

    def load(self, path: str | Path) -> dict:
        loaded = load_model(path)
        self.network = loaded.network
        self.normalizer = loaded.normalizer
        self.norm_kind = loaded.normalizer.kind if loaded.normalizer else None
        self.task = loaded.task
        self.meta = dict(loaded.meta)

        if self.task == "classification" and self.meta.get("class_labels"):
            self.encoder = LabelEncoder.from_classes(self.meta["class_labels"])
        else:
            self.encoder = None

        self.raw_dataset = None
        self.split_raw = None
        self.split = None
        self.history = []
        return self.describe_network()
"""Estado da interface: o que já foi feito em cada etapa (sem Flet).

O estado *de verdade* (rede, pesos, dados) fica no backend. Aqui só fica o que a UI
precisa para habilitar ou desabilitar ações e mostrar resumos. As regras de
invalidação imitam as do ``MLPService``. Exemplo: recriar a rede descarta a divisão
e o histórico.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AppState:
    task: str = "regression"
    activations: list[str] = field(default_factory=lambda: ["linear", "sigmoid", "tanh", "relu"])

    # Arquitetura (describe_network + o que a UI enviou)
    network: dict | None = None
    model_origin: str | None = None  # "created" | "loaded"
    model_path: str | None = None

    # Dados
    csv_path: str | None = None
    columns: list[str] = field(default_factory=list)
    input_cols: list[str] = field(default_factory=list)
    output_cols: list[str] = field(default_factory=list)
    dataset: dict | None = None  # retorno de load_dataset
    split: dict | None = None  # retorno de split_dataset
    normalization: str | None = None

    # Treinamento
    training: bool = False
    history: list[dict] = field(default_factory=list)
    trained_epochs: int = 0

    # Avaliação
    evaluations: dict[str, dict] = field(default_factory=dict)  # subset -> resultado

    # Modelo carregado
    loaded_meta: dict = field(default_factory=dict)

    _listeners: list[Callable[[], None]] = field(default_factory=list, repr=False)

    # ----------------------------------------------------------- observador
    def subscribe(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def notify(self) -> None:
        for listener in list(self._listeners):
            listener()

    # ------------------------------------------------------------ derivados
    @property
    def has_network(self) -> bool:
        return self.network is not None

    @property
    def has_dataset(self) -> bool:
        return self.dataset is not None

    @property
    def has_split(self) -> bool:
        return self.split is not None

    @property
    def is_trained(self) -> bool:
        return self.trained_epochs > 0 or self.model_origin == "loaded"

    @property
    def n_inputs(self) -> int | None:
        return int(self.network["n_inputs"]) if self.network else None

    @property
    def layer_specs(self) -> list[dict]:
        """Camadas no formato ``{"neurons", "activation"}``, aceitando ``n_neurons``."""
        if not self.network:
            return []
        specs = []
        for layer in self.network.get("layers", []):
            neurons = layer.get("neurons", layer.get("n_neurons"))
            specs.append({"neurons": int(neurons), "activation": layer.get("activation", "?")})
        return specs

    @property
    def n_outputs(self) -> int | None:
        if not self.network:
            return None
        if "n_outputs" in self.network:
            return int(self.network["n_outputs"])
        specs = self.layer_specs
        return specs[-1]["neurons"] if specs else None

    @property
    def n_params(self) -> int | None:
        if not self.network:
            return None
        if "n_params" in self.network:
            return int(self.network["n_params"])
        from frontend.analysis import count_parameters

        return count_parameters(self.n_inputs or 0, [s["neurons"] for s in self.layer_specs])

    @property
    def class_labels(self) -> list[str] | None:
        if self.dataset and self.dataset.get("classes"):
            return [str(c) for c in self.dataset["classes"]]
        labels = self.loaded_meta.get("class_labels")
        return [str(c) for c in labels] if labels else None

    def input_names(self) -> list[str]:
        n = self.n_inputs or 0
        names = (self.dataset or {}).get("inputs") or self.loaded_meta.get("input_cols") or []
        if len(names) == n:
            return [str(x) for x in names]
        return [f"x{_subscript(i)}" for i in range(1, n + 1)]

    def output_names(self) -> list[str]:
        n = self.n_outputs or 0
        if self.task == "classification" and self.class_labels and len(self.class_labels) == n:
            return [f"Classe {c}" for c in self.class_labels]
        names = (self.dataset or {}).get("outputs") or self.loaded_meta.get("output_cols") or []
        if len(names) == n:
            return [str(x) for x in names]
        return [f"ŷ{_subscript(i)}" for i in range(1, n + 1)]

    def architecture_text(self) -> str:
        if not self.network:
            return "—"
        sizes = [str(self.n_inputs)] + [str(s["neurons"]) for s in self.layer_specs]
        return "–".join(sizes)

    # ---------------------------------------------------------- invalidação
    def on_network_created(self, description: dict, task: str) -> None:
        self.network = description
        self.task = task
        self.model_origin = "created"
        self.model_path = None
        self.loaded_meta = {}
        # O service reseta divisão/normalização/histórico; o dataset precisa
        # ser recarregado para checar compatibilidade (e recodificar classes).
        self.dataset = None
        self._clear_after_dataset()

    def on_model_loaded(self, description: dict, path: str) -> None:
        self.network = description
        self.task = description.get("task", self.task)
        self.model_origin = "loaded"
        self.model_path = path
        self.dataset = None
        self._clear_after_dataset()

    def on_csv_selected(self, path: str, columns: list[str]) -> None:
        self.csv_path = path
        self.columns = list(columns)
        self.input_cols, self.output_cols = [], []
        self.dataset = None
        self._clear_after_dataset()

    def on_dataset_loaded(self, info: dict) -> None:
        self.dataset = info
        self._clear_after_dataset()

    def on_split(self, sizes: dict) -> None:
        self.split = sizes
        self.history = []
        self.trained_epochs = 0 if self.model_origin != "loaded" else self.trained_epochs
        self.evaluations = {}

    def _clear_after_dataset(self) -> None:
        self.split = None
        self.normalization = None
        self.history = []
        self.trained_epochs = 0
        self.evaluations = {}

    def split_size(self, subset: str) -> int | None:
        if not self.split:
            return None
        aliases = {"train": ("train", "treino", "training"), "val": ("val", "validation", "validacao"),
                   "test": ("test", "teste")}
        for key in aliases[subset]:
            if key in self.split:
                value = self.split[key]
                return len(value) if isinstance(value, (list, tuple)) else int(value)
        return None


_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _subscript(n: int) -> str:
    return str(n).translate(_SUB)

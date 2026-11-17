"""Estado da interface: o que já foi feito em cada etapa (sem Flet).

O estado *de verdade* (rede, pesos, dados) fica no backend. Aqui só fica o que a UI
precisa para habilitar ou desabilitar ações e mostrar resumos. As regras de
invalidação imitam as do ``MLPService``. Exemplo: recriar a rede descarta a divisão
e o histórico.

No modo "Criar novo modelo" a aba Dados vem primeiro: as escolhas de lá (tarefa, CSV,
colunas) existem antes da rede e permitem preencher entradas/saídas automaticamente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AppState:
    mode: str | None = None  # "create" (Criar novo modelo) | "use" (Usar modelo existente)
    task: str = "regression"  # tarefa da rede criada/carregada
    activations: list[str] = field(default_factory=lambda: ["linear", "sigmoid", "tanh", "relu"])

    # Arquitetura (describe_network + o que a UI enviou)
    network: dict | None = None
    model_origin: str | None = None  # "created" | "loaded"
    model_path: str | None = None

    # Escolhas da aba Dados (valem antes de a rede existir)
    task_choice: str = "regression"
    csv_path: str | None = None
    columns: list[str] = field(default_factory=list)
    input_cols: list[str] = field(default_factory=list)  # seleção atual
    output_cols: list[str] = field(default_factory=list)
    class_labels_preview: list[str] = field(default_factory=list)  # classes contadas no CSV
    selection_problem: str | None = None  # por que a seleção não serve (ex.: saída de texto na regressão)
    data_dirty: bool = False  # escolhas mudaram depois de aplicadas à rede

    # Dados aplicados à rede (backend)
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
    loaded_normalization: str | None = None  # normalização salva no arquivo do modelo

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
    def data_ready(self) -> bool:
        """Dados carregados, divididos e atualizados com as escolhas da aba Dados."""
        return self.has_split and not self.data_dirty

    @property
    def is_trained(self) -> bool:
        return self.trained_epochs > 0 or self.model_origin == "loaded"

    @property
    def data_selection_ready(self) -> bool:
        return bool(self.csv_path and self.input_cols and self.output_cols)

    def suggested_io(self) -> tuple[int, int] | None:
        """Entradas e neurônios de saída que o CSV escolhido pede (modo "Automático")."""
        if not self.data_selection_ready or self.selection_problem:
            return None
        n_in = len(self.input_cols)
        if self.task_choice == "classification":
            k = len(self.class_labels_preview)
            if len(self.output_cols) != 1 or k < 2:
                return None
            return n_in, (1 if k == 2 else k)  # binária: 0/1; multiclasse: one-hot
        return n_in, len(self.output_cols)

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
        for key in ("total_parameters", "n_params"):  # backend real usa "total_parameters"
            if key in self.network:
                return int(self.network[key])
        from frontend.analysis import count_parameters

        return count_parameters(self.n_inputs or 0, [s["neurons"] for s in self.layer_specs])

    @property
    def class_labels(self) -> list[str] | None:
        if self.dataset and self.dataset.get("classes"):
            return [str(c) for c in self.dataset["classes"]]
        labels = self.loaded_meta.get("class_labels")
        if labels:
            return [str(c) for c in labels]
        if self.task_choice == "classification" and self.class_labels_preview:
            return list(self.class_labels_preview)
        return None

    def input_names(self, n: int | None = None) -> list[str]:
        n = self.n_inputs if n is None else n
        n = n or 0
        for names in ((self.dataset or {}).get("inputs"), self.loaded_meta.get("input_cols"), self.input_cols):
            if names and len(names) == n:
                return [str(x) for x in names]
        return [f"x{_subscript(i)}" for i in range(1, n + 1)]

    def output_names(self, n: int | None = None, task: str | None = None) -> list[str]:
        n = self.n_outputs if n is None else n
        n = n or 0
        task = task or self.task
        labels = self.class_labels
        if task == "classification" and labels and len(labels) == n:
            return [f"Classe {c}" for c in labels]
        for names in ((self.dataset or {}).get("outputs"), self.loaded_meta.get("output_cols"), self.output_cols):
            if names and len(names) == n:
                return [str(x) for x in names]
        return [f"ŷ{_subscript(i)}" for i in range(1, n + 1)]

    def model_columns(self) -> tuple[list[str] | None, list[str] | None]:
        """Nomes das colunas de entrada e de saída do CSV de treino, se conhecidos.

        Na classificação a saída é a coluna do rótulo (1 coluna), não os neurônios.
        """
        n_in = self.n_inputs or 0
        n_out = 1 if self.task == "classification" else (self.n_outputs or 0)
        sources = (self.dataset or {}, {"inputs": self.loaded_meta.get("input_cols"),
                                         "outputs": self.loaded_meta.get("output_cols")},
                   {"inputs": self.input_cols, "outputs": self.output_cols})
        ins = next(([str(c) for c in s["inputs"]] for s in sources
                    if s.get("inputs") and len(s["inputs"]) == n_in), None)
        outs = next(([str(c) for c in s["outputs"]] for s in sources
                     if s.get("outputs") and len(s["outputs"]) == n_out), None)
        return ins, outs

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
        self.loaded_normalization = None
        self.trained_epochs = 0  # pesos novos
        # O service reseta dados/divisão/normalização/histórico ao criar a rede.
        self.dataset = None
        self._clear_after_dataset()

    def on_model_loaded(self, description: dict, path: str) -> None:
        self.network = description
        self.task = description.get("task", self.task)
        self.model_origin = "loaded"
        self.model_path = path
        self.dataset = None
        self._clear_after_dataset()

    def forget_network(self) -> None:
        """Volta ao estado "sem rede" (ex.: depois de usar um modelo só como base)."""
        self.network = None
        self.model_origin = None
        self.model_path = None
        self.trained_epochs = 0
        self.dataset = None
        self._clear_after_dataset()

    def on_csv_selected(self, path: str, columns: list[str]) -> None:
        self.csv_path = path
        self.columns = list(columns)
        self.input_cols, self.output_cols = [], []
        self.class_labels_preview = []
        self.selection_problem = None
        self.mark_data_dirty()

    def mark_data_dirty(self) -> None:
        """Escolhas da aba Dados mudaram: os dados aplicados à rede ficaram desatualizados."""
        if self.has_dataset or self.has_split:
            self.data_dirty = True

    def on_dataset_loaded(self, info: dict) -> None:
        self.dataset = info
        self._clear_after_dataset()

    def clear_data(self) -> None:
        """Nenhum dado aplicado à rede (ex.: falha ao carregar/dividir)."""
        self.dataset = None
        self._clear_after_dataset()

    def on_split(self, sizes: dict) -> None:
        self.split = sizes
        self.history = []
        self.evaluations = {}
        self.data_dirty = False

    def _clear_after_dataset(self) -> None:
        # Os pesos continuam na rede: trained_epochs não é zerado aqui.
        self.split = None
        self.normalization = None
        self.history = []
        self.evaluations = {}
        self.data_dirty = False

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

"""Serviço de DEMONSTRAÇÃO com o mesmo contrato de ``backend/service.py``.

Serve só para abrir e testar a interface sem o backend (``MLP_FRONTEND_DEMO=1``).
**Não é uma rede neural**: as curvas de erro e as previsões são simuladas.
A validação de entradas, os erros e os **formatos de retorno são os mesmos do
``backend/service.py`` real** (ex.: ``describe_network`` com ``n_neurons`` e
``total_parameters``; prévia ``{"x": [...], "y": [...]}``; rótulos das classes só em
``meta``), então a UI se comporta igual com os dois.
"""

from __future__ import annotations

import csv
import json
import math
import random
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

ACTIVATIONS = ["linear", "sigmoid", "tanh", "relu"]


class DemoBackendError(Exception):
    pass


class ConfigError(DemoBackendError):
    pass


class DatasetError(DemoBackendError):
    pass


class StateError(DemoBackendError):
    pass


class PersistenceError(DemoBackendError):
    pass


class DemoService:
    def __init__(self) -> None:
        self.network: dict | None = None
        self.task = "regression"
        self.meta: dict = {}
        self._reset_data()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._skill = 0.0  # "qualidade" simulada da rede (cresce com o treino)

    # ------------------------------------------------------------------ util
    def _reset_data(self) -> None:
        self.raw_rows: list[dict[str, str]] | None = None
        self.dataset: dict | None = None
        self._split_idx: dict[str, list[int]] | None = None
        # Mesmo formato do MLPService: split.train/.val/.test com .Y (saídas sem normalização)
        self.split: SimpleNamespace | None = None
        self.norm_kind: str | None = None
        self.history: list[dict] = []
        self.classes: list[str] | None = None

    def _require_network(self) -> dict:
        if self.network is None:
            raise StateError("Crie ou carregue uma rede antes desta etapa.")
        return self.network

    # ------------------------------------------------------------ arquitetura
    def list_activations(self) -> list[str]:
        return list(ACTIVATIONS)

    def create_network(self, n_inputs: int, layers: list[dict], seed: int | None = None,
                       task: str = "regression") -> dict:
        if task not in ("regression", "classification"):
            raise ConfigError(f"Tarefa desconhecida: {task}")
        if int(n_inputs) < 1:
            raise ConfigError("A rede precisa de ao menos 1 entrada.")
        if not layers:
            raise ConfigError("A rede precisa de ao menos 1 camada.")
        for spec in layers:
            if int(spec["neurons"]) < 1:
                raise ConfigError("A camada precisa de ao menos 1 neurônio.")
            if spec["activation"] not in ACTIVATIONS:
                raise ConfigError("Ativação desconhecida: " + str(spec["activation"]))
        self.network = {
            "n_inputs": int(n_inputs),
            "layers": [{"neurons": int(s["neurons"]), "activation": s["activation"]} for s in layers],
            "seed": seed,
        }
        self.task = task
        self.meta = {}
        self._reset_data()
        self._skill = 0.0
        return self.describe_network()

    def describe_network(self) -> dict:
        net = self._require_network()
        layers, fan_in, total = [], net["n_inputs"], 0
        for i, spec in enumerate(net["layers"]):
            params = (fan_in + 1) * spec["neurons"]
            total += params
            layers.append({"index": i, "n_inputs": fan_in, "n_neurons": spec["neurons"],
                           "activation": spec["activation"], "parameters": params})
            fan_in = spec["neurons"]
        return {"task": self.task, "n_inputs": net["n_inputs"], "n_outputs": fan_in,
                "total_parameters": total, "layers": layers}

    # ------------------------------------------------------------------ dados
    def get_csv_columns(self, path: str) -> list[str]:
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                header = next(csv.reader(f), None)
        except OSError as exc:
            raise DatasetError(f"Não foi possível abrir o arquivo: {exc}") from exc
        if not header:
            raise DatasetError("O arquivo CSV está vazio.")
        return [h.strip() for h in header]

    def load_dataset(self, path: str, input_cols: list[str], output_cols: list[str]) -> dict:
        net = self._require_network()
        columns = self.get_csv_columns(path)
        for col in [*input_cols, *output_cols]:
            if col not in columns:
                raise DatasetError(f"Coluna inexistente: {col}")
        if set(input_cols) & set(output_cols):
            raise DatasetError("Uma coluna não pode ser entrada e saída ao mesmo tempo.")
        if len(set(input_cols)) != len(input_cols) or len(set(output_cols)) != len(output_cols):
            raise DatasetError("Coluna repetida na seleção.")
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
        if not rows:
            raise DatasetError("O CSV não tem linhas de dados.")
        try:
            for r in rows:
                for c in input_cols:
                    float(r[c])
                if self.task == "regression":
                    for c in output_cols:
                        float(r[c])
        except ValueError as exc:
            raise DatasetError(f"Valor não numérico no CSV: {exc}") from exc

        n_out_expected = len(output_cols)
        self.classes = None
        if self.task == "classification":
            if len(output_cols) != 1:
                raise DatasetError("Na classificação, escolha exatamente 1 coluna de saída (o rótulo).")
            self.classes = sorted({r[output_cols[0]] for r in rows})
            n_out_expected = 1 if len(self.classes) == 2 else len(self.classes)

        if len(input_cols) != net["n_inputs"]:
            raise DatasetError(
                f"A rede tem {net['n_inputs']} entradas, mas foram escolhidas "
                f"{len(input_cols)} colunas de entrada."
            )
        n_out_net = net["layers"][-1]["neurons"]
        if n_out_net != n_out_expected:
            raise DatasetError(
                f"A camada de saída tem {n_out_net} neurônio(s), mas os dados pedem {n_out_expected}. "
                f"Ajuste a camada de saída para {n_out_expected} neurônio(s)."
            )
        self.raw_rows = rows
        self.dataset = {"inputs": list(input_cols), "outputs": list(output_cols)}
        self.meta = {"input_cols": list(input_cols), "output_cols": list(output_cols),
                     "class_labels": self.classes}
        self.split = self._split_idx = None
        self.norm_kind = None
        self.history = []
        targets = self._encode([r for r in rows[:5]])
        # Mesmo formato do MLPService: x/y numéricos; na classificação, y já codificado
        # e os rótulos das classes ficam só em `meta`.
        return {"rows": len(rows), "inputs": list(input_cols), "outputs": list(output_cols),
                "preview": [{"x": [float(r[c]) for c in input_cols], "y": y}
                            for r, y in zip(rows[:5], targets)]}

    def split_dataset(self, train_pct: float, val_pct: float, test_pct: float,
                      seed: int | None = None) -> dict:
        if self.raw_rows is None:
            raise StateError("Carregue um dataset antes de dividir.")
        if min(train_pct, val_pct, test_pct) < 0 or abs(train_pct + val_pct + test_pct - 100) > 1e-9:
            raise DatasetError("Os percentuais devem ser positivos e somar 100%.")
        n = len(self.raw_rows)
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        n_tr, n_va = round(n * train_pct / 100), round(n * val_pct / 100)
        if n_tr == 0 or n_va == 0:
            raise DatasetError("Treinamento e validação não podem ficar vazios.")
        self._split_idx = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}
        self.split = SimpleNamespace(**{k: SimpleNamespace(Y=self._targets(k)) for k in self._split_idx})
        self.history = []  # como no MLPService, a normalização atual (norm_kind) é reaplicada
        return {k: len(v) for k, v in self._split_idx.items()}

    def set_normalization(self, kind: str | None) -> dict:
        if self.split is None:
            raise StateError("Divida o dataset antes de normalizar.")
        if kind not in (None, "minmax", "zscore"):
            raise ConfigError("Normalização desconhecida: " + str(kind))
        self.norm_kind = kind
        return {"normalization": kind}

    # ------------------------------------------------------------------ treino
    def start_training(self, epochs: int, lr: float, shuffle_seed: int | None = None,
                       on_epoch: Callable[[dict], None] | None = None,
                       on_finish: Callable[[], None] | None = None) -> None:
        self._require_network()
        if self.split is None:
            raise StateError("Divida o dataset antes de treinar.")
        if lr <= 0 or epochs < 1:
            raise ConfigError("lr deve ser > 0 e epochs >= 1.")
        if self._thread and self._thread.is_alive():
            raise StateError("Já existe um treinamento em andamento.")
        self._stop.clear()
        self.history = []
        rng = random.Random(shuffle_seed)
        # Curva simulada: queda exponencial; validação volta a subir no fim (sobreajuste);
        # lr > 1 diverge, para exercitar os avisos da interface.
        start = 1.0 + rng.random()
        speed = min(0.25, 0.02 + lr * 2)
        floor = 0.02 + 0.03 * rng.random()
        turn = int(epochs * 0.7)
        delay = min(0.03, 3.0 / epochs)

        def run() -> None:
            try:
                for epoch in range(1, epochs + 1):
                    if self._stop.is_set():
                        break
                    time.sleep(delay)
                    if lr > 1:
                        train = start * math.exp(0.08 * epoch)
                    else:
                        train = floor + start * math.exp(-speed * epoch) * (1 + 0.04 * rng.random())
                    val = train * (1.15 + 0.05 * rng.random())
                    if lr <= 1 and epoch > turn:
                        val += 0.004 * (epoch - turn) * floor * 10 / max(1, epochs / 100)
                    r = {"epoch": epoch, "train_mse": train, "val_mse": val}
                    self.history.append(r)
                    self._skill = 1 / (1 + 10 * train)
                    if on_epoch:
                        on_epoch(dict(r))
            finally:
                if on_finish:
                    on_finish()

        self._thread = threading.Thread(target=run, name="demo-training", daemon=True)
        self._thread.start()

    def stop_training(self) -> None:
        self._stop.set()

    def get_history(self) -> list[dict]:
        return [dict(h) for h in self.history]

    # --------------------------------------------------------- avaliação/uso
    def _targets(self, subset: str) -> list[list[float]]:
        assert self.raw_rows is not None and self._split_idx is not None
        return self._encode([self.raw_rows[i] for i in self._split_idx[subset]])

    def _encode(self, rows: list[dict[str, str]]) -> list[list[float]]:
        """Saídas esperadas como o MLPService guarda: floats; classes em 0/1 ou one-hot."""
        assert self.dataset is not None
        outs = self.dataset["outputs"]
        if self.task == "classification":
            assert self.classes is not None
            k = len(self.classes)
            enc = []
            for r in rows:
                c = self.classes.index(r[outs[0]])
                enc.append([float(c)] if k == 2 else [1.0 if j == c else 0.0 for j in range(k)])
            return enc
        return [[float(r[c]) for c in outs] for r in rows]

    def evaluate(self, subset: str = "test") -> dict:
        self._require_network()
        if self.split is None:
            raise StateError("Divida o dataset antes de avaliar.")
        if subset not in (self._split_idx or {}):
            raise ConfigError(f"Subconjunto desconhecido: {subset}")
        y_true = self._targets(subset)
        if not y_true:
            raise DatasetError(f"O subconjunto '{subset}' está vazio.")
        rng = random.Random(len(self.history) * 31 + len(y_true))
        noise = 1 - self._skill
        flat = [v for row in y_true for v in row]
        spread = (max(flat) - min(flat)) or 1.0
        y_pred = [[v + rng.gauss(0, 0.35 * noise * spread) for v in row] for row in y_true]
        mse = sum((t - p) ** 2 for tr, pr in zip(y_true, y_pred) for t, p in zip(tr, pr)) / len(flat)
        # Mesmas chaves do MLPService.evaluate (sem y_true/y_pred).
        result: dict[str, Any] = {"subset": subset, "mse": mse}
        if self.task == "classification":
            assert self.classes is not None
            k = len(self.classes)
            matrix = [[0] * k for _ in range(k)]
            for tr, pr in zip(y_true, y_pred):
                real = int(tr[0]) if k == 2 else tr.index(max(tr))
                pred = (1 if pr[0] >= 0.5 else 0) if k == 2 else pr.index(max(pr))
                matrix[real][pred] += 1
            result["accuracy"] = sum(matrix[i][i] for i in range(k)) / len(y_true)
            result["confusion_matrix"] = matrix
            result["classes"] = list(self.classes)
        return result

    def predict(self, values: list[float]) -> dict:
        net = self._require_network()
        if len(values) != net["n_inputs"]:
            raise DatasetError(f"Esperava {net['n_inputs']} valores de entrada, recebeu {len(values)}.")
        n_out = net["layers"][-1]["neurons"]
        base = sum(values) / len(values)
        outputs = [0.5 * base + 0.1 * j for j in range(n_out)]
        result: dict[str, Any] = {"outputs": outputs}
        classes = self.meta.get("class_labels")
        if self.task == "classification" and classes:  # como o backend: rótulo só se houver classes
            if n_out == 1:
                outputs = [1 / (1 + math.exp(-base))]
                result["outputs"] = outputs
                result["label"] = classes[1] if outputs[0] >= 0.5 else classes[0]
            else:
                result["label"] = classes[outputs.index(max(outputs))]
        return result

    # ------------------------------------------------------------ persistência
    def save(self, path: str) -> None:
        net = self._require_network()
        rng = random.Random(net.get("seed"))
        layers, fan_in = [], net["n_inputs"]
        for spec in net["layers"]:
            layers.append({"n_neurons": spec["neurons"], "activation": spec["activation"],
                           "neurons": [{"weights": [rng.uniform(-1, 1) for _ in range(fan_in)],
                                        "bias": 0.0} for _ in range(spec["neurons"])]})
            fan_in = spec["neurons"]
        data = {"format_version": 1, "task": self.task, "n_inputs": net["n_inputs"], "layers": layers,
                "normalization": ({"kind": self.norm_kind, "params": {}}
                                  if self.norm_kind else None),
                "meta": {"input_cols": self.meta.get("input_cols", []),
                         "output_cols": self.meta.get("output_cols", []),
                         "class_labels": self.meta.get("class_labels")}}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            raise PersistenceError(f"Não foi possível salvar: {exc}") from exc

    def load(self, path: str) -> dict:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as exc:
            raise PersistenceError("Arquivo não encontrado.") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise PersistenceError(f"Arquivo inválido: {exc}") from exc
        try:
            if data["format_version"] != 1:
                raise PersistenceError("Versão de arquivo não suportada.")
            layers = [{"neurons": L["n_neurons"], "activation": L["activation"]} for L in data["layers"]]
            n_inputs = data["n_inputs"]
            task = data["task"]
            meta = data.get("meta") or {}
        except (KeyError, TypeError) as exc:
            raise PersistenceError(f"Campo obrigatório ausente: {exc}") from exc
        normalization = data.get("normalization") or {}
        self.network = {"n_inputs": n_inputs, "layers": layers, "seed": None}
        self.task = task
        self.meta = dict(meta)
        self._reset_data()
        self.classes = meta.get("class_labels")
        self.norm_kind = normalization.get("kind")
        self._skill = 0.8
        return self.describe_network()

"""Abrir um modelo salvo (JSON) pelo service, para os dois modos da aplicação.

- "Usar modelo existente": o modelo aberto vira a rede usada nas previsões.
- "Criar novo modelo" → "Usar modelo como base": só a configuração (tarefa, camadas,
  ativações, colunas, normalização) é aproveitada no formulário; os pesos não.
"""

from __future__ import annotations

from dataclasses import dataclass

from frontend.service_gateway import ServiceGateway
from frontend.state import AppState


@dataclass
class ModelInfo:
    path: str
    description: dict  # describe_network()
    meta: dict  # input_cols, output_cols, class_labels
    normalization: str | None  # "minmax" | "zscore" | None

    @property
    def task(self) -> str:
        return self.description.get("task", "regression")

    @property
    def layer_specs(self) -> list[dict]:
        return [{"neurons": int(layer.get("neurons", layer.get("n_neurons"))),
                 "activation": layer.get("activation", "linear")}
                for layer in self.description.get("layers", [])]


async def read_model(gateway: ServiceGateway, path: str) -> ModelInfo:
    """Carrega o JSON no service e lê tudo o que a UI precisa (GatewayError se falhar)."""
    description = dict(await gateway.call("load", path) or {})  # load() devolve describe_network()
    meta = {k: v for k, v in (description.get("meta") or {}).items() if v is not None}
    return ModelInfo(path=path, description=description, meta=meta,
                     normalization=description.get("normalization"))


def apply_loaded_model(state: AppState, info: ModelInfo) -> None:
    """Modo "Usar modelo existente": o modelo aberto passa a ser a rede atual."""
    state.loaded_meta = info.meta
    state.on_model_loaded(info.description, info.path)
    state.loaded_normalization = info.normalization

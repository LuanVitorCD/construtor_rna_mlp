"""Ponte entre a interface e ``backend/service.py`` (``MLPService``).

- A UI nunca chama o backend diretamente: tudo passa por ``ServiceGateway``.
- Chamadas potencialmente lentas rodam fora do laço de eventos (``asyncio.to_thread``),
  para a janela não travar.
- ``BackendError`` (e subclasses) vira ``GatewayError`` com título e mensagem prontos.
- Se o pacote ``backend`` ainda não existir, usa ``DemoService`` e avisa na tela.
  Para forçar o modo demonstração: ``MLP_FRONTEND_DEMO=1``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)

ERROR_TITLES = {
    "ConfigError": "Configuração inválida",
    "DatasetError": "Problema nos dados",
    "StateError": "Etapa pendente",
    "PersistenceError": "Erro no arquivo do modelo",
}


class GatewayError(Exception):
    def __init__(self, title: str, message: str, *, kind: str = "BackendError"):
        super().__init__(message)
        self.title = title
        self.message = message
        self.kind = kind


class ServiceGateway:
    def __init__(self, service: Any, backend_error: type[BaseException], *, demo: bool):
        self.service = service
        self.backend_error = backend_error
        self.demo = demo
        self._lock = threading.Lock()

    def _translate(self, exc: BaseException) -> GatewayError:
        if isinstance(exc, self.backend_error):
            kind = type(exc).__name__
            return GatewayError(ERROR_TITLES.get(kind, "Erro"), str(exc) or kind, kind=kind)
        log.exception("Erro inesperado no backend", exc_info=exc)
        return GatewayError(
            "Erro inesperado",
            f"{type(exc).__name__}: {exc}",
            kind="Unexpected",
        )

    def call_sync(self, method: str, *args: Any, **kwargs: Any) -> Any:
        fn: Callable[..., Any] = getattr(self.service, method)
        try:
            with self._lock:  # uma chamada por vez: o MLPService guarda estado
                return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - convertido para a UI
            raise self._translate(exc) from exc

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self.call_sync, method, *args, **kwargs)

    def reset(self) -> None:
        """Começa do zero com um service novo (ao trocar de modo / voltar ao menu)."""
        with self._lock:
            self.service = type(self.service)()

    # Leituras somente-leitura de atributos públicos do MLPService que a fachada não devolve
    # (backend/service.py): rótulos das classes e nomes das colunas (`meta`) e a normalização
    # ativa (`norm_kind`). Se o atributo não existir, a UI segue com valores padrão.
    def read_meta(self) -> dict:
        with self._lock:
            meta = getattr(self.service, "meta", None)
            return dict(meta) if isinstance(meta, dict) else {}

    def read_norm_kind(self) -> str | None:
        with self._lock:
            kind = getattr(self.service, "norm_kind", None)
            return kind if kind in ("minmax", "zscore") else None

    def read_targets(self, subset: str) -> list[list[float]] | None:
        """Saídas esperadas (y) de um subconjunto, lidas de ``service.split`` (não normalizadas).

        Usado para o R² da regressão (o evaluate() só devolve o MSE). None se indisponível.
        """
        with self._lock:
            split = getattr(self.service, "split", None)
            dataset = getattr(split, subset, None) if split is not None else None
            targets = getattr(dataset, "Y", None)
            if targets is None:
                return None
            try:
                return [[float(v) for v in (row if hasattr(row, "__len__") else [row])] for row in targets]
            except (TypeError, ValueError):
                return None


def create_gateway() -> ServiceGateway:
    if os.environ.get("MLP_FRONTEND_DEMO") != "1":
        try:
            from backend.errors import BackendError  # type: ignore[import-not-found]
            from backend.service import MLPService  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            # Só cai no modo demonstração se o que falta é o próprio backend;
            # um import quebrado *dentro* do backend deve aparecer como erro.
            if exc.name not in ("backend", "backend.service", "backend.errors"):
                raise
            log.warning("Pacote 'backend' não encontrado: usando o modo demonstração.")
        else:
            return ServiceGateway(MLPService(), BackendError, demo=False)

    from frontend.demo_service import DemoBackendError, DemoService

    return ServiceGateway(DemoService(), DemoBackendError, demo=True)

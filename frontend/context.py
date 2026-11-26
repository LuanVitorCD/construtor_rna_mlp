"""Objeto compartilhado pelas telas: página, serviço, estado e utilidades de UI."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Awaitable, Callable


def updates_page(handler):
    """Decorador para handlers assíncronos: sempre termina com ``page.update()``.

    O Flet 1.x só atualiza a página sozinho no fim de um evento se o handler NÃO
    chamou ``update()`` no meio (``session.after_event``). Os handlers chamam
    ``update()`` antes de esperar o backend (para desabilitar o botão), então sem
    isto o resultado só aparecia na próxima atualização, por exemplo ao trocar de aba,
    e listas alteradas sem envio podiam aparecer duplicadas depois.
    O objeto decorado precisa ter ``self.ctx`` (AppContext).
    """

    @functools.wraps(handler)
    async def wrapper(self, *args, **kwargs):
        try:
            return await handler(self, *args, **kwargs)
        finally:
            self.ctx.update()

    return wrapper

import flet as ft

from frontend.service_gateway import ServiceGateway
from frontend.state import AppState


async def _no_prepare() -> tuple[bool, str | None]:
    return False, None


async def _no_open() -> None:
    return None


@dataclass
class AppContext:
    page: ft.Page
    gateway: ServiceGateway
    state: AppState
    file_picker: ft.FilePicker
    # Preenchidos pelo AppShell (ou pelos testes):
    toast: Callable[[str, str], None] = lambda level, text: None
    navigate: Callable[[int], None] = lambda index: None
    confirm: Callable[[str, str, str], Awaitable[bool]] = None  # type: ignore[assignment]
    # Aplica as escolhas da aba Dados à rede (carregar + dividir + normalizar): (ok, erro)
    prepare_data: Callable[[], Awaitable[tuple[bool, str | None]]] = _no_prepare
    # Abre outro modelo (modo "Usar modelo existente")
    open_model: Callable[[], Awaitable[None]] = _no_open
    extras: dict = field(default_factory=dict)

    def update(self) -> None:
        self.page.update()

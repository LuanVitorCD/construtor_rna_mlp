"""Objeto compartilhado pelas telas: página, serviço, estado e utilidades de UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

import flet as ft

from frontend.service_gateway import ServiceGateway
from frontend.state import AppState


@dataclass
class AppContext:
    page: ft.Page
    gateway: ServiceGateway
    state: AppState
    file_picker: ft.FilePicker
    # Preenchidos pelo AppShell:
    toast: Callable[[str, str], None] = lambda level, text: None
    navigate: Callable[[int], None] = lambda index: None
    confirm: Callable[[str, str, str], Awaitable[bool]] = None  # type: ignore[assignment]
    adjust_architecture: Callable[[int, int], None] = lambda n_in, n_out: None
    extras: dict = field(default_factory=dict)

    def update(self) -> None:
        self.page.update()

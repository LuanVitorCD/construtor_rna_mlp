"""Tela inicial: escolher entre "Criar novo modelo" e "Usar modelo existente"."""

from __future__ import annotations

from typing import Awaitable, Callable

import flet as ft

from frontend.components.ui import body, heading
from frontend.texts import MODE_DESCRIPTIONS, MODE_LABELS

C = ft.Colors


def _steps(labels: list[str]) -> ft.Control:
    items: list[ft.Control] = []
    for i, label in enumerate(labels):
        if i:
            items.append(ft.Icon(ft.Icons.CHEVRON_RIGHT, size=18, color=C.ON_SURFACE_VARIANT))
        items.append(ft.Container(
            content=ft.Text(label, size=13, color=C.ON_PRIMARY_CONTAINER, weight=ft.FontWeight.W_500),
            bgcolor=C.PRIMARY_CONTAINER,
            border_radius=12,
            padding=ft.Padding.symmetric(horizontal=10, vertical=4),
        ))
    return ft.Row(items, wrap=True, spacing=4, run_spacing=6,
                  vertical_alignment=ft.CrossAxisAlignment.CENTER)


def _card(icon: ft.IconData, title: str, description: str, steps: list[str], button: ft.Control) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                ft.Icon(icon, size=44, color=C.PRIMARY),
                heading(title, style=ft.TextThemeStyle.HEADLINE_SMALL),
                body(description, muted=False),
                ft.Text("Etapas:", theme_style=ft.TextThemeStyle.LABEL_LARGE, color=C.ON_SURFACE_VARIANT),
                _steps(steps),
                ft.Container(height=4),
                button,
            ],
            spacing=12,
            tight=True,
        ),
        bgcolor=C.SURFACE_CONTAINER_LOW,
        border=ft.Border.all(1, C.OUTLINE_VARIANT),
        border_radius=16,
        padding=28,
        col={"xs": 12, "md": 6},
    )


class ModeMenu:
    def __init__(self, on_create: Callable[..., Awaitable[None]], on_use: Callable[..., Awaitable[None]]) -> None:
        button_style = ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=24, vertical=18))
        self.create_btn = ft.FilledButton(MODE_LABELS["create"], icon=ft.Icons.ADD, on_click=on_create,
                                          style=button_style, autofocus=True, tooltip="Ctrl+N")
        self.use_btn = ft.FilledButton("Abrir modelo salvo…", icon=ft.Icons.FOLDER_OPEN, on_click=on_use,
                                       style=button_style, tooltip="Ctrl+O")
        cards = ft.ResponsiveRow(
            [
                _card(ft.Icons.HUB, MODE_LABELS["create"], MODE_DESCRIPTIONS["create"],
                      ["Dados", "Arquitetura", "Treinamento", "Avaliação", "Previsão"], self.create_btn),
                _card(ft.Icons.FOLDER_OPEN, MODE_LABELS["use"], MODE_DESCRIPTIONS["use"],
                      ["Abrir JSON", "Informações do modelo", "Previsão"], self.use_btn),
            ],
            spacing=24,
            run_spacing=24,
        )
        self.root = ft.Container(
            content=ft.Column(
                [
                    heading("Como você quer começar?", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
                    body("Escolha um modo. Dá para voltar a este menu a qualquer momento pelo botão "
                         "“Voltar ao menu”, na barra superior."),
                    ft.Container(height=8),
                    cards,
                ],
                spacing=12,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            ),
            padding=ft.Padding.symmetric(horizontal=40, vertical=32),
            expand=True,
        )

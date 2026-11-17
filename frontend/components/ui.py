"""Blocos visuais reutilizáveis.

Princípios de acessibilidade aplicados em todos os componentes:
- estados (erro, aviso, sucesso) usam **ícone + texto**, nunca só a cor (WCAG 1.4.1);
- títulos de seção são marcados como *header* para leitores de tela (WCAG 1.3.1);
- mensagens de resultado ficam numa *live region*: o leitor de tela anuncia a
  mudança sem o usuário precisar procurar (WCAG 4.1.3);
- toda cor vem de um token do tema, que tem contraste conferido (ver ``frontend/theme.py``).
"""

from __future__ import annotations

from typing import Callable, Iterable

import flet as ft

from frontend.analysis import Insight
from frontend.texts import activation_label
from frontend.theme import ON_SUCCESS_CONTAINER, SUCCESS, SUCCESS_CONTAINER

C = ft.Colors

LEVELS: dict[str, tuple[ft.IconData, str, str, str]] = {
    # nível: (ícone, fundo, texto, nome falado)
    "info": (ft.Icons.INFO_OUTLINE, C.SURFACE_CONTAINER_HIGH, C.ON_SURFACE, "Informação"),
    "success": (ft.Icons.CHECK_CIRCLE_OUTLINE, SUCCESS_CONTAINER, ON_SUCCESS_CONTAINER, "Concluído"),
    "warning": (ft.Icons.WARNING_AMBER, C.TERTIARY_CONTAINER, C.ON_TERTIARY_CONTAINER, "Atenção"),
    "error": (ft.Icons.ERROR_OUTLINE, C.ERROR_CONTAINER, C.ON_ERROR_CONTAINER, "Erro"),
}

# Cor de destaque (borda e ícone): verde = sucesso, vermelho = erro.
ACCENTS: dict[str, str] = {
    "info": C.OUTLINE_VARIANT,  # discreta: o destaque fica para verde/vermelho
    "success": SUCCESS,
    "warning": C.TERTIARY,
    "error": C.ERROR,
}


def level_icon_color(level: str) -> str:
    """Ícone verde/vermelho em sucesso/erro; nos demais níveis, a cor do texto."""
    if level in ("success", "error"):
        return ACCENTS[level]
    return LEVELS.get(level, LEVELS["info"])[2]


def heading(text: str, *, style: ft.TextThemeStyle = ft.TextThemeStyle.TITLE_MEDIUM) -> ft.Control:
    return ft.Semantics(
        header=True,
        content=ft.Text(text, theme_style=style, weight=ft.FontWeight.W_600, color=C.ON_SURFACE),
    )


def body(text: str, *, muted: bool = True, selectable: bool = False, size: int | None = None) -> ft.Text:
    return ft.Text(
        text,
        theme_style=ft.TextThemeStyle.BODY_MEDIUM,
        color=C.ON_SURFACE_VARIANT if muted else C.ON_SURFACE,
        selectable=selectable,
        size=size,
    )


def section(
    title: str,
    controls: Iterable[ft.Control],
    *,
    icon: ft.IconData | None = None,
    subtitle: str | None = None,
    col: dict | None = None,
    trailing: ft.Control | None = None,
) -> ft.Container:
    header: list[ft.Control] = []
    if icon:
        header.append(ft.Icon(icon, color=C.PRIMARY, size=22))
    header.append(ft.Container(heading(title), expand=True))
    if trailing:
        header.append(trailing)
    items: list[ft.Control] = [ft.Row(header, spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER)]
    if subtitle:
        items.append(body(subtitle))
    items.extend(controls)
    return ft.Container(
        content=ft.Column(items, spacing=14, tight=True),
        bgcolor=C.SURFACE_CONTAINER_LOW,
        border=ft.Border.all(1, C.OUTLINE_VARIANT),
        border_radius=12,
        padding=20,
        col=col,
    )


def _message_box(level: str, title: str, text: str | None, action: tuple[str, Callable] | None = None) -> ft.Container:
    icon, bg, fg, spoken = LEVELS.get(level, LEVELS["info"])
    lines: list[ft.Control] = [ft.Text(title, weight=ft.FontWeight.W_600, color=fg, selectable=True)]
    if text:
        lines.append(ft.Text(text, color=fg, selectable=True))
    if action:  # correção rápida; usa a cor do texto da mensagem (contraste garantido no tema)
        label, on_click = action
        lines.append(ft.Container(
            ft.OutlinedButton(label, on_click=on_click,
                              style=ft.ButtonStyle(color=fg, side=ft.BorderSide(1, fg))),
            padding=ft.Padding.only(top=6),
        ))
    return ft.Container(
        content=ft.Row(
            [
                ft.Icon(icon, color=level_icon_color(level), size=22, semantics_label=spoken),
                ft.Column(lines, spacing=2, tight=True, expand=True),
            ],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.START,
        ),
        bgcolor=bg,
        border=ft.Border.all(1.5, ACCENTS.get(level, C.OUTLINE)),
        border_radius=10,
        padding=ft.Padding.symmetric(horizontal=14, vertical=12),
    )


def insight_box(insight: Insight) -> ft.Container:
    return _message_box(insight.level, insight.title, insight.message)


class Message:
    """Mensagem de status em linha (vazia até ``show``), anunciada por leitores de tela."""

    # Obs.: no Flet 1.x o conteúdo de um Semantics precisa estar sempre visível;
    # por isso quem aparece/some é o próprio Semantics (root).
    def __init__(self) -> None:
        self._holder = ft.Container()
        self.root = ft.Semantics(live_region=True, content=self._holder, visible=False)
        self.level: str | None = None
        self.title: str | None = None
        self.action: tuple[str, Callable] | None = None

    @property
    def visible(self) -> bool:
        return bool(self.root.visible)

    def show(self, level: str, title: str, text: str | None = None,
             action: tuple[str, Callable] | None = None) -> None:
        """``action`` = (rótulo, handler) mostra um botão de correção rápida na mensagem."""
        self.level, self.title = level, title
        self.action = action
        self._holder.content = _message_box(level, title, text, action)
        self.root.visible = True

    def clear(self) -> None:
        self.level = self.title = self.action = None
        self._holder.content = None
        self.root.visible = False


def metric(label: str, value: str, *, hint: str | None = None, col: dict | None = None) -> ft.Container:
    items: list[ft.Control] = [
        ft.Text(label, theme_style=ft.TextThemeStyle.LABEL_LARGE, color=C.ON_SURFACE_VARIANT),
        ft.Text(value, theme_style=ft.TextThemeStyle.HEADLINE_SMALL, color=C.PRIMARY,
                weight=ft.FontWeight.W_600, selectable=True),
    ]
    if hint:
        items.append(ft.Text(hint, theme_style=ft.TextThemeStyle.BODY_SMALL, color=C.ON_SURFACE_VARIANT))
    return ft.Container(
        content=ft.Semantics(label=f"{label}: {value}", content=ft.Column(items, spacing=2, tight=True)),
        bgcolor=C.SURFACE,
        border=ft.Border.all(1, C.OUTLINE_VARIANT),
        border_radius=10,
        padding=14,
        col=col or {"xs": 12, "sm": 6, "md": 4},
    )


NUMBER_FILTER = ft.InputFilter(regex_string=r"^[0-9eE+\-.,]*$", allow=True, replacement_string="")
INT_FILTER = ft.InputFilter(regex_string=r"^[0-9]*$", allow=True, replacement_string="")


def number_field(
    label: str,
    value: str = "",
    *,
    helper: str | None = None,
    hint: str | None = None,
    integer: bool = False,
    width: float | None = None,
    col: dict | None = None,
    on_change: Callable | None = None,
    on_submit: Callable | None = None,
    suffix: str | None = None,
    tooltip: str | None = None,
    autofocus: bool = False,
) -> ft.TextField:
    return ft.TextField(
        label=label,
        value=value,
        helper=helper,
        hint_text=hint,
        keyboard_type=ft.KeyboardType.NUMBER,
        input_filter=INT_FILTER if integer else NUMBER_FILTER,
        width=width,
        col=col,
        on_change=on_change,
        on_submit=on_submit,
        suffix=suffix,
        tooltip=tooltip,
        autofocus=autofocus,
        dense=False,
        border_color=C.OUTLINE,
        error_max_lines=3,
        helper_max_lines=2,
    )


def activation_dropdown(
    value: str,
    activations: list[str],
    *,
    label: str = "Função de ativação",
    on_select: Callable | None = None,
    width: float | None = None,
    col: dict | None = None,
) -> ft.Dropdown:
    return ft.Dropdown(
        label=label,
        value=value,
        options=[ft.DropdownOption(key=a, text=activation_label(a)) for a in activations],
        on_select=on_select,
        width=width,
        col=col,
        border_color=C.OUTLINE,
    )


def requirement_list(items: list[tuple[str, bool]]) -> ft.Column:
    """Lista de pré-requisitos: ✓ feito / ○ pendente, com texto (não só ícone)."""
    rows = []
    for text, done in items:
        rows.append(
            ft.Row(
                [
                    ft.Icon(
                        ft.Icons.CHECK_CIRCLE if done else ft.Icons.RADIO_BUTTON_UNCHECKED,
                        color=SUCCESS if done else C.ON_SURFACE_VARIANT,
                        size=20,
                        semantics_label="feito" if done else "pendente",
                    ),
                    ft.Text(text + ("" if done else " (pendente)"),
                            color=C.ON_SURFACE if done else C.ON_SURFACE_VARIANT, expand=True),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )
        )
    return ft.Column(rows, spacing=6, tight=True)


def learn_more(title: str, text: str) -> ft.ExpansionTile:
    """Explicação longa, recolhida por padrão (acessível por teclado)."""
    return ft.ExpansionTile(
        title=ft.Text(title, color=C.PRIMARY, weight=ft.FontWeight.W_500),
        leading=ft.Icon(ft.Icons.HELP_OUTLINE, color=C.PRIMARY),
        controls=[ft.Container(body(text, muted=False, selectable=True), padding=ft.Padding.only(left=16, right=16, bottom=12))],
        tile_padding=ft.Padding.symmetric(horizontal=4),
        shape=ft.RoundedRectangleBorder(radius=8),
        collapsed_shape=ft.RoundedRectangleBorder(radius=8),
        dense=True,
    )


def architecture_table(n_inputs: int, layer_specs: list[dict]) -> ft.Control:
    """Tabela "camada × neurônios × ativação × parâmetros" (com rolagem horizontal)."""
    from frontend.parsing import format_number
    from frontend.texts import ACTIVATION_RANGES

    rows = []
    fan_in = n_inputs
    for i, spec in enumerate(layer_specs):
        is_out = i == len(layer_specs) - 1
        params = (fan_in + 1) * spec["neurons"]
        fan_in = spec["neurons"]
        rows.append(ft.DataRow(cells=[
            ft.DataCell(ft.Text("Saída" if is_out else f"Oculta {i + 1}")),
            ft.DataCell(ft.Text(str(spec["neurons"]))),
            ft.DataCell(ft.Text(activation_label(spec["activation"]))),
            ft.DataCell(ft.Text(ACTIVATION_RANGES.get(spec["activation"], ""))),
            ft.DataCell(ft.Text(format_number(params))),
        ]))
    table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Camada")),
            ft.DataColumn(ft.Text("Neurônios"), numeric=True),
            ft.DataColumn(ft.Text("Ativação")),
            ft.DataColumn(ft.Text("Faixa")),
            ft.DataColumn(ft.Text("Parâmetros"), numeric=True),
        ],
        rows=rows,
        heading_text_style=ft.TextStyle(weight=ft.FontWeight.W_600, color=C.ON_SURFACE),
        column_spacing=14,
        horizontal_margin=4,
    )
    return ft.Row([table], scroll=ft.ScrollMode.AUTO)

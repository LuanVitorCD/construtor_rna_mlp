"""Diagrama da MLP desenhado em Canvas, no estilo dos slides da Aula 14.

Os nós de entrada aparecem só com contorno: eles apenas fornecem os valores das
características e não calculam nada (slide 5 da Aula 12). Os neurônios aparecem
preenchidos. Para leitores de tela há uma descrição textual equivalente
(``Semantics.label``) e uma legenda visível abaixo do desenho.
"""

from __future__ import annotations

import flet as ft
import flet.canvas as cv

from frontend.texts import activation_label

C = ft.Colors
MAX_NODES = 8  # acima disso, mostra 7 nós + "⋮ +k"


def _short(name: str, limit: int = 12) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


def describe(n_inputs: int, layers: list[dict]) -> str:
    parts = [f"{n_inputs} entrada{'s' if n_inputs != 1 else ''}"]
    for i, layer in enumerate(layers):
        kind = "camada de saída" if i == len(layers) - 1 else f"camada oculta {i + 1}"
        n = layer["neurons"]
        parts.append(f"{kind} com {n} neurônio{'s' if n != 1 else ''} {activation_label(layer['activation'])}")
    return "Diagrama da rede: " + "; ".join(parts) + "."


class NetworkDiagram:
    def __init__(self, height: int = 300) -> None:
        self.height = height
        self.canvas = cv.Canvas(shapes=[], width=560, height=height)
        self._semantics = ft.Semantics(label="Diagrama da rede", image=True, content=self.canvas)
        self.caption = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=C.ON_SURFACE_VARIANT)
        self.root = ft.Column(
            [
                ft.Row([self._semantics], scroll=ft.ScrollMode.AUTO),
                self.caption,
            ],
            spacing=6,
            tight=True,
        )

    def clear(self, message: str) -> None:
        """Sem desenho (ex.: entradas/saídas ainda indefinidas): mostra só a explicação."""
        self.canvas.shapes = []
        self.canvas.height = 0
        self._semantics.label = message
        self.caption.value = message

    def render(
        self,
        n_inputs: int,
        layers: list[dict],
        *,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ) -> None:
        columns: list[tuple[str, int, str | None]] = [("Entrada", n_inputs, None)]
        for i, layer in enumerate(layers):
            title = "Saída" if i == len(layers) - 1 else f"Oculta {i + 1}"
            columns.append((title, int(layer["neurons"]), layer["activation"]))

        h = self.height
        width = max(560, 150 * len(columns) + 120)
        left, right = 110, width - 110
        top, bottom = 24, h - 64
        xs = [left + (right - left) * i / max(1, len(columns) - 1) for i in range(len(columns))]

        positions: list[list[float]] = []
        hidden_counts: list[int] = []
        for _title, n, _act in columns:
            shown = n if n <= MAX_NODES else MAX_NODES - 1
            hidden_counts.append(n - shown)
            slots = shown + (1 if n > MAX_NODES else 0)
            if slots == 1:
                ys = [(top + bottom) / 2]
            else:
                gap = min(44.0, (bottom - top) / (slots - 1))
                start = (top + bottom) / 2 - gap * (slots - 1) / 2
                ys = [start + gap * k for k in range(slots)]
            positions.append(ys[:shown])
            positions[-1].append(ys[-1] if n > MAX_NODES else None)  # posição do "⋮"

        radius = 15 if max(min(n, MAX_NODES) for _, n, _ in columns) <= 6 else 12
        shapes: list[cv.Shape] = []
        edge_paint = ft.Paint(color=ft.Colors.with_opacity(0.45, C.ON_SURFACE_VARIANT), stroke_width=1)

        # Arestas (desenhadas primeiro, ficam atrás dos nós)
        for c in range(len(columns) - 1):
            for y1 in [y for y in positions[c] if y is not None]:
                for y2 in [y for y in positions[c + 1] if y is not None]:
                    shapes.append(cv.Line(xs[c] + radius, y1, xs[c + 1] - radius, y2, paint=edge_paint))

        label_style = ft.TextStyle(size=12, color=C.ON_SURFACE_VARIANT)
        for c, (title, n, act) in enumerate(columns):
            is_input, is_output = c == 0, c == len(columns) - 1
            if is_input:
                fill, stroke = C.SURFACE, C.OUTLINE
            elif is_output:
                fill, stroke = C.SECONDARY_CONTAINER, C.SECONDARY
            else:
                fill, stroke = C.PRIMARY_CONTAINER, C.PRIMARY
            ys = positions[c]
            for k, y in enumerate(ys[:-1]):
                if y is None:
                    continue
                shapes.append(cv.Circle(xs[c], y, radius, paint=ft.Paint(color=fill, style=ft.PaintingStyle.FILL)))
                shapes.append(cv.Circle(xs[c], y, radius,
                                        paint=ft.Paint(color=stroke, stroke_width=2, style=ft.PaintingStyle.STROKE)))
                if is_input:
                    name = (input_names or [])[k] if input_names and k < len(input_names) else f"x{k + 1}"
                    shapes.append(cv.Text(xs[c] - radius - 8, y, _short(name), style=label_style,
                                          alignment=ft.Alignment.CENTER_RIGHT))
                elif is_output:
                    name = (output_names or [])[k] if output_names and k < len(output_names) else f"ŷ{k + 1}"
                    shapes.append(cv.Text(xs[c] + radius + 8, y, _short(name), style=label_style,
                                          alignment=ft.Alignment.CENTER_LEFT))
            if ys[-1] is not None:
                shapes.append(cv.Text(xs[c], ys[-1], f"⋮ +{hidden_counts[c]}",
                                      style=ft.TextStyle(size=13, weight=ft.FontWeight.W_600, color=C.ON_SURFACE),
                                      alignment=ft.Alignment.CENTER))
            sub = f"{n} nó{'s' if n != 1 else ''}" if is_input else f"{n} · {act}"
            shapes.append(cv.Text(xs[c], h - 44, title,
                                  style=ft.TextStyle(size=13, weight=ft.FontWeight.W_600, color=C.ON_SURFACE),
                                  alignment=ft.Alignment.TOP_CENTER))
            shapes.append(cv.Text(xs[c], h - 24, sub, style=label_style, alignment=ft.Alignment.TOP_CENTER))

        self.canvas.shapes = shapes
        self.canvas.width = width
        self.canvas.height = h
        text = describe(n_inputs, layers)
        self._semantics.label = text
        self.caption.value = "Nós vazados: entradas (só repassam valores). Preenchidos: neurônios (z = Σwx + b, a = f(z))."

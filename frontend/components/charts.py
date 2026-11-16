"""Gráficos (flet-charts).

- ``LossChart``: curva de aprendizagem, com o MSE de treinamento e de validação por época.
  O eixo pode ser logarítmico, porque o MSE costuma cair várias ordens de grandeza.
  A melhor época de validação (ponto de parada antecipada) fica marcada.
- ``ComparisonChart``: "valores desejados × valores estimados pela rede" por amostra,
  como nos "Resultados comparativos" de Ivan Nunes da Silva (cap. 5).

As séries não dependem só da cor (WCAG 1.4.1): a validação é tracejada, o marcador
é um círculo, e há uma legenda em texto. Os tooltips mostram o valor real (não o log).
"""

from __future__ import annotations

import math
from typing import Sequence

import flet as ft
import flet_charts as fch

from frontend.analysis import best_epoch, downsample_indices
from frontend.parsing import format_number

C = ft.Colors
_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Marcas "redondas" (1, 2, 2,5 ou 5 × 10^k) cobrindo [lo, hi].

    A primeira marca é ≤ lo e a última é ≥ hi. Elas viram os limites do eixo, então a
    curva nunca passa da borda do gráfico.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return []
    if hi <= lo:
        hi = lo + (abs(lo) or 1.0)
    raw = (hi - lo) / max(1, target)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.floor(lo / step) * step
    last = math.ceil(hi / step - 1e-9) * step  # tolera erro de ponto flutuante (ex.: 10/2 = 5,0000001)
    count = int(round((last - first) / step))
    # 12 algarismos significativos: tira o ruído de ponto flutuante sem juntar marcas de valores minúsculos
    return [float(f"{first + k * step:.12g}") for k in range(count + 1)]


def _axis_label(value: float, text: str) -> fch.ChartAxisLabel:
    return fch.ChartAxisLabel(value=value, label=ft.Text(text, size=12, color=C.ON_SURFACE_VARIANT))


def _legend_item(text: str, *, dashed: bool = False, dot: bool = False, color: str = C.PRIMARY) -> ft.Control:
    if dot:
        swatch: ft.Control = ft.Container(width=12, height=12, border_radius=6, bgcolor=color,
                                          border=ft.Border.all(2, C.SURFACE))
    elif dashed:
        swatch = ft.Row([ft.Container(width=7, height=3, bgcolor=color) for _ in range(3)], spacing=3, tight=True)
    else:
        swatch = ft.Container(width=27, height=3, bgcolor=color)
    return ft.Row([swatch, ft.Text(text, size=13, color=C.ON_SURFACE)], spacing=8, tight=True)


def _tooltip(text: str) -> fch.LineChartDataPointTooltip:
    return fch.LineChartDataPointTooltip(text=text, text_style=ft.TextStyle(size=12, color=C.ON_INVERSE_SURFACE))


def _empty(text: str, height: int) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [ft.Icon(ft.Icons.SHOW_CHART, size=36, color=C.ON_SURFACE_VARIANT),
             ft.Text(text, color=C.ON_SURFACE_VARIANT, text_align=ft.TextAlign.CENTER)],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            tight=True,
        ),
        height=height,
        alignment=ft.Alignment.CENTER,
        border=ft.Border.all(1, C.OUTLINE_VARIANT),
        border_radius=8,
    )


def _base_chart(height: int) -> fch.LineChart:
    # Sem expand: o gráfico fica dentro de colunas com rolagem (altura ilimitada).
    return fch.LineChart(
        data_series=[],
        height=height,
        interactive=True,
        animation=0,  # sem animação (atualiza ao vivo). Obs.: `False` quebra o LineChart no flet-charts 1.1.2

        border=ft.Border.all(1, C.OUTLINE_VARIANT),
        horizontal_grid_lines=fch.ChartGridLines(color=C.OUTLINE_VARIANT, width=1, dash_pattern=[3, 3]),
        vertical_grid_lines=fch.ChartGridLines(color=C.OUTLINE_VARIANT, width=1, dash_pattern=[3, 3]),
        tooltip=fch.LineChartTooltip(bgcolor=C.INVERSE_SURFACE, fit_inside_horizontally=True,
                                     fit_inside_vertically=True, max_width=220),
    )


def _semantic_wrapper(chart: fch.LineChart, label: str) -> ft.Semantics:
    # O conteúdo de Semantics precisa estar visível: esconde-se o wrapper, não o gráfico.
    return ft.Semantics(label=label, content=chart, visible=False)


class LossChart:
    MAX_POINTS = 400

    def __init__(self, height: int = 340) -> None:
        self.log_scale = False
        self.height = height
        self.chart = _base_chart(height)
        self.placeholder = _empty("A curva de aprendizagem aparece aqui durante o treinamento.", height)
        self._semantics = _semantic_wrapper(self.chart, "Gráfico do MSE por época, ainda vazio.")
        self.legend = ft.Row(
            [
                _legend_item("Treinamento (linha contínua)", color=C.PRIMARY),
                _legend_item("Validação (linha tracejada)", dashed=True, color=C.TERTIARY),
                _legend_item("Menor MSE de validação", dot=True, color=C.SECONDARY),
            ],
            wrap=True,
            spacing=20,
            run_spacing=8,
        )
        self.root = ft.Column([self.placeholder, self._semantics, self.legend], spacing=10, tight=True)
        self._history: list[dict] = []
        self._total = 0

    def set_log_scale(self, value: bool) -> None:
        self.log_scale = value
        self.render(self._history, self._total)

    def _y(self, v: float) -> float:
        if self.log_scale:
            return math.log10(max(v, 1e-12))
        return v

    def render(self, history: Sequence[dict], total_epochs: int) -> None:
        self._history = list(history)
        self._total = total_epochs
        rows = [h for h in history if math.isfinite(h.get("train_mse", math.nan))
                and math.isfinite(h.get("val_mse", math.nan))]
        if not rows:
            self._semantics.visible = False
            self.placeholder.visible = True
            self._semantics.label = "Gráfico do MSE por época, ainda vazio."
            return
        self._semantics.visible = True
        self.placeholder.visible = False

        best = best_epoch(rows)
        best_index = rows.index(best) if best else -1
        idx = downsample_indices(len(rows), self.MAX_POINTS, keep=[best_index])
        train_pts, val_pts = [], []
        for i in idx:
            r = rows[i]
            label = f"Época {r['epoch']}"
            train_pts.append(fch.LineChartDataPoint(
                r["epoch"], self._y(r["train_mse"]),
                tooltip=_tooltip(f"{label}\nTreinamento: {format_number(r['train_mse'])}")))
            val_pts.append(fch.LineChartDataPoint(
                r["epoch"], self._y(r["val_mse"]),
                tooltip=_tooltip(f"Validação: {format_number(r['val_mse'])}")))
        series = [
            fch.LineChartData(points=train_pts, color=C.PRIMARY, stroke_width=3, point=False, curved=False),
            fch.LineChartData(points=val_pts, color=C.TERTIARY, stroke_width=3, point=False,
                              dash_pattern=[8, 5], curved=False),
        ]
        if best:
            series.append(fch.LineChartData(
                points=[fch.LineChartDataPoint(best["epoch"], self._y(best["val_mse"]),
                                               tooltip=_tooltip(f"Menor validação (época {best['epoch']})"))],
                color=C.SECONDARY, stroke_width=0,
                point=fch.ChartCirclePoint(radius=6, color=C.SECONDARY, stroke_color=C.SURFACE, stroke_width=2),
            ))
        self.chart.data_series = series

        values = [self._y(r["train_mse"]) for r in rows] + [self._y(r["val_mse"]) for r in rows]
        lo, hi = min(values), max(values)
        if self.log_scale:
            lo_t, hi_t = math.floor(lo), math.ceil(hi)
            if hi_t == lo_t:
                hi_t += 1
            ticks = [float(k) for k in range(lo_t, hi_t + 1)]
            labels = [_axis_label(k, "10" + str(int(k)).translate(_SUPERSCRIPT)) for k in ticks]
            title = "MSE (escala log)"
        else:
            ticks = nice_ticks(0.0, hi * 1.05 if hi > 0 else 1.0)
            labels = [_axis_label(t, format_number(t, 3)) for t in ticks]
            title = "MSE"
        self.chart.min_y, self.chart.max_y = ticks[0], ticks[-1]
        self.chart.left_axis = fch.ChartAxis(
            title=ft.Text(title, size=13, color=C.ON_SURFACE), title_size=22, labels=labels, label_size=56)

        max_x = max(total_epochs, rows[-1]["epoch"], 2)
        x_ticks = [t for t in nice_ticks(0, max_x, 8) if 1 <= t <= max_x] or [1]  # só na faixa visível
        self.chart.min_x, self.chart.max_x = 1, max_x
        self.chart.bottom_axis = fch.ChartAxis(
            title=ft.Text("Época", size=13, color=C.ON_SURFACE), title_size=22,
            labels=[_axis_label(t, str(int(t))) for t in x_ticks if t == int(t)], label_size=28)

        last = rows[-1]
        summary = (f"Gráfico do MSE por época. Época {last['epoch']} de {total_epochs}: "
                   f"treinamento {format_number(last['train_mse'])}, validação {format_number(last['val_mse'])}.")
        if best:
            summary += f" Menor MSE de validação na época {best['epoch']}: {format_number(best['val_mse'])}."
        self._semantics.label = summary


class ComparisonChart:
    """Valores desejados (y) × estimados pela rede (ŷ), amostra a amostra."""

    MAX_POINTS = 300

    def __init__(self, height: int = 300) -> None:
        self.chart = _base_chart(height)
        self.placeholder = _empty("Avalie um subconjunto para comparar desejado × estimado.", height)
        self._semantics = _semantic_wrapper(self.chart, "Gráfico desejado versus estimado, vazio.")
        self.legend = ft.Row(
            [
                _legend_item("Desejado (y), linha contínua", color=C.PRIMARY),
                _legend_item("Estimado pela rede (ŷ), tracejada", dashed=True, color=C.TERTIARY),
            ],
            wrap=True,
            spacing=20,
        )
        self.root = ft.Column([self.placeholder, self._semantics, self.legend], spacing=10, tight=True)

    def render(self, y_true: Sequence[float], y_pred: Sequence[float], *, output_name: str = "saída") -> None:
        n = min(len(y_true), len(y_pred))
        if n == 0:
            self._semantics.visible, self.placeholder.visible = False, True
            return
        self._semantics.visible, self.placeholder.visible = True, False
        idx = downsample_indices(n, self.MAX_POINTS)
        true_pts = [fch.LineChartDataPoint(i + 1, y_true[i],
                                           tooltip=_tooltip(f"Amostra {i + 1}\ny = {format_number(y_true[i])}"))
                    for i in idx]
        pred_pts = [fch.LineChartDataPoint(i + 1, y_pred[i], tooltip=_tooltip(f"ŷ = {format_number(y_pred[i])}"))
                    for i in idx]
        show_points = n <= 60
        dot = fch.ChartCirclePoint(radius=3, color=C.PRIMARY, stroke_width=0) if show_points else False
        dot2 = fch.ChartCirclePoint(radius=3, color=C.TERTIARY, stroke_width=0) if show_points else False
        self.chart.data_series = [
            fch.LineChartData(points=true_pts, color=C.PRIMARY, stroke_width=2, point=dot),
            fch.LineChartData(points=pred_pts, color=C.TERTIARY, stroke_width=2, point=dot2, dash_pattern=[6, 4]),
        ]
        lo = min(min(y_true[:n]), min(y_pred[:n]))
        hi = max(max(y_true[:n]), max(y_pred[:n]))
        pad = (hi - lo) * 0.05 or 1.0
        ticks = nice_ticks(lo - pad, hi + pad)
        self.chart.min_y, self.chart.max_y = ticks[0], ticks[-1]
        self.chart.left_axis = fch.ChartAxis(
            title=ft.Text(output_name, size=13, color=C.ON_SURFACE), title_size=22,
            labels=[_axis_label(t, format_number(t, 3)) for t in ticks], label_size=56)
        x_ticks = [t for t in nice_ticks(0, n, 8) if 1 <= t <= max(n, 2)] or [1]
        self.chart.min_x, self.chart.max_x = 1, max(n, 2)
        self.chart.bottom_axis = fch.ChartAxis(
            title=ft.Text("Amostra", size=13, color=C.ON_SURFACE), title_size=22,
            labels=[_axis_label(t, str(int(t))) for t in x_ticks if t == int(t)], label_size=28)
        self._semantics.label = (f"Gráfico desejado versus estimado para {n} amostras de {output_name}. "
                                 "As métricas numéricas estão acima do gráfico.")

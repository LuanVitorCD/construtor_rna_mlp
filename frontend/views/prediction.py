"""Aba 5: previsão com novos valores de entrada (somente propagação direta)."""

from __future__ import annotations

import flet as ft

from frontend.components.ui import Message, body, number_field, requirement_list, section
from frontend.context import AppContext, updates_page
from frontend.parsing import ParseError, format_number, parse_float, parse_values_line, to_input_text
from frontend.service_gateway import GatewayError
from frontend.texts import HELP

C = ft.Colors
HISTORY_SIZE = 10


class PredictionView:
    title = "Previsão"

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self._signature: tuple | None = None
        self._history: list[tuple[list[float], dict]] = []

        self.status = body("", muted=False)
        self.requirements = ft.Container()
        self.fields_holder = ft.ResponsiveRow(spacing=12, run_spacing=12)
        self.fields: list[ft.TextField] = []
        self.paste = ft.TextField(
            label="Colar uma linha de valores",
            hint_text="ex.: 5,1; 3,5; 1,4",
            helper="Separe com ; (ponto e vírgula), tab ou espaço.",
            on_submit=self._on_fill,
            expand=True,
            border_color=C.OUTLINE,
        )
        self.fill_btn = ft.OutlinedButton("Preencher campos", icon=ft.Icons.CONTENT_PASTE, on_click=self._on_fill)
        self.predict_btn = ft.FilledButton("Prever", icon=ft.Icons.CALCULATE, on_click=self._on_predict,
                                           style=ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=28, vertical=18)))
        self.clear_btn = ft.TextButton("Limpar", icon=ft.Icons.RESTART_ALT, on_click=self._on_clear)
        self.message = Message()
        self.result_holder = ft.Column(tight=True, spacing=12)
        self.history_holder = ft.Container()

        left = ft.Column(
            [
                section(
                    "Valores de entrada",
                    [
                        self.status,
                        self.requirements,
                        self.fields_holder,
                        ft.Row([self.paste, self.fill_btn], vertical_alignment=ft.CrossAxisAlignment.START,
                               spacing=12),
                        ft.Row([self.predict_btn, self.clear_btn], spacing=12, wrap=True),
                        self.message.root,
                        body(HELP["predict"]),
                        body(HELP["extrapolation"]),
                    ],
                    icon=ft.Icons.KEYBOARD,
                ),
            ],
            spacing=16,
            col={"xs": 12, "lg": 6},
        )
        right = ft.Column(
            [
                section("Resultado", [self.result_holder], icon=ft.Icons.CALCULATE),
                section("Previsões recentes", [self.history_holder], icon=ft.Icons.TABLE_CHART),
            ],
            spacing=16,
            col={"xs": 12, "lg": 6},
        )
        self.root = ft.ResponsiveRow([left, right], spacing=16, run_spacing=16)
        self.result_holder.controls = [body("Preencha as entradas e clique em “Prever”.")]
        self.history_holder.content = body("Nenhuma previsão ainda.")
        self.refresh()

    # ----------------------------------------------------------------- campos
    def _rebuild_fields(self) -> None:
        state = self.ctx.state
        names = state.input_names()
        old = [f.value for f in self.fields]
        self.fields = []
        for i, name in enumerate(names):
            tf = number_field(name, old[i] if i < len(old) else "", on_submit=self._on_predict,
                              col={"xs": 12, "sm": 6, "xl": 4}, autofocus=(i == 0))
            self.fields.append(tf)
        self.fields_holder.controls = self.fields

    def _on_fill(self, _e) -> None:
        if not self.fields:
            return
        try:
            values = parse_values_line(self.paste.value or "", expected=len(self.fields))
        except ParseError as err:
            self.paste.error = str(err)
            return
        self.paste.error = None
        for tf, v in zip(self.fields, values):
            tf.value = to_input_text(v)
            tf.error = None
        self.message.clear()

    def _on_clear(self, _e) -> None:
        for tf in self.fields:
            tf.value = ""
            tf.error = None
        self.paste.value = ""
        self.paste.error = None
        self.message.clear()

    # ------------------------------------------------------------------ prever
    @updates_page
    async def _on_predict(self, _e) -> None:
        state = self.ctx.state
        if not state.has_network or state.training:
            return
        values: list[float] = []
        errors = 0
        for tf in self.fields:
            try:
                values.append(parse_float(tf.value, field=tf.label or "Entrada"))  # type: ignore[arg-type]
                tf.error = None
            except ParseError as err:
                tf.error = str(err).split(": ", 1)[-1]
                errors += 1
        if errors:
            self.message.show("error", "Valores inválidos", f"{errors} campo(s) precisam de correção.")
            return
        try:
            result = await self.ctx.gateway.call("predict", values)
        except GatewayError as err:
            self.message.show("error", err.title, err.message)
            return
        result = dict(result or {})
        self.message.clear()
        self._render_result(values, result)
        self._history.insert(0, (values, result))
        del self._history[HISTORY_SIZE:]
        self._render_history()

    def _render_result(self, values: list[float], result: dict) -> None:
        state = self.ctx.state
        outputs = [float(v) for v in (result.get("outputs") or [])]
        names = state.output_names()
        controls: list[ft.Control] = []
        label = result.get("label")
        if label is not None:
            controls.append(ft.Semantics(
                live_region=True,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("Classe prevista", theme_style=ft.TextThemeStyle.LABEL_LARGE,
                                color=C.ON_PRIMARY_CONTAINER),
                        ft.Text(str(label), theme_style=ft.TextThemeStyle.HEADLINE_MEDIUM,
                                weight=ft.FontWeight.W_700, color=C.ON_PRIMARY_CONTAINER, selectable=True),
                    ], spacing=2, tight=True),
                    bgcolor=C.PRIMARY_CONTAINER, border_radius=10, padding=16,
                ),
            ))
        rows = []
        for i, v in enumerate(outputs):
            name = names[i] if i < len(names) else f"ŷ{i + 1}"
            rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(name)),
                                          ft.DataCell(ft.Text(format_number(v, 6), selectable=True,
                                                              weight=ft.FontWeight.W_600, color=C.PRIMARY))]))
        controls.append(ft.Semantics(
            live_region=label is None,
            label="Saídas da rede: " + "; ".join(
                f"{names[i] if i < len(names) else i + 1} = {format_number(v, 6)}" for i, v in enumerate(outputs)),
            content=ft.DataTable(
                columns=[ft.DataColumn(ft.Text("Saída")), ft.DataColumn(ft.Text("Valor estimado (ŷ)"), numeric=True)],
                rows=rows, column_spacing=32, horizontal_margin=8,
                heading_text_style=ft.TextStyle(weight=ft.FontWeight.W_600, color=C.ON_SURFACE),
            ),
        ))
        if not state.is_trained:
            controls.append(body("Atenção: a rede ainda não foi treinada; o resultado usa os pesos iniciais."))
        controls.append(body("Entradas usadas: " + "; ".join(format_number(v, 6) for v in values)))
        self.result_holder.controls = controls

    def _render_history(self) -> None:
        if not self._history:
            self.history_holder.content = body("Nenhuma previsão ainda.")
            return
        rows = []
        for values, result in self._history:
            outs = result.get("outputs") or []
            out_text = result.get("label") if result.get("label") is not None else \
                "; ".join(format_number(float(v), 5) for v in outs)
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text("; ".join(format_number(v, 5) for v in values))),
                ft.DataCell(ft.Text(str(out_text), weight=ft.FontWeight.W_600)),
            ]))
        self.history_holder.content = ft.Row([ft.DataTable(
            columns=[ft.DataColumn(ft.Text("Entradas")), ft.DataColumn(ft.Text("Saída"))],
            rows=rows, column_spacing=28, horizontal_margin=8,
            heading_text_style=ft.TextStyle(weight=ft.FontWeight.W_600, color=C.ON_SURFACE),
        )], scroll=ft.ScrollMode.AUTO)

    # ------------------------------------------------------------------ estado
    def refresh(self) -> None:
        state = self.ctx.state
        signature = (state.n_inputs, tuple(state.input_names()), state.model_origin, state.model_path)
        if signature != self._signature:
            self._signature = signature
            self._history = []
            self._render_history()
            self.result_holder.controls = [body("Preencha as entradas e clique em “Prever”.")]
            self._rebuild_fields()
        ready = state.has_network and not state.training
        for c in (self.predict_btn, self.fill_btn, self.paste, *self.fields):
            c.disabled = not ready
        if not state.has_network:
            self.status.value = ""
            self.requirements.content = requirement_list([("Rede criada (aba Arquitetura)", False)])
            return
        self.requirements.content = None
        origin = "Modelo carregado de arquivo" if state.model_origin == "loaded" else "Rede criada nesta sessão"
        trained = "treinada" if state.is_trained else "ainda não treinada"
        self.status.value = (f"{origin} ({trained}) · arquitetura {state.architecture_text()} · "
                             f"{state.n_inputs} entrada(s)")

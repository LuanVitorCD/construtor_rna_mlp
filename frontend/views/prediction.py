"""Aba Previsão: novos valores de entrada (somente propagação direta).

Dois jeitos de informar as entradas: digitar uma linha ou abrir um arquivo CSV com várias
linhas (``CsvPredictionPanel``, que pode esconder a coluna da saída real e comparar).
"""

from __future__ import annotations

import flet as ft

from frontend.components.ui import Message, body, number_field, requirement_list, section
from frontend.context import AppContext, updates_page
from frontend.parsing import ParseError, format_number, parse_float, parse_values_line, to_input_text
from frontend.service_gateway import GatewayError
from frontend.texts import HELP
from frontend.theme import SUCCESS
from frontend.views.prediction_csv import CsvPredictionPanel

C = ft.Colors
HISTORY_SIZE = 10
SOURCE_MANUAL, SOURCE_CSV = "manual", "csv"


class PredictionView:
    title = "Previsão"

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self._signature: tuple | None = None
        self._history: list[tuple[list[float], dict, list[float] | str | None]] = []

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
        self.expected_field = ft.TextField(
            label="Resultado real (opcional)",
            hint_text="Para comparar com o previsto",
            on_submit=self._on_predict,
            border_color=C.OUTLINE,
            width=600,
        )
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
                        ft.Divider(height=1, thickness=1, color=C.OUTLINE_VARIANT),
                        self.expected_field,
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
        self.manual_root = ft.ResponsiveRow([left, right], spacing=16, run_spacing=16)
        self.csv_panel = CsvPredictionPanel(ctx)
        self.source = ft.SegmentedButton(
            segments=[
                ft.Segment(value=SOURCE_MANUAL, label=ft.Text("Digitar valores"), icon=ft.Icon(ft.Icons.KEYBOARD)),
                ft.Segment(value=SOURCE_CSV, label=ft.Text("Arquivo CSV (várias linhas)"),
                           icon=ft.Icon(ft.Icons.UPLOAD_FILE)),
            ],
            selected=[SOURCE_MANUAL],
            allow_empty_selection=False,
            on_change=self._on_source_change,
        )
        self.root = ft.Column([ft.Row([self.source], wrap=True), self.manual_root, self.csv_panel.root], spacing=16)
        self._on_source_change(None)
        self.result_holder.controls = [body("Preencha as entradas e clique em “Prever”.")]
        self.history_holder.content = body("Nenhuma previsão ainda.")
        self.refresh()

    # ------------------------------------------------------------------ origem
    def _on_source_change(self, _e) -> None:
        csv_mode = (self.source.selected or [SOURCE_MANUAL])[0] == SOURCE_CSV
        self.manual_root.visible = not csv_mode
        self.csv_panel.root.visible = csv_mode

    def show_csv(self) -> None:
        self.source.selected = [SOURCE_CSV]
        self._on_source_change(None)

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

    def _sync_expected_field(self) -> None:
        state = self.ctx.state
        outs = state.output_names()
        if not state.has_network:
            return
        if state.task == "classification":
            classes = state.class_labels or []
            sample = f"ex.: {classes[0]}" if classes else "ex.: classe real"
            self.expected_field.label = f"{outs[0]} real (opcional)"
            self.expected_field.hint_text = f"{sample} · não vai para a rede; serve só para comparar"
            self.expected_field.helper = None
        else:
            n_out = state.n_outputs or 1
            if n_out == 1:
                self.expected_field.label = f"{outs[0]} real (opcional)"
                self.expected_field.hint_text = "ex.: 12,5 · não vai para a rede; serve só para comparar"
                self.expected_field.helper = None
            else:
                self.expected_field.label = f"{n_out} reais (opcional)"
                self.expected_field.hint_text = "ex.: 10,5; 20,0"
                self.expected_field.helper = f"Separe com ; na ordem: {', '.join(outs)}."

    def _parse_expected(self) -> list[float] | str | None:
        raw = (self.expected_field.value or "").strip()
        if not raw:
            self.expected_field.error = None
            return None
        state = self.ctx.state
        if state.task == "classification":
            self.expected_field.error = None
            return raw
        n_out = state.n_outputs or 1
        try:
            vals = parse_values_line(raw, expected=n_out)
        except ParseError as err:
            self.expected_field.error = str(err)
            raise
        self.expected_field.error = None
        return vals

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
        self.expected_field.value = ""
        self.expected_field.error = None
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
        expected: list[float] | str | None = None
        try:
            expected = self._parse_expected()
        except ParseError:
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
        self._render_result(values, result, expected)
        self._history.insert(0, (values, result, expected))
        del self._history[HISTORY_SIZE:]
        self._render_history()

    def _render_result(
        self,
        values: list[float],
        result: dict,
        expected: list[float] | str | None = None,
    ) -> None:
        state = self.ctx.state
        outputs = [float(v) for v in (result.get("outputs") or [])]
        names = state.output_names()
        controls: list[ft.Control] = []
        label = result.get("label")
        if label is not None:
            card_items: list[ft.Control] = [
                ft.Text("Classe prevista", theme_style=ft.TextThemeStyle.LABEL_LARGE,
                        color=C.ON_PRIMARY_CONTAINER),
                ft.Text(str(label), theme_style=ft.TextThemeStyle.HEADLINE_MEDIUM,
                        weight=ft.FontWeight.W_700, color=C.ON_PRIMARY_CONTAINER, selectable=True),
            ]
            if isinstance(expected, str):
                hit = expected.strip().lower() == str(label).strip().lower()
                badge_color = SUCCESS if hit else C.ERROR
                card_items.append(
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.CHECK_CIRCLE if hit else ft.Icons.CANCEL,
                                    color=badge_color, size=18),
                            ft.Text(
                                f"Real: {expected} ({'acerto' if hit else 'erro'})",
                                weight=ft.FontWeight.W_600,
                                color=C.ON_PRIMARY_CONTAINER,
                            ),
                        ],
                        spacing=6,
                        tight=True,
                    )
                )
            controls.append(ft.Semantics(
                live_region=True,
                content=ft.Container(
                    content=ft.Column(card_items, spacing=4, tight=True),
                    bgcolor=C.PRIMARY_CONTAINER, border_radius=10, padding=16,
                ),
            ))

        has_reg_expected = isinstance(expected, list) and len(expected) == len(outputs)
        columns = [ft.DataColumn(ft.Text("Saída")), ft.DataColumn(ft.Text("Valor estimado (ŷ)"), numeric=True)]
        if has_reg_expected:
            columns.extend([
                ft.DataColumn(ft.Text("Real (y)"), numeric=True),
                ft.DataColumn(ft.Text("Erro (ŷ − y)"), numeric=True, tooltip="Previsto menos real"),
            ])

        rows = []
        for i, v in enumerate(outputs):
            name = names[i] if i < len(names) else f"ŷ{i + 1}"
            cells = [
                ft.DataCell(ft.Text(name)),
                ft.DataCell(ft.Text(format_number(v, 6), selectable=True,
                                    weight=ft.FontWeight.W_600, color=C.PRIMARY)),
            ]
            if has_reg_expected:
                y_real = expected[i]  # type: ignore[index]
                err_val = v - y_real
                cells.extend([
                    ft.DataCell(ft.Text(format_number(y_real, 6), selectable=True)),
                    ft.DataCell(ft.Text(format_number(err_val, 5), selectable=True, weight=ft.FontWeight.W_600)),
                ])
            rows.append(ft.DataRow(cells=cells))

        controls.append(ft.Semantics(
            live_region=label is None,
            label="Saídas da rede: " + "; ".join(
                f"{names[i] if i < len(names) else i + 1} = {format_number(v, 6)}" for i, v in enumerate(outputs)),
            content=ft.Row(
                [
                    ft.DataTable(
                        columns=columns, rows=rows, column_spacing=24, horizontal_margin=8,
                        heading_text_style=ft.TextStyle(weight=ft.FontWeight.W_600, color=C.ON_SURFACE),
                    )
                ],
                scroll=ft.ScrollMode.AUTO,
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
        has_expected = any(exp is not None for _, _, exp in self._history)
        columns = [ft.DataColumn(ft.Text("Entradas")), ft.DataColumn(ft.Text("Previsto"))]
        if has_expected:
            columns.extend([ft.DataColumn(ft.Text("Real")), ft.DataColumn(ft.Text("Comparação"))])

        rows = []
        for values, result, expected in self._history:
            outs = result.get("outputs") or []
            label = result.get("label")
            out_text = label if label is not None else "; ".join(format_number(float(v), 5) for v in outs)
            cells = [
                ft.DataCell(ft.Text("; ".join(format_number(v, 5) for v in values))),
                ft.DataCell(ft.Text(str(out_text), weight=ft.FontWeight.W_600)),
            ]
            if has_expected:
                if expected is None:
                    cells.extend([ft.DataCell(ft.Text("—")), ft.DataCell(ft.Text("—"))])
                elif isinstance(expected, str):
                    hit = label is not None and expected.strip().lower() == str(label).strip().lower()
                    cells.append(ft.DataCell(ft.Text(expected)))
                    cells.append(ft.DataCell(ft.Row(
                        [
                            ft.Icon(ft.Icons.CHECK_CIRCLE if hit else ft.Icons.CANCEL,
                                    color=SUCCESS if hit else C.ERROR, size=18),
                            ft.Text("acerto" if hit else "erro"),
                        ],
                        spacing=6,
                        tight=True,
                    )))
                else:
                    real_txt = "; ".join(format_number(v, 5) for v in expected)
                    err_txt = "; ".join(format_number(float(p) - r, 4) for p, r in zip(outs, expected))
                    cells.extend([ft.DataCell(ft.Text(real_txt)), ft.DataCell(ft.Text(f"erro {err_txt}"))])
            rows.append(ft.DataRow(cells=cells))
        self.history_holder.content = ft.Row([ft.DataTable(
            columns=columns,
            rows=rows, column_spacing=24, horizontal_margin=8,
            heading_text_style=ft.TextStyle(weight=ft.FontWeight.W_600, color=C.ON_SURFACE),
        )], scroll=ft.ScrollMode.AUTO)

    # ------------------------------------------------------------------ estado
    def refresh(self) -> None:
        self.csv_panel.refresh()
        state = self.ctx.state
        # model_path só conta para modelo aberto (salvar a rede criada não apaga o histórico)
        loaded_path = state.model_path if state.model_origin == "loaded" else None
        signature = (state.n_inputs, tuple(state.input_names()), state.model_origin, loaded_path)
        if signature != self._signature:
            self._signature = signature
            self._history = []
            self._render_history()
            self.result_holder.controls = [body("Preencha as entradas e clique em “Prever”.")]
            self._rebuild_fields()
        self._sync_expected_field()
        ready = state.has_network and not state.training
        for c in (self.predict_btn, self.fill_btn, self.paste, self.expected_field, *self.fields):
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
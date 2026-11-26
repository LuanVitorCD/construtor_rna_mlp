"""Aba Previsão, modo "Arquivo CSV": prever todas as linhas de um arquivo.

A coluna com a saída real pode continuar no arquivo: marcada como "Saída (oculta)", ela
não vai para a rede e só aparece quando o usuário liga "Mostrar a saída real e comparar".
A lógica (sem Flet) fica em ``frontend/batch_prediction.py``.
"""

from __future__ import annotations

import os

import flet as ft

from frontend.batch_prediction import (
    ROLE_HIDDEN,
    ROLE_IGNORE,
    ROLE_INPUT,
    BatchError,
    ColumnPlan,
    CsvTable,
    compare_classification,
    compare_regression,
    comparison_problem,
    default_roles,
    export_rows,
    parse_inputs,
    plan_columns,
    read_table,
    write_csv,
)
from frontend.components.charts import ComparisonChart
from frontend.components.ui import Message, body, learn_more, metric, requirement_list, section
from frontend.context import AppContext, updates_page
from frontend.parsing import format_fixed, format_number, format_percent, format_r2
from frontend.service_gateway import GatewayError
from frontend.texts import HELP
from frontend.theme import SUCCESS

C = ft.Colors
TABLE_ROWS = 100  # linhas mostradas na tabela (o CSV salvo tem todas)

MSE_R2_EXPLANATION = (
    "MSE (Erro Quadrático Médio): média dos quadrados das diferenças entre o valor real (y) e o valor estimado pela rede (ŷ)."
    "Por depender da escala dos dados, seu valor deve ser avaliado de forma relativa; quanto menor em comparação a outros modelos, melhor o ajuste.\n\n"
    "R² (Coeficiente de Determinação): mede o quanto a rede explica a variação dos dados reais. "
    "1 indica ajuste perfeito (100%); 0 equivale a simplesmente prever a média (0%); "
    "valores negativos indicam desempenho pior que a média."
)

MSE_ACC_EXPLANATION = (
    "MSE (Erro Quadrático Médio): média dos quadrados das diferenças entre a saída esperada (codificada em 0 e 1) "
    "e a saída contínua estimada pela rede. Quanto mais próximo de 0, mais confiante e próxima da classe correta está a rede.\n\n"
    "Taxa de Acertos (Acurácia): percentual de amostras em que a classe prevista pela rede (maior saída) "
    "coincide com a classe real."
)

class CsvPredictionPanel:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.table: CsvTable | None = None
        self._roles: dict[str, str] = {}
        self._roles_key: tuple | None = None  # rede para a qual os papéis foram sugeridos
        self.plan: ColumnPlan | None = None
        self.outputs: list[list[float]] | None = None  # resultado da última previsão
        self.labels: list[str] | None = None
        self._used_inputs: list[str] | None = None
        self._result_key: tuple | None = None  # rede usada na última previsão
        self.comparison: dict | None = None

        # --- 1. arquivo
        self.pick_btn = ft.FilledButton("Escolher arquivo CSV…", icon=ft.Icons.UPLOAD_FILE, on_click=self._on_pick)
        self.path_text = ft.Text("Nenhum arquivo selecionado.", color=C.ON_SURFACE_VARIANT, selectable=True)
        self.file_message = Message()

        # --- 2. colunas
        self.columns_list = ft.Column(spacing=4, tight=True)
        self.role_counts = body("", muted=False)
        self.plan_message = Message()
        self.columns_section = section(
            "2. Papel de cada coluna",
            [
                body(HELP["predict_csv"]),
                ft.Container(self.columns_list, border=ft.Border.all(1, C.OUTLINE_VARIANT), border_radius=8,
                             padding=8),
                self.role_counts,
                self.plan_message.root,
            ],
            icon=ft.Icons.VIEW_COLUMN,
        )
        self.columns_section.visible = False

        # --- 3. prever
        self.requirements = ft.Container()
        self.predict_btn = ft.FilledButton("Prever todas as linhas", icon=ft.Icons.CALCULATE,
                                           on_click=self._on_predict,
                                           style=ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=24,
                                                                                             vertical=18)))
        self.busy = ft.ProgressRing(width=20, height=20, stroke_width=3, visible=False)
        self.predict_message = Message()

        # --- resultados
        self.export_btn = ft.OutlinedButton("Salvar resultados (CSV)", icon=ft.Icons.DOWNLOAD,
                                            on_click=self._on_export, disabled=True,
                                            tooltip="Grava todas as linhas: entradas, previsto e, se a "
                                                    "comparação estiver ligada, o real e o erro")
        self.compare_switch = ft.Switch(label="Mostrar a saída real (coluna oculta) e comparar", value=False,
                                        on_change=self._on_compare_toggle, disabled=True)
        self.compare_hint = body("")
        self.compare_message = Message()
        self.metrics = ft.ResponsiveRow(spacing=12, run_spacing=12)
        self.metrics_help_reg = learn_more("O que significam MSE e R²?", MSE_R2_EXPLANATION)
        self.metrics_help_cls = learn_more("O que significam MSE e taxa de acertos?", MSE_ACC_EXPLANATION)
        self.metrics_help_reg.visible = False
        self.metrics_help_cls.visible = False
        self.table_holder = ft.Column(tight=True, spacing=8)
        self.results_section = section(
            "Resultados",
            [
                self.compare_switch,
                self.compare_hint,
                self.compare_message.root,
                self.metrics,
                self.metrics_help_reg,
                self.metrics_help_cls,
            ],
            icon=ft.Icons.INSIGHTS,
            trailing=self.export_btn,
        )
        self.rows_section = section("Linha a linha", [self.table_holder], icon=ft.Icons.TABLE_CHART)
        self.output_select = ft.Dropdown(label="Saída exibida", visible=False, on_select=self._on_output_change,
                                         width=260, border_color=C.OUTLINE)
        self.chart = ComparisonChart()
        self.chart_section = section(
            "Desejado × estimado pela rede",
            [self.output_select, self.chart.root, learn_more("Posso confiar nessa comparação?",
                                                             HELP["compare_csv"])],
            icon=ft.Icons.SHOW_CHART,
            subtitle="Cada ponto é uma linha do arquivo; o desejado (y) vem da coluna oculta.",
        )
        self.chart_section.visible = False

        left = ft.Column(
            [
                section("1. Arquivo CSV", [ft.Row([self.pick_btn], wrap=True), self.path_text,
                                           self.file_message.root], icon=ft.Icons.FOLDER_OPEN),
                self.columns_section,
                section("3. Prever", [self.requirements,
                                      ft.Row([self.predict_btn, self.busy], spacing=12, wrap=True,
                                             vertical_alignment=ft.CrossAxisAlignment.CENTER),
                                      self.predict_message.root, body(HELP["predict"])],
                        icon=ft.Icons.CALCULATE),
            ],
            spacing=16,
            col={"xs": 12, "xl": 5},
        )
        right = ft.Column([self.results_section, self.chart_section, self.rows_section], spacing=16,
                          col={"xs": 12, "xl": 7})
        self.root = ft.ResponsiveRow([left, right], spacing=16, run_spacing=16)
        self._render_results()
        self.refresh()

    # ------------------------------------------------------------------ arquivo
    @updates_page
    async def _on_pick(self, _e) -> None:
        try:
            files = await self.ctx.file_picker.pick_files(
                dialog_title="Escolha o CSV com as linhas para prever",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["csv"],
            )
        except Exception as exc:  # noqa: BLE001
            self.file_message.show("error", "Não foi possível abrir o seletor de arquivos", str(exc))
            return
        if not files:
            return
        path = files[0].path
        if not path:
            self.file_message.show("error", "Caminho indisponível",
                                   "Este modo de execução não informa o caminho do arquivo. Use o app desktop.")
            return
        await self.open_csv(path)

    @updates_page
    async def open_csv(self, path: str) -> None:
        try:
            table = read_table(path)
        except BatchError as err:
            self.file_message.show("error", "Não foi possível usar este arquivo", str(err))
            return
        self.table = table
        self.path_text.value = path
        self.path_text.color = C.ON_SURFACE
        self.file_message.show("success", f"{format_number(len(table))} linha(s) e {len(table.columns)} coluna(s)",
                               f"Arquivo: {os.path.basename(path)}. Confira abaixo o papel de cada coluna.")
        self._clear_results()
        self._suggest_roles()
        self.refresh()

    # ------------------------------------------------------------------ colunas
    def _network_key(self) -> tuple:
        state = self.ctx.state
        return (state.n_inputs, state.n_outputs, state.task, tuple(state.input_names()))

    def _suggest_roles(self) -> None:
        state = self.ctx.state
        if not self.table or not state.has_network:
            return
        ins, outs = state.model_columns()
        expected = 1 if state.task == "classification" else (state.n_outputs or 0)
        self._roles = default_roles(self.table.columns, state.n_inputs or 0, ins, outs, expected)
        self._roles_key = self._network_key()
        self._render_columns()
        self._update_plan()

    def _render_columns(self) -> None:
        rows = []
        for col in self.table.columns if self.table else []:
            seg = ft.SegmentedButton(
                segments=[
                    ft.Segment(value=ROLE_INPUT, label=ft.Text("Entrada")),
                    ft.Segment(value=ROLE_HIDDEN, label=ft.Text("Saída (oculta)"),
                               tooltip="Resposta certa: não vai para a rede; serve só para comparar"),
                    ft.Segment(value=ROLE_IGNORE, label=ft.Text("Ignorar")),
                ],
                selected=[self._roles.get(col, ROLE_IGNORE)],
                allow_empty_selection=False,
                show_selected_icon=False,
                on_change=lambda e, c=col: self._on_role_change(c, e.control.selected),
                tooltip=f"Papel da coluna {col}",
            )
            rows.append(ft.Row(
                [ft.Text(col, expand=True, weight=ft.FontWeight.W_500, color=C.ON_SURFACE, selectable=True), seg],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ))
        self.columns_list.controls = rows
        self.columns_section.visible = bool(rows)

    def _on_role_change(self, col: str, selected: list[str]) -> None:
        self._roles[col] = (selected or [ROLE_IGNORE])[0]
        self._update_plan()
        if self.plan is None or self.plan.input_cols != self._used_inputs:
            self._clear_results()  # as entradas mudaram: a previsão anterior não vale mais
        else:
            self._update_comparison()  # só a coluna oculta mudou: recompara sem prever de novo
        self._render_results()
        self.refresh()

    def _update_plan(self) -> None:
        state = self.ctx.state
        self.plan = None
        if not self.table or not state.has_network:
            self.plan_message.clear()
            return
        n_in = state.n_inputs or 0
        marked = [c for c in self.table.columns if self._roles.get(c) == ROLE_INPUT]
        hidden = [c for c in self.table.columns if self._roles.get(c) == ROLE_HIDDEN]
        self.role_counts.value = (f"Entradas: {len(marked)} de {n_in} · Saídas ocultas: {len(hidden)} · "
                                  f"Ignoradas: {len(self.table.columns) - len(marked) - len(hidden)}")
        ins, outs = state.model_columns()
        try:
            self.plan = plan_columns(self.table.columns, self._roles, n_in, ins, outs)
        except BatchError as err:
            self.plan_message.show("error", "Colunas de entrada não fecham", str(err))
            return
        if self.plan.names_match is False:
            self.plan_message.show(
                "warning", "Colunas diferentes das do treinamento",
                f"A rede foi treinada com: {', '.join(ins or [])}. Ela vai receber as colunas marcadas na ordem "
                f"do arquivo: {', '.join(self.plan.input_cols)}. Confira se a ordem corresponde.")
        elif self.plan.reordered:
            self.plan_message.show("info", "Entradas na ordem do treinamento",
                                   "A rede vai receber: " + ", ".join(self.plan.input_cols) + ".")
        else:
            self.plan_message.clear()

    # ------------------------------------------------------------------ prever
    def _result_signature(self) -> tuple:
        """Muda quando a rede ou os pesos mudam (a previsão anterior fica desatualizada)."""
        state = self.ctx.state
        return (id(state.network), state.trained_epochs, state.normalization, id(state.split), state.task)

    @updates_page
    async def _on_predict(self, _e) -> None:
        state = self.ctx.state
        if not self.table or not state.has_network or state.training:
            return
        self._update_plan()
        if self.plan is None:
            self.predict_message.show("error", "Confira as colunas", "Marque as colunas de entrada da rede no passo 2.")
            return
        try:
            rows = parse_inputs(self.table, self.plan.input_cols)
        except BatchError as err:
            self.predict_message.show("error", "Valor inválido no arquivo", str(err))
            return
        self._set_busy(True)
        try:
            results = await self.ctx.gateway.predict_many(rows)
        except GatewayError as err:
            where = f"Linha {self.table.lines[err.row]} do arquivo: " if err.row is not None else ""
            self.predict_message.show("error", err.title, where + err.message)
            return
        finally:
            self._set_busy(False)
        self.outputs = [[float(v) for v in (r.get("outputs") or [])] for r in results]
        labels = [r.get("label") for r in results]
        self.labels = [str(lb) for lb in labels] if labels and all(lb is not None for lb in labels) else None
        self._used_inputs = list(self.plan.input_cols)
        self._result_key = self._result_signature()
        self.predict_message.show("success", f"{format_number(len(results))} linha(s) previstas",
                                  "Veja os resultados em “Resultados”." if not self.plan.hidden_cols else
                                  "A coluna oculta não foi usada. Ligue “Mostrar a saída real (coluna oculta) "
                                  "e comparar” para ver o erro.")
        self._update_comparison()
        self._render_results()
        self.refresh()

    def _set_busy(self, busy: bool) -> None:
        self.busy.visible = busy
        self.predict_btn.disabled = busy or not self._can_predict()
        self.pick_btn.disabled = busy
        self.ctx.update()

    def _can_predict(self) -> bool:
        state = self.ctx.state
        return bool(self.table and state.has_network and not state.training)

    def _clear_results(self) -> None:
        self.outputs = self.labels = self._used_inputs = self._result_key = None
        self.comparison = None
        self.compare_switch.value = False
        self.predict_message.clear()
        self.compare_message.clear()

    # --------------------------------------------------------------- comparar
    def _comparison_problem(self) -> str | None:
        state = self.ctx.state
        if not self.plan:
            return "Defina as colunas de entrada primeiro."
        return comparison_problem(state.task, self.plan.hidden_cols, state.n_outputs or 0)

    def _update_comparison(self) -> None:
        """Calcula a comparação com a coluna oculta (só é mostrada com o interruptor ligado)."""
        self.comparison = None
        self.compare_message.clear()
        if self.outputs is None or not self.table or self._comparison_problem():
            return
        try:
            if self.labels is not None:
                self.comparison = compare_classification(self.table, self.plan.hidden_cols[0], self.labels,
                                                         self.ctx.state.class_labels)
            else:
                self.comparison = compare_regression(self.table, self.plan.hidden_cols, self.outputs)
        except BatchError as err:
            self.compare_message.show("error", "Não dá para comparar", str(err))

    def _on_compare_toggle(self, _e) -> None:
        self._render_results()

    def _on_output_change(self, _e) -> None:
        self._render_chart()

    # ------------------------------------------------------------------ render
    @property
    def comparing(self) -> bool:
        return bool(self.compare_switch.value and self.comparison)

    def _render_results(self) -> None:
        has_results = self.outputs is not None
        self.export_btn.disabled = not has_results
        problem = self._comparison_problem() if has_results else None
        can_compare = has_results and not problem and self.comparison is not None
        self.compare_switch.disabled = not can_compare
        if not can_compare:
            self.compare_switch.value = False
        if not has_results:
            self.compare_hint.value = ""
        elif self.labels is None and len(self.ctx.state.output_names()) == 1 and not problem:
            out = self.ctx.state.output_names()[0]
            self.compare_hint.value = (f"Previsto (ŷ): {out}. Coluna oculta"
                                       + (" usada como resposta esperada (y): " if self.compare_switch.value
                                          else " (não usada na previsão): ")
                                       + ", ".join(self.plan.hidden_cols) + ".")
        elif problem:
            self.compare_hint.value = "Comparação indisponível: " + problem
        elif self.compare_switch.value:
            self.compare_hint.value = "Coluna oculta usada como resposta esperada: " + ", ".join(self.plan.hidden_cols)
        else:
            self.compare_hint.value = ("Coluna oculta (não usada na previsão): " + ", ".join(self.plan.hidden_cols)
                                       + ".")
        self.compare_hint.visible = bool(self.compare_hint.value)
        self._render_metrics()
        self._render_table()
        self._render_chart()

    def _render_metrics(self) -> None:
        if self.outputs is None:
            hint = body("Escolha um arquivo, confira as colunas e clique em “Prever todas as linhas”.")
            hint.col = {"xs": 12}
            self.metrics.controls = [hint]
            self.metrics_help_reg.visible = False
            self.metrics_help_cls.visible = False
            return
        if self.compare_message.level == "info":
            self.compare_message.clear()
        n = len(self.outputs)
        tiles = [metric("Linhas previstas", format_number(n))]
        is_cls = self.labels is not None
        self.metrics_help_reg.visible = self.comparing and not is_cls
        self.metrics_help_cls.visible = self.comparing and is_cls
        if self.comparing:
            cmp = self.comparison
            if is_cls:
                acc_tile = metric("Taxa de acertos", format_percent(100 * cmp["accuracy"]))
                acc_tile.tooltip = "acurácia"
                tiles.append(acc_tile)
                tiles.append(metric("Acertos", f"{cmp['correct']} de {cmp['n']}"))
                if cmp["unknown"]:
                    self.compare_message.show(
                        "info", "Classes que a rede não conhece",
                        f"{cmp['unknown']} linha(s) têm uma classe que não apareceu no treinamento; contam como erro.")
            else:
                mse_tile = metric("MSE", format_number(cmp["mse"]))
                mse_tile.tooltip = "Comparado à coluna oculta"
                tiles.append(mse_tile)
                if cmp["r2"] is not None:
                    r2_tile = metric("R²", format_fixed(cmp["r2"]), hint=format_r2(cmp["r2"], include_raw=False))
                    r2_tile.tooltip = "Coeficiente de determinação · 100% = perfeito · 0% = igual a prever a média"
                    tiles.append(r2_tile)
        self.metrics.controls = tiles

    def _render_table(self) -> None:
        self.rows_section.visible = self.outputs is not None
        if self.outputs is None or not self.table or not self.plan:
            self.table_holder.controls = []
            return
        state = self.ctx.state
        names = state.output_names()
        classification = self.labels is not None
        heading = ft.TextStyle(weight=ft.FontWeight.W_600, color=C.ON_SURFACE)
        columns = [ft.DataColumn(ft.Text("Linha"), numeric=True, tooltip="Linha no arquivo (1 = cabeçalho)")]
        columns += [ft.DataColumn(ft.Text(c), numeric=True) for c in self.plan.input_cols]
        single = len(names) == 1  # uma saída: cabeçalhos curtos (o nome aparece acima da tabela)
        if classification:
            columns.append(ft.DataColumn(ft.Text("Classe prevista")))
        else:
            columns += [ft.DataColumn(ft.Text("Previsto (ŷ)" if single else f"Previsto: {n}"), numeric=True,
                                      tooltip=f"Saída da rede para {n}") for n in names]
        if self.comparing:
            if classification:
                columns += [ft.DataColumn(ft.Text("Classe real")), ft.DataColumn(ft.Text("Resultado"))]
            else:
                for name, col in zip(names, self.plan.hidden_cols):
                    columns += [ft.DataColumn(ft.Text("Real (y)" if single else f"Real: {col}"), numeric=True,
                                              tooltip=f"Coluna oculta {col}"),
                                ft.DataColumn(ft.Text("Erro (ŷ − y)" if single else f"Erro: {name}"),
                                              numeric=True, tooltip="Previsto menos real")]
        in_idx = [self.table.columns.index(c) for c in self.plan.input_cols]
        rows = []
        for k in range(min(TABLE_ROWS, len(self.outputs))):
            raw = self.table.rows[k]
            cells = [ft.DataCell(ft.Text(str(self.table.lines[k]), color=C.ON_SURFACE_VARIANT))]
            cells += [ft.DataCell(ft.Text(format_number(float(raw[j]), 6))) for j in in_idx]
            if classification:
                cells.append(ft.DataCell(ft.Text(self.labels[k], weight=ft.FontWeight.W_600, color=C.PRIMARY)))
            else:
                cells += [ft.DataCell(ft.Text(format_number(v, 5), weight=ft.FontWeight.W_600, color=C.PRIMARY))
                          for v in self.outputs[k]]
            if self.comparing:
                cmp = self.comparison
                if classification:
                    hit = cmp["hits"][k]
                    cells.append(ft.DataCell(ft.Text(cmp["y_true"][k])))
                    cells.append(ft.DataCell(ft.Row(
                        [ft.Icon(ft.Icons.CHECK_CIRCLE if hit else ft.Icons.CANCEL,
                                 color=SUCCESS if hit else C.ERROR, size=18),
                         ft.Text("acerto" if hit else "erro", color=C.ON_SURFACE)],
                        spacing=6, tight=True)))
                else:
                    for p, t in zip(self.outputs[k], cmp["y_true"][k]):
                        cells += [ft.DataCell(ft.Text(format_number(t, 5))),
                                  ft.DataCell(ft.Text(format_number(p - t, 4)))]
            rows.append(ft.DataRow(cells=cells))
        table = ft.DataTable(columns=columns, rows=rows, column_spacing=20, horizontal_margin=8,
                             heading_text_style=heading, data_row_min_height=36, data_row_max_height=44)
        controls: list[ft.Control] = [ft.Row([table], scroll=ft.ScrollMode.AUTO)]
        if len(self.outputs) > TABLE_ROWS:
            controls.append(body(f"Mostrando as {TABLE_ROWS} primeiras de {format_number(len(self.outputs))} "
                                 "linhas. “Salvar resultados” grava todas."))
        if not state.is_trained:
            controls.append(body("Atenção: a rede ainda não foi treinada; o resultado usa os pesos iniciais."))
        self.table_holder.controls = controls

    def _render_chart(self) -> None:
        if not self.comparing or self.labels is not None:
            self.chart_section.visible = False
            return
        names = self.ctx.state.output_names()
        if len(names) > 1:
            self.output_select.options = [ft.DropdownOption(key=str(i), text=name) for i, name in enumerate(names)]
            if self.output_select.value is None or int(self.output_select.value) >= len(names):
                self.output_select.value = "0"
            self.output_select.visible = True
            idx = int(self.output_select.value)
        else:
            self.output_select.visible = False
            idx = 0
        y_true = [row[idx] for row in self.comparison["y_true"]]
        y_pred = [row[idx] for row in self.outputs]
        self.chart.render(y_true, y_pred, output_name=names[idx] if names else "saída")
        self.chart_section.visible = True

    # ------------------------------------------------------------------ salvar
    @updates_page
    async def _on_export(self, _e) -> None:
        if self.outputs is None or not self.table or not self.plan:
            return
        base = os.path.splitext(os.path.basename(self.table.path))[0]
        try:
            path = await self.ctx.file_picker.save_file(
                dialog_title="Salvar resultados da previsão",
                file_name=f"{base}_previsoes.csv",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["csv"],
            )
        except Exception as exc:  # noqa: BLE001
            self.ctx.toast("error", f"Não foi possível abrir o diálogo: {exc}")
            return
        if not path:
            return
        self.export(path)

    def export(self, path: str) -> bool:
        if not path.lower().endswith(".csv"):
            path += ".csv"
        header, rows = export_rows(self.table, self.plan, self.outputs, self.labels,
                                   self.ctx.state.output_names(), self.comparison if self.comparing else None)
        try:
            write_csv(path, header, rows)
        except OSError as exc:
            self.ctx.toast("error", f"Erro ao salvar: {exc}")
            return False
        self.ctx.toast("success", f"Resultados salvos em {path}")
        return True

    # ------------------------------------------------------------------ estado
    def refresh(self) -> None:
        state = self.ctx.state
        if self.table and state.has_network and self._network_key() != self._roles_key:
            self._clear_results()  # outra rede (ex.: recriada com outras entradas): sugere de novo
            self._suggest_roles()
            self._render_results()
        elif self.outputs is not None and self._result_signature() != self._result_key:
            self._clear_results()
            self.predict_message.show("info", "A rede mudou depois da previsão",
                                      "Clique em “Prever todas as linhas” de novo para usar a rede atual.")
            self._render_results()
        elif self.outputs is not None:
            self._render_metrics()
        ready = self._can_predict() and not self.busy.visible
        self.predict_btn.disabled = not ready
        self.pick_btn.disabled = self.busy.visible
        if not state.has_network:
            self.requirements.content = requirement_list([("Rede criada (aba Arquitetura)", False)])
        elif state.training:
            self.requirements.content = body("Aguarde o fim do treinamento para prever.")
        else:
            self.requirements.content = None
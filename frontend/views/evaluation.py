"""Aba 4: avaliação nos subconjuntos (MSE do teste ao final, como pede o enunciado).

Métricas sempre presentes (contrato do service): MSE (e o RMSE derivado dele);
na classificação, também acurácia e matriz de confusão.
Se o backend também devolver ``y_true``/``y_pred`` (extensão opcional), a tela mostra
MAE, R², erro relativo médio e o gráfico "desejado × estimado" (Ivan Nunes, cap. 5).
"""

from __future__ import annotations

import flet as ft

from frontend.analysis import column, regression_metrics, rmse
from frontend.components.charts import ComparisonChart
from frontend.components.ui import (
    Message,
    body,
    learn_more,
    metric,
    requirement_list,
    section,
)
from frontend.context import AppContext
from frontend.parsing import format_number, format_percent
from frontend.service_gateway import GatewayError
from frontend.texts import HELP, SUBSET_LABELS

C = ft.Colors


class EvaluationView:
    title = "Avaliação"

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.current: str | None = None

        self.subset = ft.SegmentedButton(
            segments=[ft.Segment(value=k, label=ft.Text(SUBSET_LABELS[k])) for k in ("test", "val", "train")],
            selected=["test"],
            allow_empty_selection=False,
            on_change=self._on_subset_change,
        )
        self.eval_btn = ft.FilledButton("Avaliar", icon=ft.Icons.ASSESSMENT, on_click=self._on_evaluate,
                                        style=ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=24, vertical=18)))
        self.eval_all_btn = ft.OutlinedButton("Avaliar os três subconjuntos", icon=ft.Icons.FACT_CHECK,
                                              on_click=self._on_evaluate_all)
        self.subset_hint = Message()
        self.requirements = ft.Container()
        self.message = Message()

        self.metrics = ft.ResponsiveRow(spacing=12, run_spacing=12)
        self.output_select = ft.Dropdown(label="Saída exibida", visible=False, on_select=self._on_output_change,
                                         width=260, border_color=C.OUTLINE)
        self.comparison = ComparisonChart()
        self.comparison_section = section(
            "Desejado × estimado pela rede",
            [self.output_select, self.comparison.root],
            icon=ft.Icons.SHOW_CHART,
            subtitle="Cada ponto é uma amostra do subconjunto avaliado.",
        )
        self.comparison_section.visible = False
        self.confusion_holder = ft.Column(tight=True, spacing=8)
        self.confusion_section = section("Matriz de confusão", [self.confusion_holder], icon=ft.Icons.CALL_SPLIT,
                                         subtitle="Linhas: classe real. Colunas: classe prevista. "
                                                  "A diagonal (em negrito) são os acertos.")
        self.confusion_section.visible = False
        self.summary_holder = ft.Container()

        left = ft.Column(
            [
                section(
                    "Escolha o subconjunto",
                    [self.subset, self.subset_hint.root, ft.Row([self.eval_btn, self.eval_all_btn], wrap=True,
                                                                spacing=12),
                     self.requirements, self.message.root],
                    icon=ft.Icons.ASSESSMENT,
                ),
                section("Resultados", [self.metrics, learn_more("O que significam MSE e RMSE?", HELP["mse"])],
                        icon=ft.Icons.INSIGHTS),
                section("Comparação entre subconjuntos", [self.summary_holder], icon=ft.Icons.TABLE_CHART,
                        subtitle="Treinamento muito melhor que validação/teste indica sobreajuste."),
            ],
            spacing=16,
            col={"xs": 12, "xl": 5},
        )
        right = ft.Column([self.comparison_section, self.confusion_section], spacing=16, col={"xs": 12, "xl": 7})
        self.root = ft.ResponsiveRow([left, right], spacing=16, run_spacing=16)
        self._on_subset_change(None)
        self.refresh()

    # ------------------------------------------------------------------ ações
    def _selected_subset(self) -> str:
        return (self.subset.selected or ["test"])[0]

    def _on_subset_change(self, _e) -> None:
        if self._selected_subset() == "test":
            self.subset_hint.show("info", "Conjunto de teste", HELP["test"])
        else:
            self.subset_hint.clear()

    async def _evaluate(self, subset: str) -> dict | None:
        try:
            result = await self.ctx.gateway.call("evaluate", subset)
        except GatewayError as err:
            self.message.show("error", err.title, err.message)
            return None
        result = dict(result or {})
        self.ctx.state.evaluations[subset] = result
        return result

    async def _on_evaluate(self, _e) -> None:
        subset = self._selected_subset()
        self._set_busy(True)
        try:
            result = await self._evaluate(subset)
        finally:
            self._set_busy(False)
        if result is None:
            return
        self.current = subset
        self.message.show("success", f"Avaliação: {SUBSET_LABELS[subset]}",
                          f"MSE = {format_number(result.get('mse'))}")
        self._render()

    async def _on_evaluate_all(self, _e) -> None:
        self._set_busy(True)
        try:
            for subset in ("train", "val", "test"):
                if await self._evaluate(subset) is None:
                    return
        finally:
            self._set_busy(False)
        self.current = self._selected_subset()
        self.message.show("success", "Três subconjuntos avaliados", "Veja a comparação abaixo.")
        self._render()

    def _set_busy(self, busy: bool) -> None:
        self.eval_btn.disabled = busy
        self.eval_all_btn.disabled = busy
        self.ctx.update()

    def _on_output_change(self, _e) -> None:
        self._render_comparison()

    # ----------------------------------------------------------------- render
    def _render(self) -> None:
        state = self.ctx.state
        result = state.evaluations.get(self.current or "")
        self._render_summary()
        if not result:
            self.metrics.controls = []
            self.comparison_section.visible = False
            self.confusion_section.visible = False
            return
        mse = result.get("mse")
        tiles = [
            metric("MSE", format_number(mse), hint=SUBSET_LABELS.get(self.current or "", "")),
            metric("RMSE (√MSE)", format_number(rmse(mse)), hint="na unidade da saída"),
        ]
        n = result.get("n_samples") or state.split_size(self.current or "")
        if n is not None:
            tiles.append(metric("Amostras", format_number(int(n))))
        if "accuracy" in result:
            tiles.append(metric("Taxa de acertos", format_percent(100 * float(result["accuracy"])),
                                hint="acurácia"))
        if state.task != "classification" and result.get("y_true") and result.get("y_pred"):
            extra = regression_metrics(column(result["y_true"], 0), column(result["y_pred"], 0)) \
                if (state.n_outputs or 1) == 1 else None
            if extra:
                tiles.append(metric("MAE", format_number(extra["mae"]), hint="erro absoluto médio"))
                tiles.append(metric("R²", format_number(extra["r2"], 3) if extra["r2"] is not None else "—",
                                    hint="1 = ajuste perfeito"))
                if extra["mean_rel_error_pct"] is not None:
                    tiles.append(metric("Erro relativo médio", format_percent(extra["mean_rel_error_pct"], 2)))
        self.metrics.controls = tiles
        self._render_comparison()
        self._render_confusion(result)

    def _render_comparison(self) -> None:
        state = self.ctx.state
        result = state.evaluations.get(self.current or "") or {}
        y_true, y_pred = result.get("y_true"), result.get("y_pred")
        if state.task == "classification" or not y_true or not y_pred:
            self.comparison_section.visible = False
            return
        names = state.output_names()
        if len(names) > 1:
            self.output_select.options = [ft.DropdownOption(key=str(i), text=name) for i, name in enumerate(names)]
            if self.output_select.value is None or int(self.output_select.value) >= len(names):
                self.output_select.value = "0"
            self.output_select.visible = True
            idx = int(self.output_select.value)
        else:
            self.output_select.visible = False
            idx = 0
        self.comparison.render(column(y_true, idx), column(y_pred, idx),
                               output_name=names[idx] if names else "saída")
        self.comparison_section.visible = True

    def _render_confusion(self, result: dict) -> None:
        matrix = result.get("confusion_matrix")
        if not matrix:
            self.confusion_section.visible = False
            return
        k = len(matrix)
        labels = result.get("classes") or self.ctx.state.class_labels or [str(i) for i in range(k)]
        labels = [str(x) for x in labels][:k]
        total = sum(sum(r) for r in matrix) or 1
        biggest = max(max(r) for r in matrix) or 1
        cell_w = 92

        def cell(content: ft.Control, *, bg=None, bold=False) -> ft.Container:
            return ft.Container(content=content, width=cell_w, height=48, alignment=ft.Alignment.CENTER,
                                bgcolor=bg, border=ft.Border.all(1, C.OUTLINE_VARIANT))

        header = [cell(ft.Text("Real \\ Prev.", size=12, color=C.ON_SURFACE_VARIANT))]
        header += [cell(ft.Text(lbl, weight=ft.FontWeight.W_600, color=C.ON_SURFACE, max_lines=2,
                                overflow=ft.TextOverflow.ELLIPSIS, tooltip=lbl)) for lbl in labels]
        rows = [ft.Row(header, spacing=0)]
        for i, row in enumerate(matrix):
            cells = [cell(ft.Text(labels[i], weight=ft.FontWeight.W_600, color=C.ON_SURFACE, max_lines=2,
                                  overflow=ft.TextOverflow.ELLIPSIS))]
            for j, value in enumerate(row):
                alpha = 0.06 + 0.29 * (value / biggest)  # máx. 0,35: texto continua legível
                cells.append(cell(
                    ft.Semantics(
                        label=f"Real {labels[i]}, previsto {labels[j]}: {value} amostras",
                        content=ft.Text(str(value), color=C.ON_SURFACE,
                                        weight=ft.FontWeight.BOLD if i == j else ft.FontWeight.NORMAL),
                    ),
                    bg=ft.Colors.with_opacity(alpha, C.PRIMARY),
                ))
            rows.append(ft.Row(cells, spacing=0))
        correct = sum(matrix[i][i] for i in range(k))
        self.confusion_holder.controls = [
            ft.Row([ft.Column(rows, spacing=0, tight=True)], scroll=ft.ScrollMode.AUTO),
            body(f"{correct} de {total} amostras classificadas corretamente.", muted=False),
        ]
        self.confusion_section.visible = True

    def _render_summary(self) -> None:
        evals = self.ctx.state.evaluations
        if not evals:
            self.summary_holder.content = body("Nenhuma avaliação ainda.")
            return
        has_acc = any("accuracy" in r for r in evals.values())
        columns = [ft.DataColumn(ft.Text("Subconjunto")), ft.DataColumn(ft.Text("MSE"), numeric=True),
                   ft.DataColumn(ft.Text("RMSE"), numeric=True)]
        if has_acc:
            columns.append(ft.DataColumn(ft.Text("Acertos"), numeric=True))
        rows = []
        for key in ("train", "val", "test"):
            r = evals.get(key)
            if not r:
                continue
            cells = [ft.DataCell(ft.Text(SUBSET_LABELS[key])),
                     ft.DataCell(ft.Text(format_number(r.get("mse")))),
                     ft.DataCell(ft.Text(format_number(rmse(r.get("mse")))))]
            if has_acc:
                acc = r.get("accuracy")
                cells.append(ft.DataCell(ft.Text(format_percent(100 * acc) if acc is not None else "—")))
            rows.append(ft.DataRow(cells=cells))
        self.summary_holder.content = ft.DataTable(
            columns=columns, rows=rows, column_spacing=28, horizontal_margin=8,
            heading_text_style=ft.TextStyle(weight=ft.FontWeight.W_600, color=C.ON_SURFACE),
        )

    def refresh(self) -> None:
        state = self.ctx.state
        ready = state.has_network and state.has_split
        self.eval_btn.disabled = state.training or not ready
        self.eval_all_btn.disabled = state.training or not ready
        if ready:
            self.requirements.content = None
            if not state.is_trained and not state.training:
                self.requirements.content = body("A rede ainda não foi treinada: a avaliação usará os pesos iniciais.")
        else:
            self.requirements.content = requirement_list([
                ("Rede criada ou carregada", state.has_network),
                ("Dataset carregado e dividido (aba Dados)", state.has_split),
            ])
        if not state.evaluations:
            self.current = None
        self._render()

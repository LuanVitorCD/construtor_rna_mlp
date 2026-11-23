"""Aba 4: avaliação sequencial nos subconjuntos (Treinamento -> Validação -> Teste).

Métricas devolvidas pelo ``MLPService.evaluate``: MSE e nº de amostras; na regressão,
R² e as saídas desejadas/estimadas (``y_true``/``y_pred``) para o gráfico "desejado ×
estimado" (Ivan Nunes, cap. 5); na classificação, acurácia e matriz de confusão.
Só aparecem as métricas vistas em aula (MSE, R², acurácia).
"""

from __future__ import annotations

import flet as ft

from frontend.analysis import column
from frontend.components.charts import ComparisonChart
from frontend.components.ui import (
    Message,
    body,
    learn_more,
    metric,
    requirement_list,
    section,
)
from frontend.context import AppContext, updates_page
from frontend.parsing import format_fixed, format_number, format_percent, format_r2
from frontend.service_gateway import GatewayError
from frontend.texts import SUBSET_LABELS
from frontend.theme import SUCCESS, WARNING

C = ft.Colors

SUBSET_ORDER = ("train", "val", "test")
SUBSET_ICONS = {
    "train": ft.Icons.SCHOOL,
    "val": ft.Icons.RULE,
    "test": ft.Icons.SCIENCE,
}

MSE_R2_EXPLANATION = (
    "MSE (Erro Quadrático Médio): média dos quadrados das diferenças entre o valor real (y) e o valor estimado pela rede (ŷ). "
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


class EvaluationView:
    title = "Avaliação"

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.current: str | None = None

        # Indicador visual da sequência (não clicável): Treinamento -> Validação -> Teste
        self.sequence_holder = ft.Row(spacing=8, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.eval_all_btn = ft.FilledButton(
            "Avaliar os três subconjuntos",
            icon=ft.Icons.FACT_CHECK,
            on_click=self._on_evaluate_all,
            style=ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=24, vertical=18)),
        )
        self.requirements = ft.Container()
        self.message = Message()
        self.summary_holder = ft.Container()

        # Abas no topo da coluna direita para alternar o subconjunto exibido no gráfico e resultados
        self.view_tabs = ft.SegmentedButton(
            segments=[
                ft.Segment(value=k, label=ft.Text(SUBSET_LABELS[k]), icon=ft.Icon(SUBSET_ICONS[k]))
                for k in SUBSET_ORDER
            ],
            selected=["test"],
            allow_empty_selection=False,
            on_change=self._on_tab_change,
        )

        self.output_select = ft.Dropdown(
            label="Saída exibida",
            visible=False,
            on_select=self._on_output_change,
            width=260,
            border_color=C.OUTLINE,
        )
        self.comparison = ComparisonChart()
        self.comparison_section = section(
            "Desejado × estimado pela rede",
            [self.output_select, self.comparison.root],
            icon=ft.Icons.SHOW_CHART,
            subtitle="Cada ponto é uma amostra do subconjunto avaliado.",
        )
        self.comparison_section.visible = False

        self.confusion_holder = ft.Column(tight=True, spacing=8)
        self.confusion_section = section(
            "Matriz de confusão",
            [self.confusion_holder],
            icon=ft.Icons.CALL_SPLIT,
            subtitle="Linhas: classe real. Colunas: classe prevista. "
                     "Diagonal (em negrito, verde): acertos. Fora dela (vermelho): erros.",
        )
        self.confusion_section.visible = False

        self.metrics = ft.ResponsiveRow(spacing=12, run_spacing=12)
        self.metrics_help_reg = learn_more("O que significam MSE e R²?", MSE_R2_EXPLANATION)
        self.metrics_help_cls = learn_more("O que significam MSE e taxa de acertos?", MSE_ACC_EXPLANATION)
        self.metrics_help_cls.visible = False
        self.results_section = section(
            "Resultados",
            [self.metrics, self.metrics_help_reg, self.metrics_help_cls],
            icon=ft.Icons.INSIGHTS,
        )

        left = ft.Column(
            [
                section(
                    "Avaliação dos subconjuntos",
                    [self.sequence_holder, self.eval_all_btn, self.requirements, self.message.root],
                    icon=ft.Icons.ASSESSMENT,
                    subtitle="Executa sequencialmente do treinamento ao teste.",
                ),
                section(
                    "Comparação entre subconjuntos",
                    [self.summary_holder],
                    icon=ft.Icons.TABLE_CHART,
                    subtitle="Treinamento muito melhor que validação/teste indica sobreajuste.",
                ),
            ],
            spacing=16,
            col={"xs": 12, "xl": 5},
        )
        right = ft.Column(
            [
                ft.Row([self.view_tabs], wrap=True),
                self.comparison_section,
                self.confusion_section,
                self.results_section,
            ],
            spacing=16,
            col={"xs": 12, "xl": 7},
        )
        self.root = ft.ResponsiveRow([left, right], spacing=16, run_spacing=16)
        self.refresh()

    # ------------------------------------------------------------------ ações
    def _on_tab_change(self, _e) -> None:
        selected = (self.view_tabs.selected or ["test"])[0]
        self.current = selected
        self._render()
        self.ctx.update()

    async def _evaluate(self, subset: str) -> dict | None:
        try:
            result = await self.ctx.gateway.call("evaluate", subset)
        except GatewayError as err:
            self.message.show("error", err.title, err.message)
            return None
        result = dict(result or {})
        self.ctx.state.evaluations[subset] = result
        return result

    @updates_page
    async def _on_evaluate_all(self, _e) -> None:
        self._set_busy(True)
        try:
            for subset in SUBSET_ORDER:
                if await self._evaluate(subset) is None:
                    return
                self.current = subset
                self.view_tabs.selected = [subset]
                self._render()
                self.ctx.update()
        finally:
            self._set_busy(False)
        self.current = "test"
        self.view_tabs.selected = ["test"]
        self.message.show("success", "Três subconjuntos avaliados", "Exibindo os resultados do conjunto de teste.")
        self.ctx.state.notify()
        self._render()

    def _set_busy(self, busy: bool) -> None:
        self.eval_all_btn.disabled = busy
        self.view_tabs.disabled = busy
        self.ctx.update()

    def _on_output_change(self, _e) -> None:
        self._render_comparison()

    # ----------------------------------------------------------------- render
    def _render_sequence(self) -> None:
        evals = self.ctx.state.evaluations
        items: list[ft.Control] = []
        for idx, key in enumerate(SUBSET_ORDER):
            done = key in evals
            color = SUCCESS if done else WARNING
            icon = ft.Icons.CHECK_CIRCLE_ROUNDED if done else ft.Icons.WARNING_AMBER_ROUNDED
            badge = ft.Container(
                content=ft.Row(
                    [
                        ft.Icon(icon, size=16, color=color),
                        ft.Text(
                            SUBSET_LABELS[key],
                            size=13,
                            weight=ft.FontWeight.W_600,
                            color=C.ON_SURFACE,
                        ),
                    ],
                    spacing=6,
                    tight=True,
                ),
                padding=ft.Padding.symmetric(horizontal=10, vertical=6),
                border_radius=8,
                bgcolor=ft.Colors.with_opacity(0.14, color),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.45, color)),
            )
            items.append(badge)
            if idx < len(SUBSET_ORDER) - 1:
                items.append(ft.Icon(ft.Icons.ARROW_FORWARD_ROUNDED, size=16, color=C.ON_SURFACE_VARIANT))
        self.sequence_holder.controls = items

    def _render(self) -> None:
        state = self.ctx.state
        self._render_sequence()
        self._render_summary()

        has_any_eval = bool(state.evaluations)
        self.view_tabs.visible = has_any_eval
        if self.current and self.view_tabs.selected != [self.current]:
            self.view_tabs.selected = [self.current]

        result = state.evaluations.get(self.current or "")
        if not result:
            self.metrics.controls = [body("Avalie os subconjuntos para visualizar os resultados.")]
            self.comparison_section.visible = False
            self.confusion_section.visible = False
            return

        subset_label = SUBSET_LABELS.get(self.current or "", "")
        self.results_section.title = f"Resultados · {subset_label}" if subset_label else "Resultados"

        mse = result.get("mse")
        mse_tile = metric("MSE", format_number(mse))
        mse_tile.tooltip = subset_label
        tiles = [mse_tile]
        n = result.get("n_samples") or state.split_size(self.current or "")
        if n is not None:
            tiles.append(metric("Amostras", format_number(int(n))))
        if "accuracy" in result:
            acc_tile = metric("Taxa de acertos", format_percent(100 * float(result["accuracy"])))
            acc_tile.tooltip = "acurácia"
            tiles.append(acc_tile)
        if result.get("r2") is not None:
            r2_tile = metric("R²", format_fixed(result["r2"]), hint=format_r2(result["r2"], include_raw=False))
            r2_tile.tooltip = "100% = perfeito · 0% = igual a prever a média"
            tiles.append(r2_tile)
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
                # Verde = acertos (diagonal), vermelho = erros; intensidade pela quantidade.
                # Máx. 0,35 de opacidade: o texto continua legível nos dois temas.
                alpha = 0.10 + 0.25 * (value / biggest)
                if value == 0:
                    bg = None
                else:
                    bg = ft.Colors.with_opacity(alpha, SUCCESS if i == j else C.ERROR)
                kind = "acertos" if i == j else "erros"
                cells.append(cell(
                    ft.Semantics(
                        label=f"Real {labels[i]}, previsto {labels[j]}: {value} amostras ({kind})",
                        content=ft.Text(str(value), color=C.ON_SURFACE,
                                        weight=ft.FontWeight.BOLD if i == j else ft.FontWeight.NORMAL),
                    ),
                    bg=bg,
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
        has_r2 = any(r.get("r2") is not None for r in evals.values())
        columns = [ft.DataColumn(ft.Text("Subconjunto")), ft.DataColumn(ft.Text("MSE"), numeric=True)]
        if has_r2:
            columns.append(ft.DataColumn(ft.Text("R²"), numeric=True, tooltip="Coeficiente de determinação"))
        if has_acc:
            columns.append(ft.DataColumn(ft.Text("Acertos"), numeric=True))
        rows = []
        for key in SUBSET_ORDER:
            r = evals.get(key)
            if not r:
                continue
            cells = [ft.DataCell(ft.Text(SUBSET_LABELS[key])),
                     ft.DataCell(ft.Text(format_number(r.get("mse"))))]
            if has_r2:
                r2 = r.get("r2")
                cells.append(ft.DataCell(ft.Text(format_fixed(r2))))
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
        ready = state.has_network and state.data_ready
        self.eval_all_btn.disabled = state.training or not ready
        if ready:
            self.requirements.content = None
            if not state.is_trained and not state.training:
                self.requirements.content = body("A rede ainda não foi treinada: a avaliação usará os pesos iniciais.")
        else:
            self.requirements.content = requirement_list([
                ("Rede criada (aba Arquitetura)", state.has_network),
                ("Dados aplicados à rede (aba Dados)"
                 + (" — há alterações não aplicadas" if state.data_dirty else ""), state.data_ready),
            ])
        if not state.evaluations:
            self.current = None
        elif self.current not in state.evaluations:
            self.current = "test" if "test" in state.evaluations else next(iter(state.evaluations))
        is_cls = state.task == "classification"
        self.metrics_help_reg.visible = not is_cls
        self.metrics_help_cls.visible = is_cls
        self._render()
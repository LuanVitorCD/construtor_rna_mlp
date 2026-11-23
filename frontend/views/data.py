"""Aba 1 do modo "Criar novo modelo": tarefa, CSV, colunas, divisão e normalização.

Vem antes da Arquitetura: com o CSV escolhido, a UI já sabe quantas entradas e saídas a
rede precisa (modo "Automático" da aba Arquitetura). As escolhas daqui são aplicadas à
rede (carregar → dividir → normalizar) automaticamente ao criar a rede, ou pelo botão
"Aplicar dados à rede" depois de alguma alteração.
"""

from __future__ import annotations

import os

import flet as ft

from frontend.analysis import samples_per_parameter
from frontend.components.ui import (
    Message,
    body,
    insight_box,
    learn_more,
    metric,
    number_field,
    section,
)
from frontend.context import AppContext, updates_page
from frontend.csv_utils import MAX_CLASSES, profile_columns
from frontend.parsing import ParseError, format_number, format_percent, parse_float, parse_int
from frontend.service_gateway import GatewayError
from frontend.texts import HELP, NORMALIZATION_LABELS, NORMALIZATION_SHORT, SUBSET_LABELS, TASK_LABELS
from frontend.theme import SUCCESS

C = ft.Colors
ROLE_INPUT, ROLE_OUTPUT, ROLE_IGNORE = "in", "out", "skip"


class DataView:
    title = "Dados"

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self._roles: dict[str, str] = {}
        self._profiles = None  # perfil das colunas escolhidas (csv_utils)
        state = ctx.state

        # --- tarefa
        self.task = ft.SegmentedButton(
            segments=[
                ft.Segment(value="regression", label=ft.Text(TASK_LABELS["regression"]),
                           icon=ft.Icon(ft.Icons.SHOW_CHART)),
                ft.Segment(value="classification", label=ft.Text(TASK_LABELS["classification"]),
                           icon=ft.Icon(ft.Icons.CALL_SPLIT)),
            ],
            selected=[state.task_choice],
            allow_empty_selection=False,
            on_change=self._on_task_change,
        )
        self.task_message = Message()

        # --- arquivo
        self.pick_btn = ft.FilledButton("Escolher arquivo CSV…", icon=ft.Icons.UPLOAD_FILE, on_click=self._on_pick,
                                        autofocus=True)
        self.path_text = ft.Text("Nenhum arquivo selecionado.", color=C.ON_SURFACE_VARIANT, selectable=True)
        self.file_message = Message()

        # --- colunas
        self.columns_list = ft.Column(spacing=4, tight=True)
        self.role_counts = body("", muted=False)
        self.selection_message = Message()
        self.columns_section = section(
            "3. Colunas de entrada e de saída",
            [
                body(HELP["csv"]),
                ft.Container(self.columns_list, border=ft.Border.all(1, C.OUTLINE_VARIANT), border_radius=8,
                             padding=8),
                self.role_counts,
                self.selection_message.root,
            ],
            icon=ft.Icons.TABLE_CHART,
        )
        self.columns_section.visible = False

        # --- divisão
        self.train_pct = number_field("Treinamento (%)", "70", width=150, on_change=self._on_pct_change)
        self.val_pct = number_field("Validação (%)", "15", width=150, on_change=self._on_pct_change)
        self.test_pct = number_field("Teste (%)", "15", width=150, on_change=self._on_pct_change)
        self.split_seed = number_field("Semente da divisão", "42", integer=True, width=180,
                                       helper="Em branco = aleatória", hint="aleatória",
                                       on_change=lambda _e: self._changed())
        self.pct_total = ft.Text("", weight=ft.FontWeight.W_600)

        # --- normalização
        self.norm_group = ft.RadioGroup(
            value="none",
            on_change=lambda _e: self._changed(),
            content=ft.Column(
                [ft.Radio(value="none", label=NORMALIZATION_LABELS[None]),
                 ft.Radio(value="minmax", label=NORMALIZATION_LABELS["minmax"]),
                 ft.Radio(value="zscore", label=NORMALIZATION_LABELS["zscore"])],
                spacing=0,
                tight=True,
            ),
        )

        # --- aplicar à rede
        self.apply_btn = ft.FilledButton("Aplicar dados à rede", icon=ft.Icons.DATASET, on_click=self._on_apply)
        self.apply_message = Message()

        # --- resultado
        self.result_holder = ft.Column(tight=True, spacing=12)
        self.result_section = section("Dados aplicados à rede", [self.result_holder],
                                      icon=ft.Icons.TABLE_CHART_OUTLINED)
        self.result_section.visible = False

        left = ft.Column(
            [
                section("1. Tipo de tarefa", [self.task, body(HELP["task"]), self.task_message.root],
                        icon=ft.Icons.TUNE),
                section("2. Arquivo CSV", [ft.Row([self.pick_btn], wrap=True), self.path_text,
                                           self.file_message.root], icon=ft.Icons.FOLDER_OPEN),
                self.columns_section,
            ],
            spacing=16,
            col={"xs": 12, "lg": 6},
        )
        right = ft.Column(
            [
                section(
                    "4. Divisão do dataset",
                    [
                        ft.Row([self.train_pct, self.val_pct, self.test_pct], wrap=True, spacing=12),
                        self.pct_total,
                        ft.Row([self.split_seed], wrap=True),
                        learn_more("Para que serve cada subconjunto?", HELP["split"]),
                    ],
                    icon=ft.Icons.CALL_SPLIT,
                ),
                section(
                    "5. Normalização das entradas (opcional)",
                    [self.norm_group, body(HELP["normalization"])],
                    icon=ft.Icons.TUNE,
                ),
                section(
                    "6. Aplicar à rede",
                    [body(HELP["apply_data"]), ft.Row([self.apply_btn], wrap=True, spacing=12),
                     self.apply_message.root],
                    icon=ft.Icons.DATASET,
                ),
                self.result_section,
            ],
            spacing=16,
            col={"xs": 12, "lg": 6},
        )
        self.root = ft.ResponsiveRow([left, right], spacing=16, run_spacing=16)
        self._on_pct_change(None)
        self.refresh()

    # ------------------------------------------------------------------ tarefa
    def _selected_task(self) -> str:
        return (self.task.selected or ["regression"])[0]

    def _on_task_change(self, _e) -> None:
        state = self.ctx.state
        state.task_choice = self._selected_task()
        self._sync_selection()
        self._changed(notify=False)
        state.notify()

    # ------------------------------------------------------------------ arquivo
    @updates_page
    async def _on_pick(self, _e) -> None:
        try:
            files = await self.ctx.file_picker.pick_files(
                dialog_title="Escolha o arquivo CSV",
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
        state = self.ctx.state
        try:
            columns = await self.ctx.gateway.call("get_csv_columns", path)
        except GatewayError as err:
            self.file_message.show("error", err.title, err.message)
            self.ctx.update()
            return
        state.on_csv_selected(path, list(columns))
        self.path_text.value = path
        self.path_text.color = C.ON_SURFACE
        self.file_message.show("success", f"{len(columns)} colunas encontradas",
                               f"Arquivo: {os.path.basename(path)}. Defina abaixo o papel de cada coluna.")
        self._default_roles(columns)
        self.columns_section.visible = True
        self._render_columns()
        self._sync_selection()
        self._changed(notify=False)
        state.notify()

    # ------------------------------------------------------------------ colunas
    def _default_roles(self, columns: list[str]) -> None:
        """Convenção comum: última coluna = saída (ou rótulo); demais = entradas."""
        self._roles = {col: (ROLE_OUTPUT if i == len(columns) - 1 else ROLE_INPUT) for i, col in enumerate(columns)}

    def _render_columns(self) -> None:
        rows = []
        for col in self.ctx.state.columns:
            seg = ft.SegmentedButton(
                segments=[
                    ft.Segment(value=ROLE_INPUT, label=ft.Text("Entrada")),
                    ft.Segment(value=ROLE_OUTPUT, label=ft.Text("Saída")),
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

    def _on_role_change(self, col: str, selected: list[str]) -> None:
        self._roles[col] = (selected or [ROLE_IGNORE])[0]
        self._sync_selection()
        self._changed(notify=False)
        self.ctx.state.notify()

    def _selected(self) -> tuple[list[str], list[str]]:
        cols = self.ctx.state.columns
        return ([c for c in cols if self._roles.get(c) == ROLE_INPUT],
                [c for c in cols if self._roles.get(c) == ROLE_OUTPUT])

    def _sync_selection(self) -> None:
        """Leva a seleção para o estado, lê o perfil das colunas e conta as classes."""
        state = self.ctx.state
        ins, outs = self._selected()
        state.input_cols, state.output_cols = ins, outs
        self._profiles = (profile_columns(state.csv_path, ins + outs)
                          if state.csv_path and (ins or outs) else None)
        labels: list[str] = []
        if state.task_choice == "classification" and len(outs) == 1 and self._profiles:
            labels = self._profiles[outs[0]].distinct
        state.class_labels_preview = labels
        self._update_selection_message()

    def _switch_task(self, task: str) -> None:
        """Correção rápida a partir da mensagem da seleção."""
        self.task.selected = [task]
        self._on_task_change(None)

    def _diagnose(self) -> tuple[str, str, str, tuple | None, bool]:
        """(nível, título, texto, ação rápida, bloqueia?) para a seleção atual de colunas."""
        state = self.ctx.state
        ins, outs = state.input_cols, state.output_cols
        profiles = self._profiles or {}
        if not ins or not outs:
            return ("warning", "Seleção incompleta",
                    "Marque ao menos uma coluna de entrada e uma de saída.", None, True)
        for col in ins:
            prof = profiles.get(col)
            if prof and not prof.numeric:
                return ("error", f"A entrada “{col}” não é numérica",
                        f"Exemplo de valor: “{prof.non_numeric}”. As entradas da rede precisam ser números: "
                        "marque esta coluna como Ignorar (ou como Saída, se ela for o rótulo de uma "
                        "classificação).", None, True)
        if state.task_choice == "regression":
            for col in outs:
                prof = profiles.get(col)
                if prof and not prof.numeric:
                    action = (("Mudar para Classificação", lambda _e: self._switch_task("classification"))
                              if len(outs) == 1 else None)
                    return ("error", f"A saída “{col}” não é numérica",
                            f"Exemplo de valor: “{prof.non_numeric}”. Na regressão a saída precisa ser um "
                            "número. Se esta coluna é o rótulo de uma classe, a tarefa certa é Classificação.",
                            action, True)
            return ("success", f"A rede precisa de {len(ins)} entrada(s) e {len(outs)} neurônio(s) de saída",
                    "A aba Arquitetura usa esses números no modo Automático.", None, False)

        # Classificação
        if len(outs) != 1:
            return ("warning", "Classificação usa 1 coluna de rótulo",
                    "Marque exatamente uma coluna de saída: a que contém a classe.", None, True)
        col, labels = outs[0], state.class_labels_preview
        prof = profiles.get(col)
        to_regression = (("Mudar para Regressão", lambda _e: self._switch_task("regression"))
                         if prof and prof.numeric else None)
        if not prof:
            return ("error", "Não foi possível ler a coluna de rótulo",
                    f"Confira o arquivo: a coluna “{col}” não pôde ser lida.", None, True)
        if len(labels) > MAX_CLASSES:
            return ("error", f"A coluna “{col}” parece contínua",
                    f"Ela tem mais de {MAX_CLASSES} valores diferentes. Para prever valores contínuos, "
                    "a tarefa certa é Regressão.", to_regression, True)
        if len(labels) < 2:
            return ("error", "Rótulo com menos de 2 classes",
                    f"A coluna “{col}” precisa ter ao menos 2 valores diferentes.", None, True)
        shown = ", ".join(labels[:8]) + ("…" if len(labels) > 8 else "")
        n_out = 1 if len(labels) == 2 else len(labels)
        how = "1 neurônio de saída (0/1)" if n_out == 1 else f"{n_out} neurônios de saída (one-hot)"
        if prof.numeric and len(labels) > 20:
            return ("warning", f"{len(labels)} classes numéricas: “{col}” parece uma coluna contínua",
                    f"Se “{col}” é um valor a prever (e não uma categoria), use Regressão. Se são mesmo "
                    f"categorias, a rede terá {len(ins)} entrada(s) e {how}.", to_regression, False)
        return ("success", f"A rede precisa de {len(ins)} entrada(s) e {how}",
                f"{len(labels)} classes em “{col}”: {shown}. A aba Arquitetura usa esses números no modo "
                "Automático.", None, False)

    def _update_selection_message(self) -> None:
        state = self.ctx.state
        self.role_counts.value = f"Entradas: {len(state.input_cols)} · Saídas: {len(state.output_cols)}"
        if not state.csv_path:
            state.selection_problem = None
            self.selection_message.clear()
            return
        level, title, text, action, blocking = self._diagnose()
        state.selection_problem = f"{title}. {text}" if blocking else None
        suggestion = state.suggested_io()
        if state.has_network and suggestion and suggestion != (state.n_inputs, state.n_outputs):
            level, title, action = "error", "Diferente da rede atual", ("Ir para Arquitetura",
                                                                        lambda _e: self.ctx.navigate(1))
            text = (f"Os dados pedem {suggestion[0]} entrada(s) e {suggestion[1]} saída(s); a rede atual tem "
                    f"{state.n_inputs} e {state.n_outputs}. Recrie a rede na aba Arquitetura (modo Automático).")
        self.selection_message.show(level, title, text, action)

    # ---------------------------------------------------------- alterações
    def _changed(self, *, notify: bool = True) -> None:
        """Alguma escolha mudou depois de os dados terem sido aplicados à rede."""
        state = self.ctx.state
        if (state.has_dataset or state.has_split) and not state.data_dirty:
            state.mark_data_dirty()
            if notify:
                state.notify()

    def _read_pcts(self, *, strict: bool) -> tuple[float, float, float] | None:
        values = []
        for tf, name in ((self.train_pct, "Treinamento"), (self.val_pct, "Validação"), (self.test_pct, "Teste")):
            try:
                values.append(parse_float(tf.value, field=name, minimum=0, maximum=100))
                tf.error = None
            except ParseError as err:
                if strict:
                    tf.error = str(err)
                return None
        return values[0], values[1], values[2]  # type: ignore[return-value]

    def _on_pct_change(self, e) -> None:
        pcts = self._read_pcts(strict=False)
        if pcts is None:
            self.pct_total.value = "Total: —"
            self.pct_total.color = C.ON_SURFACE_VARIANT
        else:
            total = sum(pcts)
            ok = abs(total - 100) < 1e-9
            self.pct_total.value = f"Total: {format_percent(total)} " + ("✓ soma 100%" if ok else "✗ precisa somar 100%")
            self.pct_total.color = SUCCESS if ok else C.ERROR
        if e is not None:
            self._changed()

    # ------------------------------------------------------------- aplicar
    def _apply_problem(self) -> str | None:
        state = self.ctx.state
        if not state.has_network:
            return "Crie a rede na aba Arquitetura: os dados são aplicados junto."
        if not state.data_selection_ready:
            return "Escolha o CSV e marque as colunas de entrada e de saída."
        if state.task_choice != state.task:
            return (f"A rede atual é de {TASK_LABELS[state.task]}, mas a tarefa escolhida é "
                    f"{TASK_LABELS[state.task_choice]}. Recrie a rede na aba Arquitetura.")
        return None

    async def prepare(self) -> tuple[bool, str | None]:
        """Carrega, divide e normaliza os dados escolhidos (``ctx.prepare_data``)."""
        state = self.ctx.state
        problem = self._apply_problem()
        if problem:
            self.apply_message.show("error", "Não foi possível aplicar os dados", problem)
            return False, problem
        pcts = self._read_pcts(strict=True)
        if pcts is None or abs(sum(pcts) - 100) > 1e-9:
            problem = "Os percentuais da divisão precisam ser números que somam 100%."
            self.apply_message.show("error", "Divisão inválida", problem)
            return False, problem
        try:
            seed = parse_int(self.split_seed.value, field="Semente", minimum=0, allow_empty=True)
            self.split_seed.error = None
        except ParseError as err:
            self.split_seed.error = str(err)
            self.apply_message.show("error", "Semente inválida", str(err))
            return False, str(err)
        kind = None if self.norm_group.value == "none" else self.norm_group.value
        ins, outs = state.input_cols, state.output_cols

        self.apply_btn.disabled = True
        self.ctx.update()
        try:
            info = await self.ctx.gateway.call("load_dataset", state.csv_path, ins, outs)
            sizes = await self.ctx.gateway.call("split_dataset", pcts[0], pcts[1], pcts[2], seed=seed)
            await self.ctx.gateway.call("set_normalization", kind)
        except GatewayError as err:
            state.clear_data()  # nada aplicado: treinar/avaliar ficam bloqueados
            self.apply_message.show("error", err.title, err.message)
            self._render_result()
            state.notify()
            return False, f"{err.title}: {err.message}"
        finally:
            self.apply_btn.disabled = False

        info = dict(info or {})
        info.setdefault("inputs", ins)
        info.setdefault("outputs", outs)
        state.on_dataset_loaded(info)
        state.on_split(dict(sizes or {}))
        state.normalization = kind
        rows = info.get("rows")
        self.apply_message.show(
            "success", "Dados aplicados à rede",
            f"{format_number(rows) if isinstance(rows, int) else rows} linhas · treinamento "
            f"{state.split_size('train')} · validação {state.split_size('val')} · teste "
            f"{state.split_size('test')} · normalização {NORMALIZATION_SHORT[kind]}.")
        self._render_result()
        state.notify()
        return True, None

    @updates_page
    async def _on_apply(self, _e) -> None:
        await self.prepare()

    def _render_result(self) -> None:
        state = self.ctx.state
        info = state.dataset
        if not info:
            self.result_holder.controls = []
            self.result_section.visible = False
            return
        cols = [*info.get("inputs", []), *info.get("outputs", [])]
        preview = info.get("preview") or []
        tiles = [metric("Linhas", format_number(info.get("rows")) if isinstance(info.get("rows"), int) else "—",
                        col={"xs": 6, "md": 3})]
        for key in ("train", "val", "test"):
            n = state.split_size(key)
            tiles.append(metric(SUBSET_LABELS[key], format_number(n) if n is not None else "—", hint="amostras",
                                col={"xs": 6, "md": 3}))
        controls: list[ft.Control] = [ft.ResponsiveRow(tiles, spacing=12, run_spacing=12)]
        insight = samples_per_parameter(state.split_size("train"), state.n_params)
        if insight:
            controls.append(insight_box(insight))
        if info.get("classes"):
            classes = [str(c) for c in info["classes"]]
            controls.append(body(f"Classes ({len(classes)}): " + ", ".join(classes), muted=False))
        if preview and cols:
            out_set = set(info.get("outputs", []))
            data_rows = []
            for row in preview[:8]:
                values = preview_row_values(row, info.get("inputs", []), info.get("outputs", []),
                                            info.get("classes"))
                data_rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(_cell(v))) for v in values]))
            table = ft.DataTable(
                columns=[ft.DataColumn(ft.Text(c + (" (saída)" if c in out_set else ""),
                                               weight=ft.FontWeight.W_600)) for c in cols],
                rows=data_rows,
                column_spacing=18,
                horizontal_margin=8,
                heading_row_color=C.SURFACE_CONTAINER_HIGH,
            )
            controls.append(body(f"Primeiras {len(data_rows)} linhas:"))
            controls.append(ft.Row([table], scroll=ft.ScrollMode.AUTO))
        self.result_holder.controls = controls
        self.result_section.visible = True

    # ------------------------------------------------------------- modelo base
    def apply_template(self, info) -> None:
        """Usa a configuração de um modelo salvo como base (tarefa, colunas, normalização)."""
        state = self.ctx.state
        state.task_choice = info.task
        self.task.selected = [info.task]
        self.norm_group.value = info.normalization or "none"
        ins = list(info.meta.get("input_cols") or [])
        outs = list(info.meta.get("output_cols") or [])
        if state.columns and ins and outs and set(ins + outs) <= set(state.columns):
            self._roles = {c: ROLE_INPUT if c in ins else ROLE_OUTPUT if c in outs else ROLE_IGNORE
                           for c in state.columns}
            self._render_columns()
        self._sync_selection()

    # ------------------------------------------------------------------ estado
    def refresh(self) -> None:
        state = self.ctx.state
        busy = state.training
        for control in (self.pick_btn, self.task, self.norm_group, self.train_pct, self.val_pct, self.test_pct,
                        self.split_seed):
            control.disabled = busy
        self.apply_btn.disabled = busy or self._apply_problem() is not None
        self.apply_btn.visible = state.has_network
        if state.has_network and state.task != state.task_choice:
            self.task_message.show("warning", "A tarefa mudou",
                                   f"A rede atual é de {TASK_LABELS[state.task]}. Recrie a rede na aba "
                                   "Arquitetura para usar a nova tarefa.",
                                   action=("Ir para Arquitetura", lambda _e: self.ctx.navigate(1)))
        else:
            self.task_message.clear()
        if state.data_dirty:
            if self.apply_message.level != "warning":
                self.apply_message.show("warning", "Alterações ainda não aplicadas",
                                        "Clique em “Aplicar dados à rede” para carregar e dividir de novo com "
                                        "as novas escolhas. Até lá, treinar e avaliar ficam bloqueados.")
        elif not state.has_dataset and self.apply_message.level in ("success", "warning"):
            self.apply_message.clear()
        if state.columns and self.columns_section.visible:
            self._update_selection_message()
        if not state.has_dataset:
            self.result_section.visible = False
        elif not self.result_section.visible:
            self._render_result()


def preview_row_values(row, inputs: list[str], outputs: list[str], classes: list[str] | None = None) -> list:
    """Valores de uma linha da prévia, na ordem entradas + saídas.

    Aceita o formato do MLPService (``{"x": [...], "y": [...]}``, com ``y`` codificado
    0/1 ou one-hot na classificação), um dict por nome de coluna ou uma lista simples.
    """
    n = len(inputs) + len(outputs)
    if isinstance(row, dict) and "x" in row and "y" in row:
        x = list(row["x"])
        y = list(row["y"])
        if classes and len(outputs) == 1 and len(classes) >= 2:
            if len(y) == 1 and len(classes) == 2:  # binária: limiar 0,5
                y = [classes[1] if float(y[0]) >= 0.5 else classes[0]]
            elif len(y) == len(classes):  # one-hot: maior saída
                y = [classes[max(range(len(y)), key=lambda i: float(y[i]))]]
        return (x + y)[:n]
    if isinstance(row, dict):
        return [row.get(c, "") for c in [*inputs, *outputs]]
    return list(row)[:n]


def _cell(value) -> str:
    if isinstance(value, float):
        return format_number(value, 6)
    return str(value)

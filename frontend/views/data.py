"""Aba 2: CSV, escolha de colunas, divisão treino/validação/teste e normalização."""

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
from frontend.context import AppContext
from frontend.parsing import ParseError, format_number, format_percent, parse_float, parse_int
from frontend.service_gateway import GatewayError
from frontend.texts import HELP, NORMALIZATION_LABELS, NORMALIZATION_SHORT, SUBSET_LABELS
from frontend.theme import SUCCESS

C = ft.Colors
ROLE_INPUT, ROLE_OUTPUT, ROLE_IGNORE = "in", "out", "skip"


class DataView:
    title = "Dados"

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self._roles: dict[str, str] = {}

        # --- arquivo
        self.pick_btn = ft.FilledButton("Escolher arquivo CSV…", icon=ft.Icons.UPLOAD_FILE, on_click=self._on_pick)
        self.path_text = ft.Text("Nenhum arquivo selecionado.", color=C.ON_SURFACE_VARIANT, selectable=True)
        self.file_message = Message()
        self.model_note = Message()

        # --- colunas
        self.columns_list = ft.Column(spacing=4, tight=True)
        self.role_counts = body("", muted=False)
        self.compat_message = Message()
        self.adjust_btn = ft.OutlinedButton("Ajustar arquitetura às colunas", icon=ft.Icons.AUTO_FIX_HIGH,
                                            on_click=self._on_adjust, visible=False)
        self.load_btn = ft.FilledButton("Carregar dataset", icon=ft.Icons.DATASET, on_click=self._on_load)
        self.load_message = Message()
        self.columns_section = section(
            "2. Colunas de entrada e de saída",
            [
                body(HELP["csv"]),
                ft.Container(self.columns_list, border=ft.Border.all(1, C.OUTLINE_VARIANT), border_radius=8,
                             padding=8),
                self.role_counts,
                self.compat_message.root,
                ft.Row([self.load_btn, self.adjust_btn], wrap=True, spacing=12),
                self.load_message.root,
            ],
            icon=ft.Icons.TABLE_CHART,
        )
        self.columns_section.visible = False

        # --- prévia
        self.preview_holder = ft.Column(tight=True, spacing=12)
        self.preview_section = section("Prévia do dataset", [self.preview_holder], icon=ft.Icons.TABLE_CHART_OUTLINED)
        self.preview_section.visible = False

        # --- divisão
        self.train_pct = number_field("Treinamento (%)", "70", width=150, on_change=self._on_pct_change)
        self.val_pct = number_field("Validação (%)", "15", width=150, on_change=self._on_pct_change)
        self.test_pct = number_field("Teste (%)", "15", width=150, on_change=self._on_pct_change)
        self.split_seed = number_field("Semente da divisão", "42", integer=True, width=180,
                                       helper="Em branco = aleatória", hint="aleatória")
        self.pct_total = ft.Text("", weight=ft.FontWeight.W_600)
        self.split_btn = ft.FilledButton("Dividir dataset", icon=ft.Icons.CALL_SPLIT, on_click=self._on_split)
        self.split_message = Message()
        self.split_result = ft.ResponsiveRow(spacing=12, run_spacing=12)
        self.split_insight = ft.Container()
        self.split_hint = ft.Row(
            [ft.Icon(ft.Icons.INFO_OUTLINE, color=C.ON_SURFACE_VARIANT, size=20),
             body("Disponível depois de “Carregar dataset” (passo 2).")],
            spacing=8,
        )

        # --- normalização
        self.norm_group = ft.RadioGroup(
            value="none",
            on_change=self._on_norm_change,
            content=ft.Column(
                [ft.Radio(value="none", label=NORMALIZATION_LABELS[None]),
                 ft.Radio(value="minmax", label=NORMALIZATION_LABELS["minmax"]),
                 ft.Radio(value="zscore", label=NORMALIZATION_LABELS["zscore"])],
                spacing=0,
                tight=True,
            ),
        )
        self.norm_message = Message()

        left = ft.Column(
            [
                section("1. Arquivo", [ft.Row([self.pick_btn], wrap=True), self.path_text, self.file_message.root,
                                       self.model_note.root],
                        icon=ft.Icons.FOLDER_OPEN),
                self.columns_section,
                self.preview_section,
            ],
            spacing=16,
            col={"xs": 12, "lg": 6},
        )
        right = ft.Column(
            [
                section(
                    "3. Divisão do dataset",
                    [
                        self.split_hint,
                        ft.Row([self.train_pct, self.val_pct, self.test_pct], wrap=True, spacing=12),
                        self.pct_total,
                        ft.Row([self.split_seed], wrap=True),
                        ft.Row([self.split_btn], wrap=True),
                        self.split_message.root,
                        self.split_result,
                        self.split_insight,
                        learn_more("Para que serve cada subconjunto?", HELP["split"]),
                    ],
                    icon=ft.Icons.CALL_SPLIT,
                ),
                section(
                    "4. Normalização das entradas (opcional)",
                    [self.norm_group, self.norm_message.root, body(HELP["normalization"])],
                    icon=ft.Icons.TUNE,
                ),
            ],
            spacing=16,
            col={"xs": 12, "lg": 6},
        )
        self.root = ft.ResponsiveRow([left, right], spacing=16, run_spacing=16)
        self._on_pct_change(None)
        self.refresh()

    # ------------------------------------------------------------------ arquivo
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
        self.load_message.clear()
        self.preview_section.visible = False
        self.columns_section.visible = True
        self._render_columns()
        state.notify()

    # ------------------------------------------------------------------ colunas
    def _default_roles(self, columns: list[str]) -> None:
        """Convenção comum: última coluna = saída; demais = entradas.

        Se a rede já existe, tenta respeitar o nº de entradas/saídas dela.
        """
        state = self.ctx.state
        n_out = 1
        if state.has_network and state.task == "regression" and state.n_outputs:
            n_out = min(state.n_outputs, max(1, len(columns) - 1))
        self._roles = {}
        for i, col in enumerate(columns):
            self._roles[col] = ROLE_OUTPUT if i >= len(columns) - n_out else ROLE_INPUT

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
        self._update_compat()

    def _on_role_change(self, col: str, selected: list[str]) -> None:
        self._roles[col] = (selected or [ROLE_IGNORE])[0]
        self._update_compat()

    def _selected(self) -> tuple[list[str], list[str]]:
        cols = self.ctx.state.columns
        return ([c for c in cols if self._roles.get(c) == ROLE_INPUT],
                [c for c in cols if self._roles.get(c) == ROLE_OUTPUT])

    def _update_compat(self) -> None:
        state = self.ctx.state
        ins, outs = self._selected()
        self.role_counts.value = f"Entradas: {len(ins)} · Saídas: {len(outs)}"
        self.adjust_btn.visible = False
        if not ins or not outs:
            self.compat_message.show("warning", "Seleção incompleta",
                                     "Marque ao menos uma coluna de entrada e uma de saída.")
            return
        if not state.has_network:
            self.compat_message.show("info", "Crie a rede primeiro",
                                     f"Com esta seleção, a rede precisa de {len(ins)} entrada(s). "
                                     "Use o botão abaixo para levar esses números à aba Arquitetura.")
            self.adjust_btn.visible = state.task == "regression"
            return
        if state.task == "classification":
            if len(outs) != 1:
                self.compat_message.show("warning", "Classificação usa 1 coluna de rótulo",
                                         "Marque exatamente uma coluna de saída (a classe).")
                return
            problems = [] if len(ins) == state.n_inputs else [
                f"a rede tem {state.n_inputs} entrada(s), a seleção tem {len(ins)}"]
            if problems:
                self.compat_message.show("error", "Incompatível com a rede", "; ".join(problems).capitalize() + ".")
            else:
                self.compat_message.show("info", "Entradas compatíveis",
                                         "Os neurônios de saída serão conferidos com o nº de classes ao carregar "
                                         "(binária: 1; multiclasse: 1 por classe).")
            return
        problems = []
        if len(ins) != state.n_inputs:
            problems.append(f"a rede tem {state.n_inputs} entrada(s), a seleção tem {len(ins)}")
        if len(outs) != state.n_outputs:
            problems.append(f"a rede tem {state.n_outputs} saída(s), a seleção tem {len(outs)}")
        if problems:
            self.compat_message.show("error", "Incompatível com a rede", "; ".join(problems).capitalize() + ".")
            self.adjust_btn.visible = True
        else:
            self.compat_message.show("success", "Compatível com a rede",
                                     f"{len(ins)} entrada(s) e {len(outs)} saída(s), como na arquitetura atual.")

    def _on_adjust(self, _e) -> None:
        ins, outs = self._selected()
        self.ctx.adjust_architecture(len(ins), len(outs))

    async def _on_load(self, _e) -> None:
        state = self.ctx.state
        ins, outs = self._selected()
        if not state.csv_path:
            return
        self.load_btn.disabled = True
        self.ctx.update()
        try:
            info = await self.ctx.gateway.call("load_dataset", state.csv_path, ins, outs)
        except GatewayError as err:
            self.load_message.show("error", err.title, err.message)
            return
        finally:
            self.load_btn.disabled = False
        info = dict(info or {})
        info.setdefault("inputs", ins)
        info.setdefault("outputs", outs)
        if state.task == "classification" and not info.get("classes"):
            # O MLPService guarda os rótulos em `meta`, sem devolvê-los em load_dataset.
            labels = self.ctx.gateway.read_meta().get("class_labels")
            if labels:
                info["classes"] = [str(c) for c in labels]
        state.input_cols, state.output_cols = ins, outs
        state.on_dataset_loaded(info)
        self.norm_group.value = "none"
        rows = info.get("rows")
        self.load_message.show("success", "Dataset carregado e compatível com a rede",
                               f"{format_number(rows) if isinstance(rows, int) else rows} linhas. "
                               "Próximo passo: dividir o dataset (ao lado).")
        self._render_preview(info)
        state.notify()

    def _render_preview(self, info: dict) -> None:
        cols = [*info.get("inputs", []), *info.get("outputs", [])]
        preview = info.get("preview") or []
        tiles = [metric("Linhas", format_number(info.get("rows")) if isinstance(info.get("rows"), int) else "—"),
                 metric("Entradas", str(len(info.get("inputs", [])))),
                 metric("Saídas", str(len(info.get("outputs", []))))]
        controls: list[ft.Control] = [ft.ResponsiveRow(tiles, spacing=12, run_spacing=12)]
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
        self.preview_holder.controls = controls
        self.preview_section.visible = True

    # ------------------------------------------------------------------ divisão
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

    def _on_pct_change(self, _e) -> None:
        pcts = self._read_pcts(strict=False)
        if pcts is None:
            self.pct_total.value = "Total: —"
            self.pct_total.color = C.ON_SURFACE_VARIANT
            return
        total = sum(pcts)
        ok = abs(total - 100) < 1e-9
        self.pct_total.value = f"Total: {format_percent(total)} " + ("✓ soma 100%" if ok else "✗ precisa somar 100%")
        self.pct_total.color = SUCCESS if ok else C.ERROR

    async def _on_split(self, _e) -> None:
        state = self.ctx.state
        pcts = self._read_pcts(strict=True)
        if pcts is None:
            self.split_message.show("error", "Percentuais inválidos", "Corrija os campos destacados.")
            return
        if abs(sum(pcts) - 100) > 1e-9:
            self.split_message.show("error", "Os percentuais precisam somar 100%",
                                    f"Soma atual: {format_percent(sum(pcts))}.")
            return
        try:
            seed = parse_int(self.split_seed.value, field="Semente", minimum=0, allow_empty=True)
            self.split_seed.error = None
        except ParseError as err:
            self.split_seed.error = str(err)
            return
        self.split_btn.disabled = True
        self.ctx.update()
        try:
            sizes = await self.ctx.gateway.call("split_dataset", pcts[0], pcts[1], pcts[2], seed=seed)
            state.on_split(dict(sizes or {}))
            kind = None if self.norm_group.value == "none" else self.norm_group.value
            if kind:  # reaplica a normalização escolhida sobre a nova divisão
                await self.ctx.gateway.call("set_normalization", kind)
            state.normalization = kind
        except GatewayError as err:
            self.split_message.show("error", err.title, err.message)
            return
        finally:
            self.split_btn.disabled = False
        self.split_message.show("success", "Dataset dividido",
                                "Normalização: " + NORMALIZATION_SHORT[state.normalization] +
                                ". Próximo passo: aba Treinamento.")
        self._render_split()
        state.notify()

    def _render_split(self) -> None:
        state = self.ctx.state
        tiles = []
        for key in ("train", "val", "test"):
            n = state.split_size(key)
            tiles.append(metric(SUBSET_LABELS[key], format_number(n) if n is not None else "—",
                                hint="amostras", col={"xs": 4}))
        self.split_result.controls = tiles
        insight = samples_per_parameter(state.split_size("train"), state.n_params)
        self.split_insight.content = insight_box(insight) if insight else None

    async def _on_norm_change(self, _e) -> None:
        state = self.ctx.state
        kind = None if self.norm_group.value == "none" else self.norm_group.value
        if not state.has_split:
            self.norm_message.show("info", "Será aplicada ao dividir",
                                   "A normalização usa parâmetros do subconjunto de treinamento, "
                                   "então é aplicada logo após a divisão.")
            return
        try:
            await self.ctx.gateway.call("set_normalization", kind)
        except GatewayError as err:
            self.norm_message.show("error", err.title, err.message)
            self.norm_group.value = "none" if state.normalization is None else state.normalization
            return
        state.normalization = kind
        self.norm_message.show("success", "Normalização aplicada",
                               f"{NORMALIZATION_SHORT[kind]} (parâmetros calculados no treinamento).")
        state.notify()

    # ------------------------------------------------------------------ estado
    def refresh(self) -> None:
        state = self.ctx.state
        busy = state.training
        self.pick_btn.disabled = busy
        self.load_btn.disabled = busy or not state.csv_path
        self.split_btn.disabled = busy or not state.has_dataset
        self.norm_group.disabled = busy or not state.has_dataset
        self.split_hint.visible = not state.has_dataset
        if state.model_origin == "loaded" and state.loaded_normalization:
            kind = NORMALIZATION_SHORT[state.loaded_normalization]
            self.model_note.show(
                "warning", f"O modelo aberto foi treinado com normalização {kind}",
                "Ao carregar um dataset, o backend descarta a normalização salva e a recalcula a partir "
                f"do novo subconjunto de treinamento (isso vale também para as previsões). Escolha {kind} "
                "no passo 4 para avaliar o modelo em condições equivalentes.")
        else:
            self.model_note.clear()
        if state.columns and self.columns_section.visible:
            self._update_compat()
        if not state.has_dataset:
            self.preview_section.visible = False
            if self.load_message.level == "success":  # ex.: a rede foi recriada
                self.load_message.show("info", "Carregue o dataset novamente",
                                       "A rede mudou: clique em “Carregar dataset” para conferir a compatibilidade.")
        if not state.has_split:
            self.split_result.controls = []
            self.split_insight.content = None
            if not state.has_dataset:
                self.split_message.clear()
                self.norm_message.clear()
                self.norm_group.value = "none"


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

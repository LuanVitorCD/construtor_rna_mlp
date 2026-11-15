"""Aba 1: definir a arquitetura da MLP sem tocar no código (requisito 2.4 do enunciado)."""

from __future__ import annotations

import flet as ft

from frontend.analysis import count_parameters, output_activation_warning, suggest_hidden_neurons
from frontend.components.network_diagram import NetworkDiagram
from frontend.components.ui import (
    Message,
    activation_dropdown,
    body,
    learn_more,
    number_field,
    section,
)
from frontend.context import AppContext
from frontend.parsing import ParseError, format_number, parse_int
from frontend.service_gateway import GatewayError
from frontend.texts import ACTIVATION_RANGES, HELP, TASK_LABELS, activation_label

C = ft.Colors
MAX_HIDDEN = 6


class _HiddenRow:
    def __init__(self, view: "ArchitectureView", neurons: int, activation: str) -> None:
        self.view = view
        self.label = ft.Text("", weight=ft.FontWeight.W_500, color=C.ON_SURFACE)
        self.neurons = number_field("Neurônios", str(neurons), integer=True, width=140,
                                    on_change=view._on_draft_change)
        self.activation = activation_dropdown(activation, view.ctx.state.activations,
                                              on_select=view._on_draft_change)
        self.activation.expand = True
        self.remove = ft.IconButton(ft.Icons.DELETE_OUTLINE, on_click=self._on_remove, icon_color=C.ERROR)
        self.row = ft.Column(
            [
                self.label,
                ft.Row([self.neurons, self.activation, self.remove], spacing=12,
                       vertical_alignment=ft.CrossAxisAlignment.START),
            ],
            spacing=6,
            tight=True,
        )

    def set_index(self, i: int) -> None:
        self.label.value = f"Camada oculta {i}"
        self.remove.tooltip = f"Remover camada oculta {i}"

    def _on_remove(self, _e) -> None:
        self.view._remove_hidden(self)


class ArchitectureView:
    title = "Arquitetura"

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        state = ctx.state

        self.task = ft.SegmentedButton(
            segments=[
                ft.Segment(value="regression", label=ft.Text(TASK_LABELS["regression"]),
                           icon=ft.Icon(ft.Icons.SHOW_CHART)),
                ft.Segment(value="classification", label=ft.Text(TASK_LABELS["classification"]),
                           icon=ft.Icon(ft.Icons.CALL_SPLIT)),
            ],
            selected=[state.task],
            allow_empty_selection=False,
            allow_multiple_selection=False,
            on_change=self._on_task_change,
        )
        self.n_inputs = number_field("Nº de entradas", "2", integer=True, width=180,
                                     helper=None, on_change=self._on_draft_change, autofocus=True,
                                     tooltip=HELP["inputs"])
        self.hidden_rows: list[_HiddenRow] = []
        self.hidden_column = ft.Column(spacing=12, tight=True)
        self.add_hidden_btn = ft.OutlinedButton("Adicionar camada oculta", icon=ft.Icons.ADD,
                                                on_click=self._on_add_hidden)
        self.suggest_btn = ft.TextButton("Sugerir neurônios (2n + 1)", icon=ft.Icons.AUTO_FIX_HIGH,
                                         on_click=self._on_suggest, tooltip="Regra de Kolmogorov (Ivan Nunes, cap. 5)")
        self.out_neurons = number_field("Neurônios", "1", integer=True, width=140,
                                        on_change=self._on_draft_change)
        self.out_activation = activation_dropdown("linear", state.activations,
                                                  on_select=self._on_draft_change)
        self.out_activation.expand = True
        self.out_help = body("")
        self.out_warning = Message()
        self.seed = number_field("Semente (seed)", "42", integer=True, width=180, hint="aleatória",
                                 helper="Em branco = aleatória", on_change=self._on_draft_change)
        self.param_count = body("", muted=False)
        self.create_btn = ft.FilledButton("Criar rede", icon=ft.Icons.HUB, on_click=self._on_create,
                                          style=ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=24, vertical=18)))
        self.message = Message()

        self.diagram = NetworkDiagram()
        self.summary_holder = ft.Column(tight=True, spacing=8)

        self._add_hidden(3, "tanh")

        form = ft.Column(
            [
                section("1. Tipo de tarefa", [self.task, body(HELP["task"])], icon=ft.Icons.TUNE),
                section(
                    "2. Camadas",
                    [
                        ft.Row([self.n_inputs], wrap=True),
                        ft.Text("Camadas ocultas", theme_style=ft.TextThemeStyle.TITLE_SMALL, color=C.ON_SURFACE),
                        self.hidden_column,
                        ft.Row([self.add_hidden_btn, self.suggest_btn], wrap=True, spacing=8),
                        ft.Divider(color=C.OUTLINE_VARIANT),
                        ft.Text("Camada de saída", theme_style=ft.TextThemeStyle.TITLE_SMALL, color=C.ON_SURFACE),
                        ft.Row([self.out_neurons, self.out_activation, ft.Container(width=48)], spacing=12,
                               vertical_alignment=ft.CrossAxisAlignment.START),
                        self.out_help,
                        self.out_warning.root,
                        learn_more("Como escolher camadas e neurônios?", HELP["hidden"]),
                    ],
                    icon=ft.Icons.ACCOUNT_TREE,
                ),
                section(
                    "3. Inicialização",
                    [ft.Row([self.seed], wrap=True), body(HELP["init"])],
                    icon=ft.Icons.SCIENCE,
                ),
                ft.Row([self.create_btn, self.param_count], spacing=16, wrap=True,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.message.root,
            ],
            spacing=16,
            col={"xs": 12, "lg": 6},
        )
        preview = ft.Column(
            [
                section("Pré-visualização", [self.diagram.root],
                        subtitle="Atualiza enquanto você edita. Só vale depois de clicar em “Criar rede”.",
                        icon=ft.Icons.HUB_OUTLINED),
                section("Rede atual", [self.summary_holder], icon=ft.Icons.FACT_CHECK),
            ],
            spacing=16,
            col={"xs": 12, "lg": 6},
        )
        self.root = ft.ResponsiveRow([form, preview], spacing=16, run_spacing=16)
        self._apply_task_texts()
        self._on_draft_change(None)
        self.refresh()

    # ------------------------------------------------------------ camadas ocultas
    def _add_hidden(self, neurons: int, activation: str) -> None:
        row = _HiddenRow(self, neurons, activation)
        self.hidden_rows.append(row)
        self._reindex()

    def _remove_hidden(self, row: _HiddenRow) -> None:
        self.hidden_rows.remove(row)
        self._reindex()
        self._on_draft_change(None)
        self.ctx.update()

    def _reindex(self) -> None:
        for i, r in enumerate(self.hidden_rows, start=1):
            r.set_index(i)
        self.hidden_column.controls = [r.row for r in self.hidden_rows] or [
            body("Sem camadas ocultas: a rede vira um único neurônio por saída (como o ADALINE).")
        ]
        self.add_hidden_btn.disabled = len(self.hidden_rows) >= MAX_HIDDEN

    def _on_add_hidden(self, _e) -> None:
        last = self.hidden_rows[-1] if self.hidden_rows else None
        neurons = last.neurons.value if last else "3"
        activation = last.activation.value if last else "tanh"
        try:
            n = int(neurons)
        except ValueError:
            n = 3
        self._add_hidden(n, activation)
        self._on_draft_change(None)

    def _on_suggest(self, _e) -> None:
        try:
            n_in = parse_int(self.n_inputs.value, field="Nº de entradas", minimum=1)
        except ParseError as err:
            self.n_inputs.error = str(err)
            return
        if not self.hidden_rows:
            self._add_hidden(3, "tanh")
        suggestion = suggest_hidden_neurons(n_in)  # type: ignore[arg-type]
        for r in self.hidden_rows:
            r.neurons.value = str(suggestion)
        self._on_draft_change(None)
        self.ctx.toast("info", f"Sugestão aplicada: {suggestion} neurônios por camada oculta (2 × {n_in} + 1).")

    # ------------------------------------------------------------------ tarefa
    def _selected_task(self) -> str:
        return (self.task.selected or ["regression"])[0]

    def _on_task_change(self, _e) -> None:
        task = self._selected_task()
        self.out_activation.value = "linear" if task == "regression" else "sigmoid"
        self._apply_task_texts()
        self._on_draft_change(None)

    def _apply_task_texts(self) -> None:
        task = self._selected_task()
        self.out_help.value = HELP["output_regression"] if task == "regression" else HELP["output_classification"]

    # --------------------------------------------------------------- rascunho
    def _draft(self, *, strict: bool) -> tuple[int, list[dict], int | None] | None:
        """Lê o formulário. ``strict`` marca erros nos campos; senão só devolve None."""
        fields_ok = True

        def read_int(tf: ft.TextField, name: str, *, allow_empty: bool = False, minimum: int = 1):
            nonlocal fields_ok
            try:
                value = parse_int(tf.value, field=name, minimum=minimum, maximum=10_000, allow_empty=allow_empty)
                tf.error = None
                return value
            except ParseError as err:
                fields_ok = False
                if strict:  # o rótulo do campo já diz qual é; mostra só o problema
                    msg = str(err).split(": ", 1)[-1]
                    tf.error = msg[:1].upper() + msg[1:]
                return None

        n_in = read_int(self.n_inputs, "Nº de entradas")
        layers = []
        for i, r in enumerate(self.hidden_rows, start=1):
            n = read_int(r.neurons, f"Neurônios da oculta {i}")
            layers.append({"neurons": n, "activation": r.activation.value})
        n_out = read_int(self.out_neurons, "Neurônios de saída")
        layers.append({"neurons": n_out, "activation": self.out_activation.value})
        seed = read_int(self.seed, "Semente", allow_empty=True, minimum=0)
        if not fields_ok:
            return None
        return n_in, layers, seed  # type: ignore[return-value]

    def _on_draft_change(self, _e) -> None:
        draft = self._draft(strict=False)
        task = self._selected_task()
        warning = output_activation_warning(task, self.out_activation.value or "")
        if warning:
            self.out_warning.show("warning", "Ativação de saída", warning)
        else:
            self.out_warning.clear()
        if draft is None:
            self.param_count.value = ""
            return
        n_in, layers, _seed = draft
        n_params = count_parameters(n_in, [l["neurons"] for l in layers])
        self.param_count.value = f"{format_number(n_params)} parâmetros (pesos + vieses)"
        state = self.ctx.state
        names_in = state.input_names() if state.n_inputs == n_in else None
        self.diagram.render(n_in, layers, input_names=names_in)

    # ----------------------------------------------------------------- ações
    async def _on_create(self, _e) -> None:
        state = self.ctx.state
        draft = self._draft(strict=True)
        if draft is None:
            self.message.show("error", "Corrija os campos destacados", "Há valores inválidos no formulário.")
            return
        n_in, layers, seed = draft
        task = self._selected_task()
        if state.has_network and (state.is_trained or state.has_split):
            ok = await self.ctx.confirm(
                "Recriar a rede?",
                "A rede atual, os pesos treinados, a divisão do dataset e o histórico serão descartados.",
                "Recriar",
            )
            if not ok:
                return
        self.create_btn.disabled = True
        self.ctx.update()
        try:
            await self.ctx.gateway.call("create_network", n_in, layers, seed=seed, task=task)
            description = await self.ctx.gateway.call("describe_network")
        except GatewayError as err:
            self.message.show("error", err.title, err.message)
            return
        finally:
            self.create_btn.disabled = False
        state.on_network_created(description or {"n_inputs": n_in, "layers": layers}, task)
        self.message.show(
            "success",
            "Rede criada",
            f"Arquitetura {state.architecture_text()} com {format_number(state.n_params)} parâmetros. "
            "Próximo passo: carregar o dataset na aba Dados.",
        )
        state.notify()

    # ------------------------------------------------------ chamadas externas
    def set_io_counts(self, n_inputs: int, n_outputs: int) -> None:
        """Usado pela aba Dados: ajusta entradas/saídas às colunas escolhidas."""
        self.n_inputs.value = str(n_inputs)
        self.out_neurons.value = str(n_outputs)
        self._on_draft_change(None)
        self.message.show("info", "Formulário ajustado às colunas do CSV",
                          f"{n_inputs} entrada(s) e {n_outputs} neurônio(s) de saída. Clique em “Criar rede”.")

    def load_from_state(self) -> None:
        """Preenche o formulário com a rede atual (ex.: depois de abrir um modelo)."""
        state = self.ctx.state
        if not state.network:
            return
        specs = state.layer_specs
        self.task.selected = [state.task]
        self.n_inputs.value = str(state.n_inputs)
        self.hidden_rows = []
        for spec in specs[:-1]:
            self._add_hidden(spec["neurons"], spec["activation"])
        self._reindex()
        self.out_neurons.value = str(specs[-1]["neurons"])
        self.out_activation.value = specs[-1]["activation"]
        self._apply_task_texts()
        self._on_draft_change(None)

    def refresh(self) -> None:
        state = self.ctx.state
        self.create_btn.disabled = state.training
        self.summary_holder.controls = self._summary()

    def _summary(self) -> list[ft.Control]:
        state = self.ctx.state
        if not state.network:
            return [body("Nenhuma rede criada ainda. Configure o formulário e clique em “Criar rede”.")]
        rows = []
        fan_in = state.n_inputs or 0
        for i, spec in enumerate(state.layer_specs):
            is_out = i == len(state.layer_specs) - 1
            params = (fan_in + 1) * spec["neurons"]
            fan_in = spec["neurons"]
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text("Saída" if is_out else f"Oculta {i + 1}")),
                ft.DataCell(ft.Text(str(spec["neurons"]))),
                ft.DataCell(ft.Text(activation_label(spec["activation"]))),
                ft.DataCell(ft.Text(ACTIVATION_RANGES.get(spec["activation"], ""))),
                ft.DataCell(ft.Text(format_number(params))),
            ]))
        origin = "Carregada de arquivo" if state.model_origin == "loaded" else "Criada nesta sessão"
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
            column_spacing=20,
            horizontal_margin=8,
        )
        return [
            body(f"{origin} · Tarefa: {TASK_LABELS.get(state.task, state.task)} · "
                 f"{state.n_inputs} entrada(s) · {format_number(state.n_params)} parâmetros", muted=False),
            ft.Row([table], scroll=ft.ScrollMode.AUTO),
        ]

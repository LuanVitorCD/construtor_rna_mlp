"""Aba 2 do modo "Criar novo modelo": a arquitetura da MLP, sem tocar no código.

Entradas e saídas podem vir automaticamente das colunas escolhidas na aba Dados
("Automático") ou ser digitadas ("Personalizado"). Ao criar a rede, os dados da aba
Dados já são carregados, divididos e normalizados: se algo não bater, o erro aparece
aqui mesmo, sem precisar voltar uma página.
"""

from __future__ import annotations

import os

import flet as ft

from frontend.analysis import count_parameters, output_activation_warning, suggest_hidden_neurons
from frontend.components.network_diagram import NetworkDiagram
from frontend.components.ui import (
    Message,
    activation_dropdown,
    architecture_table,
    body,
    learn_more,
    number_field,
    section,
)
from frontend.context import AppContext, updates_page
from frontend.parsing import ParseError, format_number, parse_int
from frontend.service_gateway import GatewayError
from frontend.texts import HELP, TASK_LABELS

C = ft.Colors
MAX_HIDDEN = 6
IO_AUTO, IO_CUSTOM = "auto", "custom"


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
        self._last_task = state.task_choice

        # --- entradas e saídas
        self.task_text = body("", muted=False)
        self.io_mode = ft.SegmentedButton(
            segments=[
                ft.Segment(value=IO_AUTO, label=ft.Text("Automático (do CSV)"), icon=ft.Icon(ft.Icons.AUTO_FIX_HIGH)),
                ft.Segment(value=IO_CUSTOM, label=ft.Text("Personalizado"), icon=ft.Icon(ft.Icons.EDIT)),
            ],
            selected=[IO_AUTO],
            allow_empty_selection=False,
            on_change=self._on_io_mode_change,
        )
        self.n_inputs = number_field("Nº de entradas", "", integer=True, width=180,
                                     on_change=self._on_draft_change, tooltip=HELP["inputs"])
        self.n_outputs = number_field("Neurônios de saída", "", integer=True, width=180,
                                      on_change=self._on_draft_change)
        self.io_message = Message()

        # --- camadas ocultas
        self.hidden_rows: list[_HiddenRow] = []
        self.hidden_column = ft.Column(spacing=12, tight=True)
        self.add_hidden_btn = ft.OutlinedButton("Adicionar camada oculta", icon=ft.Icons.ADD,
                                                on_click=self._on_add_hidden)
        self.suggest_btn = ft.TextButton("Sugerir neurônios (2n + 1)", icon=ft.Icons.AUTO_FIX_HIGH,
                                         on_click=self._on_suggest, tooltip="Regra de Kolmogorov (Ivan Nunes, cap. 5)")

        # --- camada de saída
        self.out_activation = activation_dropdown(
            "sigmoid" if state.task_choice == "classification" else "linear", state.activations,
            label="Função de ativação da saída", on_select=self._on_draft_change)
        self.out_activation.expand = True
        self.out_help = body("")
        self.out_warning = Message()

        # --- inicialização e ação
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
                section(
                    "1. Entradas e saídas",
                    [
                        self.task_text,
                        self.io_mode,
                        ft.Row([self.n_inputs, self.n_outputs], wrap=True, spacing=12,
                               vertical_alignment=ft.CrossAxisAlignment.START),
                        self.io_message.root,
                        learn_more("Automático ou personalizado?", HELP["io_auto"] + "\n\n" + HELP["io_custom"]),
                    ],
                    icon=ft.Icons.INPUT,
                ),
                section(
                    "2. Camadas ocultas",
                    [
                        self.hidden_column,
                        ft.Row([self.add_hidden_btn, self.suggest_btn], wrap=True, spacing=8),
                        learn_more("Como escolher camadas e neurônios?", HELP["hidden"]),
                    ],
                    icon=ft.Icons.ACCOUNT_TREE,
                ),
                section(
                    "3. Camada de saída",
                    [ft.Row([self.out_activation, ft.Container(width=48)], spacing=12),
                     self.out_help, self.out_warning.root],
                    icon=ft.Icons.OUTPUT,
                ),
                section(
                    "4. Inicialização",
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
            self.n_inputs.error = str(err).split(": ", 1)[-1].capitalize()
            return
        if not self.hidden_rows:
            self._add_hidden(3, "tanh")
        suggestion = suggest_hidden_neurons(n_in)  # type: ignore[arg-type]
        for r in self.hidden_rows:
            r.neurons.value = str(suggestion)
        self._on_draft_change(None)
        self.ctx.toast("info", f"Sugestão aplicada: {suggestion} neurônios por camada oculta (2 × {n_in} + 1).")

    # ----------------------------------------------------------- entradas/saídas
    def _io_mode(self) -> str:
        return (self.io_mode.selected or [IO_AUTO])[0]

    def _on_io_mode_change(self, _e) -> None:
        self._sync_io()
        self._on_draft_change(None)
        self.create_btn.disabled = self._create_blocked()
        self.summary_holder.controls = self._summary()

    def _sync_io(self) -> None:
        """Preenche (Automático) ou confere (Personalizado) entradas/saídas com o CSV."""
        state = self.ctx.state
        auto = self._io_mode() == IO_AUTO
        suggestion = state.suggested_io()
        for field in (self.n_inputs, self.n_outputs):
            field.read_only = auto
            field.filled = auto
            field.fill_color = C.SURFACE_CONTAINER_HIGH if auto else None
        if auto:
            if suggestion:
                self.n_inputs.value, self.n_outputs.value = str(suggestion[0]), str(suggestion[1])
                self.n_inputs.error = self.n_outputs.error = None
                name = os.path.basename(state.csv_path or "")
                self.io_message.show("success", "Preenchido a partir do CSV",
                                     f"{name}: entradas = {', '.join(state.input_cols)}; "
                                     f"saída = {', '.join(state.output_cols)}.")
            else:
                self.n_inputs.value = self.n_outputs.value = ""
                if state.selection_problem:  # ex.: saída de texto com a tarefa Regressão
                    self.io_message.show("error", "Os dados escolhidos têm um problema",
                                         state.selection_problem + " Corrija na aba Dados.",
                                         action=("Ir para a aba Dados", lambda _e: self.ctx.navigate(0)))
                else:
                    self.io_message.show("info", "Escolha os dados primeiro",
                                         "Na aba Dados, escolha o CSV e marque as colunas de entrada e de saída "
                                         "(e a tarefa). Ou use o modo Personalizado.",
                                         action=("Ir para a aba Dados", lambda _e: self.ctx.navigate(0)))
            return
        if not suggestion:
            self.io_message.show("info", "Sem dados escolhidos",
                                 "A rede pode ser criada assim, mas para treiná-la será preciso escolher na aba "
                                 "Dados um CSV com o mesmo nº de entradas e saídas.")
            return
        try:
            typed = (int(self.n_inputs.value or ""), int(self.n_outputs.value or ""))
        except ValueError:
            typed = None
        if typed != suggestion:
            self.io_message.show("warning", "Diferente do CSV escolhido",
                                 f"Os dados pedem {suggestion[0]} entrada(s) e {suggestion[1]} neurônio(s) de "
                                 "saída. Com outros valores, a rede é criada, mas os dados não poderão ser usados "
                                 "para treiná-la.")
        else:
            self.io_message.show("success", "Igual ao CSV escolhido",
                                 f"{suggestion[0]} entrada(s) e {suggestion[1]} neurônio(s) de saída.")

    def _create_blocked(self) -> bool:
        state = self.ctx.state
        return state.training or (self._io_mode() == IO_AUTO and state.suggested_io() is None)

    # ------------------------------------------------------------------ tarefa
    def _apply_task_texts(self) -> None:
        task = self.ctx.state.task_choice
        self.task_text.value = f"Tarefa escolhida na aba Dados: {TASK_LABELS.get(task, task)}."
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
        n_out = read_int(self.n_outputs, "Neurônios de saída")
        layers.append({"neurons": n_out, "activation": self.out_activation.value})
        seed = read_int(self.seed, "Semente", allow_empty=True, minimum=0)
        if not fields_ok:
            return None
        return n_in, layers, seed  # type: ignore[return-value]

    def _on_draft_change(self, e) -> None:
        state = self.ctx.state
        if e is not None and self._io_mode() == IO_CUSTOM:
            self._sync_io()  # recompara com o CSV a cada digitação
        warning = output_activation_warning(state.task_choice, self.out_activation.value or "")
        if warning:
            self.out_warning.show("warning", "Ativação de saída", warning)
        else:
            self.out_warning.clear()
        draft = self._draft(strict=False)
        if draft is None:
            self.param_count.value = ""
            self.diagram.clear("O diagrama aparece quando o nº de entradas e de neurônios de saída "
                               "estiver definido (e os demais campos forem números válidos).")
            return
        n_in, layers, _seed = draft
        n_params = count_parameters(n_in, [l["neurons"] for l in layers])
        self.param_count.value = f"{format_number(n_params)} parâmetros (pesos + vieses)"
        n_out = layers[-1]["neurons"]
        self.diagram.render(n_in, layers, input_names=state.input_names(n_in),
                            output_names=state.output_names(n_out, task=state.task_choice))

    # ----------------------------------------------------------------- ações
    @updates_page
    async def _on_create(self, _e) -> None:
        state = self.ctx.state
        if self._io_mode() == IO_AUTO and state.suggested_io() is None:
            self.message.show("error", "Faltam os dados",
                              "No modo Automático, escolha o CSV e as colunas na aba Dados (ou use Personalizado).")
            return
        draft = self._draft(strict=True)
        if draft is None:
            self.message.show("error", "Corrija os campos destacados", "Há valores inválidos no formulário.")
            return
        n_in, layers, seed = draft
        task = state.task_choice
        if state.has_network and (state.is_trained or state.has_split):
            ok = await self.ctx.confirm(
                "Recriar a rede?",
                "A rede atual, os pesos treinados, os dados aplicados e o histórico serão descartados.",
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
        summary = f"Arquitetura {state.architecture_text()} com {format_number(state.n_params)} parâmetros."
        if not state.data_selection_ready:
            self.message.show("success", "Rede criada",
                              summary + " Para treinar, escolha os dados na aba Dados e clique em "
                                        "“Aplicar dados à rede”.")
            state.notify()
            return
        ok, error = await self.ctx.prepare_data()
        if ok:
            self.message.show(
                "success", "Rede criada e dados aplicados",
                f"{summary} Treinamento {state.split_size('train')} · validação {state.split_size('val')} · "
                f"teste {state.split_size('test')} amostras. Próximo passo: aba Treinamento.")
        else:
            self.message.show(
                "error", "Rede criada, mas os dados não foram aplicados",
                f"{error} Use o modo Automático (ou corrija os números) e crie a rede de novo.",
                action=("Ir para a aba Dados", lambda _e: self.ctx.navigate(0)))
        state.notify()

    # ------------------------------------------------------------- modelo base
    def apply_template(self, info) -> None:
        """Preenche o formulário com a configuração de um modelo salvo (sem os pesos)."""
        state = self.ctx.state
        specs = info.layer_specs
        self.hidden_rows = []
        for spec in specs[:-1]:
            self._add_hidden(spec["neurons"], spec["activation"])
        self._reindex()
        self.out_activation.value = specs[-1]["activation"]
        self._last_task = state.task_choice
        n_in, n_out = int(info.description.get("n_inputs", 0)), specs[-1]["neurons"]
        if state.suggested_io() == (n_in, n_out):
            self.io_mode.selected = [IO_AUTO]
        else:
            self.io_mode.selected = [IO_CUSTOM]
            self.n_inputs.value, self.n_outputs.value = str(n_in), str(n_out)
        self._apply_task_texts()
        self._sync_io()
        self._on_draft_change(None)
        self.message.show("info", f"Configuração de “{os.path.basename(info.path)}” aplicada", HELP["template"])

    # ------------------------------------------------------------------ estado
    def refresh(self) -> None:
        state = self.ctx.state
        if state.task_choice != self._last_task:  # tarefa mudou na aba Dados
            self._last_task = state.task_choice
            self.out_activation.value = "sigmoid" if state.task_choice == "classification" else "linear"
        self._apply_task_texts()
        self._sync_io()
        self._on_draft_change(None)
        self.create_btn.disabled = self._create_blocked()
        self.summary_holder.controls = self._summary()

    def _summary(self) -> list[ft.Control]:
        state = self.ctx.state
        if not state.network:
            return [body("Nenhuma rede criada ainda. Configure o formulário e clique em “Criar rede”.")]
        return [
            body(f"Tarefa: {TASK_LABELS.get(state.task, state.task)} · {state.n_inputs} entrada(s) · "
                 f"{format_number(state.n_params)} parâmetros", muted=False),
            architecture_table(state.n_inputs or 0, state.layer_specs),
        ]

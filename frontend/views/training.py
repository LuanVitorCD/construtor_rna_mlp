"""Aba 3: treinamento com SGD e curva de aprendizagem ao vivo.

O backend chama ``on_epoch(dict)`` e ``on_finish()`` na thread de treino. Esses
callbacks só enfileiram dados (com lock). Uma tarefa assíncrona no laço do Flet
(``page.run_task``) esvazia a fila a cada 150 ms e redesenha a tela. Assim a UI
nunca é tocada fora da thread dela, e um treino muito rápido (milhares de épocas
por segundo) não inunda a interface.
"""

from __future__ import annotations

import asyncio
import csv
import threading
import time

import flet as ft

from frontend.analysis import analyze_history, best_epoch
from frontend.components.charts import LossChart
from frontend.components.ui import (
    Message,
    body,
    insight_box,
    learn_more,
    metric,
    number_field,
    requirement_list,
    section,
)
from frontend.context import AppContext, updates_page
from frontend.parsing import ParseError, format_number, parse_float, parse_int
from frontend.service_gateway import GatewayError
from frontend.texts import HELP, NORMALIZATION_SHORT

C = ft.Colors
PUMP_INTERVAL = 0.15
STOP_TIMEOUT = 8.0


class TrainingView:
    title = "Treinamento"

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._finished = threading.Event()
        self._stop_requested_at: float | None = None
        self._total = 0

        self.epochs = number_field("Épocas", "200", integer=True, width=150, on_submit=self._on_start,
                                   tooltip=HELP["epochs"])
        self.lr = number_field("Taxa de aprendizagem (η)", "0,01", width=230, on_submit=self._on_start,
                               helper="Típico: 0,001 a 0,1", tooltip=HELP["lr"])
        self.shuffle_seed = number_field("Semente do embaralhamento", "", integer=True, width=300,
                                         hint="aleatória", helper="Opcional", tooltip=HELP["shuffle_seed"])
        self.start_btn = ft.FilledButton("Iniciar treinamento", icon=ft.Icons.PLAY_ARROW, on_click=self._on_start,
                                         style=ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=24, vertical=18)))
        self.stop_btn = ft.OutlinedButton("Parar", icon=ft.Icons.STOP, on_click=self._on_stop, disabled=True,
                                          style=ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=20, vertical=18)))
        self.requirements = ft.Container()
        self.config_summary = body("")
        self.message = Message()

        self.progress = ft.ProgressBar(value=0, bar_height=8, border_radius=4, color=C.PRIMARY,
                                       bgcolor=C.SURFACE_CONTAINER_HIGHEST, semantics_label="Progresso do treinamento")
        self.progress_text = body("Nenhum treinamento executado.", muted=False)
        self.metrics = ft.ResponsiveRow(spacing=12, run_spacing=12)

        self.chart = LossChart()
        self.log_switch = ft.Switch(label="Escala logarítmica no eixo do MSE", value=False,
                                    on_change=self._on_log_toggle)
        self.insights = ft.Column(spacing=10, tight=True)
        self.recent = ft.Container()
        self.export_btn = ft.TextButton("Exportar histórico (CSV)", icon=ft.Icons.DOWNLOAD,
                                        on_click=self._on_export, disabled=True)

        controls_section = section(
            "Hiperparâmetros (SGD)",
            [
                ft.Row([self.epochs, self.lr, self.shuffle_seed], wrap=True, spacing=12,
                       vertical_alignment=ft.CrossAxisAlignment.START),
                self.config_summary,
                ft.Row([self.start_btn, self.stop_btn], wrap=True, spacing=12),
                self.requirements,
                self.message.root,
                learn_more("O que é época e taxa de aprendizagem?", HELP["epochs"] + "\n\n" + HELP["lr"]),
                body(HELP["retrain"]),
            ],
            icon=ft.Icons.TUNE,
        )
        progress_section = section(
            "Progresso",
            [self.progress, self.progress_text, self.metrics],
            icon=ft.Icons.INSIGHTS,
        )
        chart_section = section(
            "Curva de aprendizagem (MSE por época)",
            [self.log_switch, self.chart.root, learn_more("Como ler este gráfico?", HELP["curve"])],
            icon=ft.Icons.SHOW_CHART,
            trailing=self.export_btn,
        )
        analysis_section = section("Leitura da curva", [self.insights, self.recent], icon=ft.Icons.FACT_CHECK)

        self.root = ft.ResponsiveRow(
            [
                ft.Column([controls_section, progress_section, analysis_section], spacing=16,
                          col={"xs": 12, "xl": 5}),
                ft.Column([chart_section], spacing=16, col={"xs": 12, "xl": 7}),
            ],
            spacing=16,
            run_spacing=16,
        )
        self.refresh()

    # --------------------------------------------------------------- callbacks
    # Rodam NA THREAD DE TREINO: não tocar em controles aqui.
    def _on_epoch_thread(self, result: dict) -> None:
        with self._lock:
            self._buffer.append(dict(result))

    def _on_finish_thread(self) -> None:
        self._finished.set()

    # ------------------------------------------------------------------- ações
    @updates_page
    async def _on_start(self, _e) -> None:
        state = self.ctx.state
        if state.training or not self._can_train():
            return
        try:
            epochs = parse_int(self.epochs.value, field="Épocas", minimum=1, maximum=1_000_000)
            self.epochs.error = None
            lr = parse_float(self.lr.value, field="Taxa de aprendizagem", minimum=0, min_inclusive=False)
            self.lr.error = None
            seed = parse_int(self.shuffle_seed.value, field="Semente", minimum=0, allow_empty=True)
            self.shuffle_seed.error = None
        except ParseError as err:
            target = {"Épocas": self.epochs, "Taxa de aprendizagem": self.lr}.get(str(err).split(":")[0],
                                                                                   self.shuffle_seed)
            target.error = str(err)
            self.message.show("error", "Corrija os hiperparâmetros", str(err))
            return
        if lr and lr > 1:
            self.message.show("warning", "Taxa de aprendizagem alta",
                              f"η = {format_number(lr)} costuma fazer o erro oscilar ou divergir.")
        else:
            self.message.clear()

        with self._lock:
            self._buffer = []
        self._finished.clear()
        self._stop_requested_at = None
        self._total = int(epochs)  # type: ignore[arg-type]
        state.history = []
        state.training = True
        self.insights.controls = []
        self.recent.content = None
        self.chart.render([], self._total)
        self._render_progress()
        try:
            self.ctx.gateway.call_sync(
                "start_training", epochs, lr, shuffle_seed=seed,
                on_epoch=self._on_epoch_thread, on_finish=self._on_finish_thread,
            )
        except GatewayError as err:
            state.training = False
            self.message.show("error", err.title, err.message)
            state.notify()
            return
        state.notify()
        self.ctx.page.run_task(self._pump)

    def _on_stop(self, _e) -> None:
        if not self.ctx.state.training:
            return
        self._stop_requested_at = time.monotonic()
        self.stop_btn.disabled = True
        self.progress_text.value = "Parando ao fim da época atual…"
        try:
            self.ctx.gateway.call_sync("stop_training")
        except GatewayError as err:
            self.message.show("error", err.title, err.message)

    async def _pump(self) -> None:
        """Leva os resultados da thread de treino para a tela."""
        state = self.ctx.state
        forced = False
        while True:
            await asyncio.sleep(PUMP_INTERVAL)
            with self._lock:
                new, self._buffer = self._buffer, []
            if new:
                state.history.extend(new)
                self._render_progress()
                self.ctx.page.update()
            if self._finished.is_set():
                with self._lock:
                    late, self._buffer = self._buffer, []
                state.history.extend(late)
                break
            if self._stop_requested_at and time.monotonic() - self._stop_requested_at > STOP_TIMEOUT:
                forced = True
                break
        self._finish(forced=forced)

    def _finish(self, *, forced: bool) -> None:
        state = self.ctx.state
        stopped = self._stop_requested_at is not None
        state.training = False
        if state.history:
            state.trained_epochs += len(state.history)
            state.evaluations = {}
        self._render_progress()
        self.insights.controls = [insight_box(i) for i in analyze_history(state.history, stopped_early=stopped)]
        self._render_recent()
        done = len(state.history)
        if forced:
            self.message.show("warning", "O backend não confirmou a parada",
                              "A interface foi liberada, mas o treinamento pode continuar em segundo plano.")
        elif done:
            last = state.history[-1]
            self.message.show(
                "success",
                "Treinamento interrompido" if stopped else "Treinamento concluído",
                f"{done} época(s). MSE final: treinamento {format_number(last['train_mse'])}, "
                f"validação {format_number(last['val_mse'])}. Próximo passo: aba Avaliação.",
            )
        else:
            self.message.show("warning", "Nenhuma época concluída", "O treinamento terminou sem resultados.")
        state.notify()
        self.ctx.page.update()

    def _on_log_toggle(self, e) -> None:
        self.chart.set_log_scale(bool(e.control.value))

    @updates_page
    async def _on_export(self, _e) -> None:
        history = self.ctx.state.history
        if not history:
            return
        try:
            path = await self.ctx.file_picker.save_file(
                dialog_title="Salvar histórico de treinamento",
                file_name="historico_treinamento.csv",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["csv"],
            )
        except Exception as exc:  # noqa: BLE001
            self.ctx.toast("error", f"Não foi possível abrir o diálogo: {exc}")
            return
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_mse", "val_mse"])
                for h in history:
                    writer.writerow([h["epoch"], repr(float(h["train_mse"])), repr(float(h["val_mse"]))])
        except OSError as exc:
            self.ctx.toast("error", f"Erro ao salvar: {exc}")
            return
        self.ctx.toast("success", f"Histórico salvo em {path}")

    # ------------------------------------------------------------------ render
    def _render_progress(self) -> None:
        state = self.ctx.state
        history = state.history
        total = self._total or (history[-1]["epoch"] if history else 0)
        last = history[-1] if history else None
        best = best_epoch(history)
        self.progress.value = (last["epoch"] / total) if (last and total) else (None if state.training else 0)
        if state.training and not last:
            self.progress_text.value = "Iniciando…"
        elif last:
            prefix = "Treinando" if state.training else "Última execução"
            self.progress_text.value = f"{prefix}: época {last['epoch']} de {total}."
        self.metrics.controls = [
            metric("MSE de treinamento", format_number(last["train_mse"]) if last else "—", col={"xs": 6, "md": 4}),
            metric("MSE de validação", format_number(last["val_mse"]) if last else "—", col={"xs": 6, "md": 4}),
            metric("Menor MSE de validação",
                   format_number(best["val_mse"]) if best else "—",
                   hint=f"época {best['epoch']}" if best else None, col={"xs": 12, "md": 4}),
        ]
        self.chart.render(history, total)

    def _render_recent(self) -> None:
        history = self.ctx.state.history
        if not history:
            self.recent.content = None
            return
        rows = [
            ft.DataRow(cells=[ft.DataCell(ft.Text(str(h["epoch"]))),
                              ft.DataCell(ft.Text(format_number(h["train_mse"], 6))),
                              ft.DataCell(ft.Text(format_number(h["val_mse"], 6)))])
            for h in history[-8:][::-1]
        ]
        self.recent.content = ft.Column(
            [
                body("Últimas épocas (o histórico completo pode ser exportado em CSV):"),
                ft.DataTable(
                    columns=[ft.DataColumn(ft.Text("Época"), numeric=True),
                             ft.DataColumn(ft.Text("MSE treinamento"), numeric=True),
                             ft.DataColumn(ft.Text("MSE validação"), numeric=True)],
                    rows=rows,
                    column_spacing=24,
                    horizontal_margin=8,
                    heading_text_style=ft.TextStyle(weight=ft.FontWeight.W_600, color=C.ON_SURFACE),
                    data_row_min_height=36,
                    data_row_max_height=40,
                ),
            ],
            tight=True,
            spacing=8,
        )

    def _can_train(self) -> bool:
        s = self.ctx.state
        return s.has_network and s.data_ready

    def refresh(self) -> None:
        state = self.ctx.state
        ready = self._can_train()
        self.start_btn.disabled = state.training or not ready
        self.stop_btn.disabled = not state.training or self._stop_requested_at is not None
        for f in (self.epochs, self.lr, self.shuffle_seed):
            f.disabled = state.training
        self.export_btn.disabled = state.training or not state.history
        if ready:
            self.requirements.content = None
            self.config_summary.value = (
                f"Rede {state.architecture_text()} · treinamento com {state.split_size('train')} amostras · "
                f"validação com {state.split_size('val')} · normalização: {NORMALIZATION_SHORT[state.normalization]}"
            )
        else:
            self.config_summary.value = "Para treinar, conclua as etapas abaixo:"
            self.requirements.content = requirement_list([
                ("CSV e colunas escolhidos (aba Dados)", state.data_selection_ready),
                ("Rede criada (aba Arquitetura)", state.has_network),
                ("Dados aplicados à rede (aba Dados)"
                 + (" — há alterações não aplicadas" if state.data_dirty else ""), state.data_ready),
            ])
        if not state.training and not state.history:
            self.progress.value = 0
            self.progress_text.value = "Nenhum treinamento executado."
            self.metrics.controls = []
            self.insights.controls = []
            self.recent.content = None
            self.chart.render([], 0)

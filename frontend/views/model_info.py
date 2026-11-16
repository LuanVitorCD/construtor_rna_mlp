"""Aba 1 do modo "Usar modelo existente": informações gerais do modelo aberto."""

from __future__ import annotations

import os

import flet as ft

from frontend.components.network_diagram import NetworkDiagram
from frontend.components.ui import architecture_table, body, metric, section
from frontend.context import AppContext
from frontend.parsing import format_number
from frontend.texts import NORMALIZATION_SHORT, TASK_LABELS

C = ft.Colors


class ModelInfoView:
    title = "Informações do modelo"

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.file_name = ft.Text("", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600,
                                 color=C.ON_SURFACE, selectable=True)
        self.file_path = body("", selectable=True)
        self.open_btn = ft.OutlinedButton("Abrir outro modelo", icon=ft.Icons.FOLDER_OPEN, on_click=self._on_open)
        self.predict_btn = ft.FilledButton("Ir para Previsão", icon=ft.Icons.CALCULATE,
                                           on_click=lambda _e: self.ctx.navigate(1), autofocus=True)
        self.metrics = ft.ResponsiveRow(spacing=12, run_spacing=12)
        self.columns_holder = ft.Column(tight=True, spacing=8)
        self.diagram = NetworkDiagram()
        self.table_holder = ft.Container()

        left = ft.Column(
            [
                section("Arquivo", [self.file_name, self.file_path,
                                    ft.Row([self.predict_btn, self.open_btn], wrap=True, spacing=12)],
                        icon=ft.Icons.DESCRIPTION),
                section("Resumo", [self.metrics], icon=ft.Icons.INSIGHTS),
                section("Colunas e normalização", [self.columns_holder], icon=ft.Icons.TABLE_CHART),
            ],
            spacing=16,
            col={"xs": 12, "lg": 5},
        )
        right = ft.Column(
            [section("Arquitetura", [self.diagram.root, self.table_holder], icon=ft.Icons.HUB)],
            spacing=16,
            col={"xs": 12, "lg": 7},
        )
        self.root = ft.ResponsiveRow([left, right], spacing=16, run_spacing=16)
        self.refresh()

    async def _on_open(self, _e) -> None:
        await self.ctx.open_model()

    def refresh(self) -> None:
        state = self.ctx.state
        busy = state.training
        self.open_btn.disabled = busy
        if not state.network:
            self.file_name.value = "Nenhum modelo aberto"
            self.file_path.value = ""
            self.metrics.controls = []
            self.columns_holder.controls = []
            self.table_holder.content = None
            return
        path = state.model_path or ""
        self.file_name.value = os.path.basename(path) or "Modelo"
        self.file_path.value = path
        specs = state.layer_specs
        norm = state.loaded_normalization
        self.metrics.controls = [
            metric("Tarefa", TASK_LABELS.get(state.task, state.task), col={"xs": 12}),
            metric("Entradas", str(state.n_inputs), col={"xs": 6, "md": 4}),
            metric("Neurônios de saída", str(state.n_outputs), col={"xs": 6, "md": 4}),
            metric("Camadas ocultas", str(max(0, len(specs) - 1)), col={"xs": 6, "md": 4}),
            metric("Parâmetros", format_number(state.n_params), hint="pesos + vieses", col={"xs": 6, "md": 4}),
            metric("Normalização", NORMALIZATION_SHORT[norm] if norm else "Nenhuma",
                   col={"xs": 6, "md": 4}),
        ]
        meta = state.loaded_meta or {}
        items: list[ft.Control] = []
        ins, outs = meta.get("input_cols") or [], meta.get("output_cols") or []
        if ins:
            items.append(body(f"Entradas ({len(ins)}): " + ", ".join(map(str, ins)), muted=False))
        if outs:
            items.append(body(f"Saída: " + ", ".join(map(str, outs)), muted=False))
        if state.class_labels:
            items.append(body(f"Classes ({len(state.class_labels)}): " + ", ".join(state.class_labels), muted=False))
        if not ins and not outs:
            items.append(body("O arquivo não traz os nomes das colunas: na previsão, as entradas aparecem "
                              "como x₁, x₂, …"))
        if norm:
            items.append(body(f"As entradas digitadas na Previsão são normalizadas automaticamente "
                              f"({NORMALIZATION_SHORT[norm]}) com os parâmetros salvos no modelo."))
        else:
            items.append(body("O modelo foi treinado sem normalização: as entradas vão direto para a rede."))
        self.columns_holder.controls = items
        self.diagram.render(state.n_inputs or 0, specs, input_names=state.input_names(),
                            output_names=state.output_names())
        self.table_holder.content = architecture_table(state.n_inputs or 0, specs)

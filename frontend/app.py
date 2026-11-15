"""Janela principal: barra superior, navegação por etapas, barra de status e diálogos."""

from __future__ import annotations

import asyncio
import os

import flet as ft

from frontend import theme
from frontend.components.ui import LEVELS, heading
from frontend.context import AppContext
from frontend.service_gateway import GatewayError, create_gateway
from frontend.state import AppState
from frontend.texts import APP_SUBTITLE, APP_TITLE, HELP, NORMALIZATION_SHORT, SHORTCUTS, TASK_LABELS
from frontend.views.architecture import ArchitectureView
from frontend.views.data import DataView
from frontend.views.evaluation import EvaluationView
from frontend.views.prediction import PredictionView
from frontend.views.training import TrainingView

C = ft.Colors
PREF_THEME = "mlp_frontend.theme_mode"

STEPS = [
    # (ícone, ícone selecionado, rótulo, descrição)
    (ft.Icons.HUB_OUTLINED, ft.Icons.HUB, "Arquitetura",
     "Defina entradas, camadas ocultas, neurônios, funções de ativação e a semente."),
    (ft.Icons.TABLE_CHART_OUTLINED, ft.Icons.TABLE_CHART, "Dados",
     "Carregue o CSV, escolha entradas e saídas, divida o dataset e (opcionalmente) normalize."),
    (ft.Icons.SHOW_CHART, ft.Icons.SHOW_CHART, "Treinamento",
     "Treine com SGD e acompanhe o MSE de treinamento e de validação a cada época."),
    (ft.Icons.ASSESSMENT_OUTLINED, ft.Icons.ASSESSMENT, "Avaliação",
     "Meça o erro no conjunto de teste (avaliação final) e compare os subconjuntos."),
    (ft.Icons.CALCULATE_OUTLINED, ft.Icons.CALCULATE, "Previsão",
     "Use a rede treinada (ou um modelo salvo) em novos valores de entrada."),
]


class AppShell:
    def __init__(self, page: ft.Page, ctx: AppContext) -> None:
        self.page = page
        self.ctx = ctx
        self.dark = False
        self.prefs = ft.SharedPreferences()  # lembra o tema escolhido entre execuções
        ctx.toast = self.toast
        ctx.navigate = self.navigate
        ctx.confirm = self.confirm
        ctx.adjust_architecture = self._adjust_architecture

        self.views = [
            ArchitectureView(ctx),
            DataView(ctx),
            TrainingView(ctx),
            EvaluationView(ctx),
            PredictionView(ctx),
        ]
        self.pages = [self._wrap(view, step) for view, step in zip(self.views, STEPS)]

        self.rail = ft.NavigationRail(
            selected_index=0,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=104,
            group_alignment=-0.95,
            bgcolor=C.SURFACE_CONTAINER_LOW,
            indicator_color=C.PRIMARY_CONTAINER,
            destinations=[
                ft.NavigationRailDestination(icon=icon, selected_icon=sel, label=f"{i + 1}. {label}")
                for i, (icon, sel, label, _d) in enumerate(STEPS)
            ],
            on_change=self._on_rail_change,
        )
        self.content = ft.Container(content=self.pages[0], expand=True,
                                    padding=ft.Padding.only(left=28, right=20, top=16))
        self.status_text = ft.Text("", size=13, color=C.ON_SURFACE_VARIANT, selectable=True)

        self.open_btn = ft.TextButton("Abrir modelo", icon=ft.Icons.FOLDER_OPEN, on_click=self._on_open_model,
                                      tooltip="Abrir uma rede salva em JSON (Ctrl+O)")
        self.save_btn = ft.TextButton("Salvar modelo", icon=ft.Icons.SAVE, on_click=self._on_save_model,
                                      tooltip="Salvar a rede atual em JSON (Ctrl+S)")
        self.theme_btn = ft.IconButton(ft.Icons.DARK_MODE, on_click=self._on_toggle_theme,
                                       tooltip="Ativar tema escuro (Ctrl+T)")
        self.keys_btn = ft.IconButton(ft.Icons.KEYBOARD, on_click=self._on_show_shortcuts,
                                      tooltip="Atalhos de teclado")
        self.title_text = ft.Text(APP_TITLE, theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_700)
        # Subtítulo longo: 1 linha com reticências em janelas estreitas; texto completo no tooltip.
        self.subtitle_text = ft.Text(APP_SUBTITLE, theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=1,
                                     overflow=ft.TextOverflow.ELLIPSIS, tooltip=APP_SUBTITLE)
        self.appbar = ft.AppBar(
            leading=ft.Icon(ft.Icons.HUB, semantics_label="Logotipo"),
            leading_width=56,
            title=ft.Column([self.title_text, self.subtitle_text], spacing=0, tight=True),
            toolbar_height=68,
            actions=[self.open_btn, self.save_btn, self.theme_btn, self.keys_btn, ft.Container(width=8)],
        )

        banner = ft.Container()
        if ctx.gateway.demo:
            icon, bg, fg, spoken = LEVELS["warning"]
            banner = ft.Semantics(
                live_region=True,
                content=ft.Container(
                    content=ft.Row([
                        ft.Icon(icon, color=fg, semantics_label=spoken),
                        ft.Text("Modo demonstração: a pasta backend/ não foi encontrada. Curvas, métricas e "
                                "previsões são SIMULADAS só para testar a interface.", color=fg, expand=True),
                    ], spacing=12),
                    bgcolor=bg,
                    padding=ft.Padding.symmetric(horizontal=20, vertical=10),
                ),
            )

        status_bar = ft.Container(
            content=ft.Row([ft.Icon(ft.Icons.INFO_OUTLINE, size=16, color=C.ON_SURFACE_VARIANT), self.status_text],
                           spacing=8),
            bgcolor=C.SURFACE_CONTAINER,
            border=ft.Border.only(top=ft.BorderSide(1, C.OUTLINE_VARIANT)),
            padding=ft.Padding.symmetric(horizontal=16, vertical=8),
        )
        self.root = ft.Column(
            [
                banner,
                ft.Row([self.rail, ft.VerticalDivider(width=1, color=C.OUTLINE_VARIANT), self.content],
                       expand=True, spacing=0, vertical_alignment=ft.CrossAxisAlignment.STRETCH),
                status_bar,
            ],
            expand=True,
            spacing=0,
        )
        ctx.state.subscribe(self.refresh)
        self._apply_theme_colors()
        self.refresh()

    # ----------------------------------------------------------------- layout
    def _wrap(self, view, step) -> ft.Control:
        _icon, sel_icon, label, desc = step
        return ft.Column(
            [
                ft.Row([ft.Icon(sel_icon, color=C.PRIMARY, size=28),
                        heading(label, style=ft.TextThemeStyle.HEADLINE_SMALL)], spacing=12),
                ft.Text(desc, color=C.ON_SURFACE_VARIANT, theme_style=ft.TextThemeStyle.BODY_LARGE),
                ft.Container(height=4),
                view.root,
                ft.Container(height=24),
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=8,
        )

    def navigate(self, index: int) -> None:
        index = max(0, min(index, len(self.pages) - 1))
        self.rail.selected_index = index
        self.content.content = self.pages[index]
        self.page.update()

    def _on_rail_change(self, e) -> None:
        self.navigate(int(e.control.selected_index))

    def _adjust_architecture(self, n_inputs: int, n_outputs: int) -> None:
        self.views[0].set_io_counts(n_inputs, n_outputs)
        self.navigate(0)

    # ----------------------------------------------------------------- estado
    def refresh(self) -> None:
        state = self.ctx.state
        for view in self.views:
            view.refresh()
        done = [state.has_network, state.has_split, state.is_trained and state.trained_epochs > 0,
                bool(state.evaluations), False]
        for dest, (icon, sel, _l, _d), ok in zip(self.rail.destinations, STEPS, done):
            dest.icon = ft.Icons.TASK_ALT if ok else icon
            dest.selected_icon = ft.Icons.TASK_ALT if ok else sel
        self.rail.disabled = False
        self.open_btn.disabled = state.training
        self.save_btn.disabled = state.training or not state.has_network
        self.status_text.value = self._status_line()

    def _status_line(self) -> str:
        s = self.ctx.state
        parts = []
        if s.network:
            parts.append(f"Rede {s.architecture_text()} ({TASK_LABELS.get(s.task, s.task)})")
        else:
            parts.append("Sem rede")
        if s.dataset:
            name = os.path.basename(s.csv_path) if s.csv_path else "dataset"
            parts.append(f"Dados: {name} ({s.dataset.get('rows', '?')} linhas)")
        if s.split:
            parts.append("Divisão {}/{}/{}".format(s.split_size("train"), s.split_size("val"), s.split_size("test")))
            parts.append(f"Normalização: {NORMALIZATION_SHORT[s.normalization]}")
        if s.training:
            parts.append(f"Treinando… ({len(s.history)} épocas)")
        elif s.trained_epochs:
            parts.append(f"Treinada: {s.trained_epochs} épocas")
        if s.model_origin == "loaded" and s.model_path:
            parts.append(f"Modelo: {os.path.basename(s.model_path)}")
        if self.ctx.gateway.demo:
            parts.append("MODO DEMONSTRAÇÃO")
        return "  ·  ".join(parts)

    # ------------------------------------------------------------------- tema
    def _apply_theme_colors(self) -> None:
        tokens = theme.DARK if self.dark else theme.LIGHT
        bg, fg = (tokens["surface_container_high"], tokens["on_surface"]) if self.dark else \
            (tokens["primary"], tokens["on_primary"])
        self.appbar.bgcolor = bg
        self.appbar.color = fg
        self.title_text.color = fg
        self.subtitle_text.color = fg
        self.appbar.leading.color = fg if not self.dark else tokens["primary"]
        for btn in (self.open_btn, self.save_btn):
            btn.style = ft.ButtonStyle(color={ft.ControlState.DISABLED: ft.Colors.with_opacity(0.6, fg),
                                              ft.ControlState.DEFAULT: fg})
            btn.icon_color = fg
        for btn in (self.theme_btn, self.keys_btn):
            btn.icon_color = fg
        self.theme_btn.icon = ft.Icons.LIGHT_MODE if self.dark else ft.Icons.DARK_MODE
        self.theme_btn.tooltip = ("Ativar tema claro" if self.dark else "Ativar tema escuro") + " (Ctrl+T)"
        self.page.theme_mode = ft.ThemeMode.DARK if self.dark else ft.ThemeMode.LIGHT
        self.page.bgcolor = tokens["surface"]

    async def set_dark(self, dark: bool, *, persist: bool = True) -> None:
        self.dark = dark
        self._apply_theme_colors()
        self.page.update()
        if persist:
            try:
                await self.prefs.set(PREF_THEME, "dark" if dark else "light")
            except Exception:  # noqa: BLE001 - preferência é opcional
                pass

    async def _on_toggle_theme(self, _e=None) -> None:
        await self.set_dark(not self.dark)

    # --------------------------------------------------------------- diálogos
    def toast(self, level: str, text: str) -> None:
        icon, bg, fg, spoken = LEVELS.get(level, LEVELS["info"])
        self.page.show_dialog(ft.SnackBar(
            content=ft.Row([ft.Icon(icon, color=fg, semantics_label=spoken),
                            ft.Text(text, color=fg, expand=True, selectable=True)], spacing=12),
            bgcolor=bg,
            behavior=ft.SnackBarBehavior.FLOATING,
            show_close_icon=True,
            close_icon_color=fg,
            duration=ft.Duration(seconds=6),
        ))

    async def confirm(self, title: str, text: str, confirm_label: str) -> bool:
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        def finish(result: bool) -> None:
            if not future.done():
                future.set_result(result)
            self.page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title),
            content=ft.Text(text),
            actions=[
                ft.TextButton("Cancelar", on_click=lambda _e: finish(False), autofocus=True),
                ft.FilledButton(confirm_label, on_click=lambda _e: finish(True)),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            on_dismiss=lambda _e: future.done() or future.set_result(False),
        )
        self.page.show_dialog(dialog)
        return await future

    def _on_show_shortcuts(self, _e=None) -> None:
        rows = [ft.DataRow(cells=[ft.DataCell(ft.Text(k, weight=ft.FontWeight.W_600)), ft.DataCell(ft.Text(d))])
                for k, d in SHORTCUTS]
        dialog = ft.AlertDialog(
            title=ft.Text("Atalhos de teclado"),
            content=ft.DataTable(columns=[ft.DataColumn(ft.Text("Tecla")), ft.DataColumn(ft.Text("Ação"))],
                                 rows=rows, column_spacing=24),
            actions=[ft.FilledButton("Fechar", on_click=lambda _e: self.page.pop_dialog(), autofocus=True)],
        )
        self.page.show_dialog(dialog)

    # ----------------------------------------------------------- persistência
    async def _on_open_model(self, _e=None) -> None:
        state = self.ctx.state
        if state.training:
            return
        try:
            files = await self.ctx.file_picker.pick_files(
                dialog_title="Abrir modelo (JSON)", file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["json"])
        except Exception as exc:  # noqa: BLE001
            self.toast("error", f"Não foi possível abrir o seletor: {exc}")
            return
        if not files or not files[0].path:
            return
        path = files[0].path
        if state.has_network and state.is_trained and state.model_origin != "loaded":
            if not await self.confirm("Abrir outro modelo?",
                                      "A rede atual será substituída. Salve-a antes se quiser mantê-la.",
                                      "Abrir"):
                return
        try:
            await self.ctx.gateway.call("load", path)
            description = await self.ctx.gateway.call("describe_network")
        except GatewayError as err:
            self.toast("error", f"{err.title}: {err.message}")
            return
        description = dict(description or {})
        meta = description.get("meta") or {}
        state.loaded_meta = meta
        state.on_model_loaded(description, path)
        self.views[0].load_from_state()
        state.notify()
        self.toast("success", f"Modelo carregado: {os.path.basename(path)} ({state.architecture_text()}). "
                              "Já pode prever; para avaliar, carregue e divida um dataset.")
        self.navigate(4)

    async def _on_save_model(self, _e=None) -> None:
        state = self.ctx.state
        if state.training or not state.has_network:
            return
        try:
            path = await self.ctx.file_picker.save_file(
                dialog_title="Salvar modelo (JSON)", file_name="modelo_mlp.json",
                file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=["json"])
        except Exception as exc:  # noqa: BLE001
            self.toast("error", f"Não foi possível abrir o diálogo: {exc}")
            return
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            await self.ctx.gateway.call("save", path)
        except GatewayError as err:
            self.toast("error", f"{err.title}: {err.message}")
            return
        state.model_path = path
        if not state.is_trained:
            self.toast("warning", f"Modelo salvo em {path}, mas a rede ainda não foi treinada.")
        else:
            self.toast("success", f"Modelo salvo em {path}. {HELP['persistence']}")
        state.notify()

    # ----------------------------------------------------------------- teclado
    async def on_keyboard(self, e: ft.KeyboardEvent) -> None:
        if not e.ctrl:
            return
        key = (e.key or "").upper()
        if key in {"1", "2", "3", "4", "5"}:
            self.navigate(int(key) - 1)
        elif key == "S":
            await self._on_save_model()
        elif key == "O":
            await self._on_open_model()
        elif key == "T":
            await self._on_toggle_theme()


async def main(page: ft.Page) -> None:
    page.title = f"{APP_TITLE} | Trabalho 02 (RNA)"
    page.theme = theme.light_theme()
    page.dark_theme = theme.dark_theme()
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    page.spacing = 0
    page.window.width = 1360
    page.window.height = 800
    page.window.min_width = 960
    page.window.min_height = 640

    gateway = create_gateway()
    state = AppState()
    try:
        state.activations = list(gateway.call_sync("list_activations")) or state.activations
    except GatewayError:
        pass
    ctx = AppContext(page=page, gateway=gateway, state=state, file_picker=ft.FilePicker())
    shell = AppShell(page, ctx)
    page.appbar = shell.appbar
    page.on_keyboard_event = shell.on_keyboard

    async def on_window_event(e: ft.WindowEvent) -> None:
        if e.type != ft.WindowEventType.CLOSE:
            return
        try:
            if state.training:  # não deixa a thread de treino órfã
                gateway.call_sync("stop_training")
        except Exception:  # noqa: BLE001 - fechar a janela tem prioridade
            pass
        finally:
            await page.window.destroy()

    page.window.prevent_close = True
    page.window.on_event = on_window_event

    page.add(shell.root)
    try:
        saved = await shell.prefs.get(PREF_THEME)
    except Exception:  # noqa: BLE001
        saved = None
    if saved == "dark":
        await shell.set_dark(True, persist=False)
    page.update()

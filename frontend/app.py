"""Janela principal: menu de modos, barra superior, navegação por etapas, status e diálogos.

Dois modos, escolhidos no menu inicial:
- "Criar novo modelo": Dados → Arquitetura → Treinamento → Avaliação → Previsão,
  com "Salvar modelo" e "Usar modelo como base".
- "Usar modelo existente": abre um JSON e mostra Informações do modelo → Previsão.
Os dois têm o botão "Voltar ao menu". Cada modo começa com um MLPService novo.
"""

from __future__ import annotations

import asyncio
import os
from typing import Callable

import flet as ft

from frontend import theme
from frontend.components.ui import LEVELS, heading, level_icon_color
from frontend.context import AppContext
from frontend.model_files import apply_loaded_model, read_model
from frontend.service_gateway import GatewayError, create_gateway
from frontend.state import AppState
from frontend.texts import (
    APP_SUBTITLE,
    APP_TITLE,
    HELP,
    MODE_LABELS,
    NORMALIZATION_SHORT,
    SHORTCUTS,
    TASK_LABELS,
)
from frontend.views.architecture import ArchitectureView
from frontend.views.data import DataView
from frontend.views.evaluation import EvaluationView
from frontend.views.mode_menu import ModeMenu
from frontend.views.model_info import ModelInfoView
from frontend.views.prediction import PredictionView
from frontend.views.training import TrainingView

C = ft.Colors
PREF_THEME = "mlp_frontend.theme_mode"

# (ícone, ícone selecionado, rótulo, descrição)
CREATE_STEPS = [
    (ft.Icons.TABLE_CHART_OUTLINED, ft.Icons.TABLE_CHART, "Dados",
     "Escolha a tarefa, o CSV e as colunas de entrada e de saída; defina a divisão e a normalização."),
    (ft.Icons.HUB_OUTLINED, ft.Icons.HUB, "Arquitetura",
     "Entradas e saídas (automático pelo CSV ou personalizado), camadas ocultas, ativações e semente."),
    (ft.Icons.SHOW_CHART, ft.Icons.SHOW_CHART, "Treinamento",
     "Treine com SGD e acompanhe o MSE de treinamento e de validação a cada época."),
    (ft.Icons.ASSESSMENT_OUTLINED, ft.Icons.ASSESSMENT, "Avaliação",
     "Meça o erro no conjunto de teste (avaliação final) e compare os subconjuntos."),
    (ft.Icons.CALCULATE_OUTLINED, ft.Icons.CALCULATE, "Previsão",
     "Use a rede em novos valores de entrada."),
]
USE_STEPS = [
    (ft.Icons.INFO_OUTLINE, ft.Icons.INFO, "Informações",
     "Arquitetura, colunas e normalização do modelo aberto."),
    (ft.Icons.CALCULATE_OUTLINED, ft.Icons.CALCULATE, "Previsão",
     "Use o modelo salvo em novos valores de entrada."),
]


def _wrap(view, step) -> ft.Control:
    _icon, sel_icon, label, desc = step
    return ft.Column(
        [
            ft.Row([ft.Icon(sel_icon, color=C.PRIMARY, size=28),
                    heading(label if label != "Informações" else "Informações do modelo",
                            style=ft.TextThemeStyle.HEADLINE_SMALL)], spacing=12),
            ft.Text(desc, color=C.ON_SURFACE_VARIANT, theme_style=ft.TextThemeStyle.BODY_LARGE),
            ft.Container(height=4),
            view.root,
            ft.Container(height=24),
        ],
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        spacing=8,
    )


class Workspace:
    """Menu lateral + conteúdo de um modo."""

    def __init__(self, views: list, steps: list, done: Callable[[AppState], list[bool]], page: ft.Page) -> None:
        self.views = views
        self.steps = steps
        self.done = done
        self.page = page
        self.pages = [_wrap(view, step) for view, step in zip(views, steps)]
        self.rail = ft.NavigationRail(
            selected_index=0,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=104,
            group_alignment=-0.95,
            bgcolor=C.SURFACE_CONTAINER_LOW,
            indicator_color=C.PRIMARY_CONTAINER,
            destinations=[
                ft.NavigationRailDestination(icon=icon, selected_icon=sel, label=f"{i + 1}. {label}")
                for i, (icon, sel, label, _d) in enumerate(steps)
            ],
            on_change=lambda e: self.navigate(int(e.control.selected_index)),
        )
        self.content = ft.Container(content=self.pages[0], expand=True,
                                    padding=ft.Padding.only(left=28, right=20, top=16))
        self.root = ft.Row([self.rail, ft.VerticalDivider(width=1, color=C.OUTLINE_VARIANT), self.content],
                           expand=True, spacing=0, vertical_alignment=ft.CrossAxisAlignment.STRETCH)

    def navigate(self, index: int) -> None:
        index = max(0, min(index, len(self.pages) - 1))
        self.rail.selected_index = index
        self.content.content = self.pages[index]
        self.page.update()

    def refresh(self, state: AppState) -> None:
        for view in self.views:
            view.refresh()
        for dest, (icon, sel, _l, _d), ok in zip(self.rail.destinations, self.steps, self.done(state)):
            # Etapa concluída: ✓ verde (o rótulo da etapa continua em texto ao lado).
            check = ft.Icon(ft.Icons.TASK_ALT, color=theme.SUCCESS, semantics_label="concluída")
            dest.icon = check if ok else icon
            dest.selected_icon = ft.Icon(ft.Icons.TASK_ALT, color=theme.SUCCESS,
                                         semantics_label="concluída") if ok else sel


class AppShell:
    def __init__(self, page: ft.Page, ctx: AppContext, activations: list[str]) -> None:
        self.page = page
        self.ctx = ctx
        self.activations = activations
        self.dark = False
        self.prefs = ft.SharedPreferences()  # lembra o tema escolhido entre execuções
        self.workspace: Workspace | None = None
        ctx.toast = self.toast
        ctx.navigate = self.navigate
        ctx.confirm = self.confirm
        ctx.open_model = self._on_open_other_model

        # --- barra superior
        self.menu_btn = ft.TextButton("Voltar ao menu", icon=ft.Icons.HOME, on_click=self._on_back_to_menu,
                                      tooltip="Voltar ao menu de escolha de modo (Ctrl+M)")
        self.template_btn = ft.TextButton("Usar modelo como base", icon=ft.Icons.FILE_OPEN,
                                          on_click=self._on_use_template,
                                          tooltip="Preencher a arquitetura com a configuração de um modelo "
                                                  "salvo (Ctrl+O)")
        self.save_btn = ft.TextButton("Salvar modelo", icon=ft.Icons.SAVE, on_click=self._on_save_model,
                                      tooltip="Salvar a rede atual em JSON (Ctrl+S)")
        self.open_other_btn = ft.TextButton("Abrir outro modelo", icon=ft.Icons.FOLDER_OPEN,
                                            on_click=self._on_open_other_model, tooltip="Ctrl+O")
        self.theme_btn = ft.IconButton(ft.Icons.DARK_MODE, on_click=self._on_toggle_theme,
                                       tooltip="Ativar tema escuro (Ctrl+T)")
        self.keys_btn = ft.IconButton(ft.Icons.KEYBOARD, on_click=self._on_show_shortcuts,
                                      tooltip="Atalhos de teclado")
        self.text_buttons = [self.menu_btn, self.template_btn, self.save_btn, self.open_other_btn]
        self.title_text = ft.Text(APP_TITLE, theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_700)
        # Subtítulo longo: 1 linha com reticências em janelas estreitas; texto completo no tooltip.
        self.subtitle_text = ft.Text(APP_SUBTITLE, theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=1,
                                     overflow=ft.TextOverflow.ELLIPSIS, tooltip=APP_SUBTITLE)
        self.appbar = ft.AppBar(
            leading=ft.Icon(ft.Icons.HUB, semantics_label="Logotipo"),
            leading_width=56,
            title=ft.Column([self.title_text, self.subtitle_text], spacing=0, tight=True),
            toolbar_height=68,
            actions=[],
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

        self.status_text = ft.Text("", size=13, color=C.ON_SURFACE_VARIANT, selectable=True)
        status_bar = ft.Container(
            content=ft.Row([ft.Icon(ft.Icons.INFO_OUTLINE, size=16, color=C.ON_SURFACE_VARIANT), self.status_text],
                           spacing=8),
            bgcolor=C.SURFACE_CONTAINER,
            border=ft.Border.only(top=ft.BorderSide(1, C.OUTLINE_VARIANT)),
            padding=ft.Padding.symmetric(horizontal=16, vertical=8),
        )
        self.body = ft.Container(expand=True)
        self.root = ft.Column([banner, self.body, status_bar], expand=True, spacing=0)
        self._apply_theme_colors()
        self.show_menu(update=False)

    # ------------------------------------------------------------------ modos
    def _new_state(self, mode: str | None) -> AppState:
        state = AppState(mode=mode, activations=list(self.activations))
        state.subscribe(self.refresh)
        self.ctx.state = state
        return state

    def _set_actions(self, buttons: list[ft.Control]) -> None:
        self.appbar.actions = [*buttons, self.theme_btn, self.keys_btn, ft.Container(width=8)]

    def show_menu(self, *, update: bool = True) -> None:
        self._new_state(None)
        self.workspace = None
        self.body.content = ModeMenu(self.start_create, self.start_use).root
        self._set_actions([])
        self.refresh()
        if update:
            self.page.update()

    async def start_create(self, _e=None) -> None:
        self.ctx.gateway.reset()
        self._new_state("create")
        views = [DataView(self.ctx), ArchitectureView(self.ctx), TrainingView(self.ctx),
                 EvaluationView(self.ctx), PredictionView(self.ctx)]
        self.ctx.prepare_data = views[0].prepare
        self.workspace = Workspace(
            views, CREATE_STEPS, page=self.page,
            done=lambda s: [s.data_ready, s.has_network, s.trained_epochs > 0, bool(s.evaluations), False])
        self.body.content = self.workspace.root
        self._set_actions([self.template_btn, self.save_btn, self.menu_btn])
        self.refresh()
        self.page.update()

    async def start_use(self, _e=None, path: str | None = None) -> None:
        path = path or await self._pick_json("Abrir modelo salvo (JSON)")
        if not path:
            return
        self.ctx.gateway.reset()
        try:
            info = await read_model(self.ctx.gateway, path)
        except GatewayError as err:
            self.toast("error", f"{err.title}: {err.message}")
            return
        state = self._new_state("use")
        apply_loaded_model(state, info)
        views = [ModelInfoView(self.ctx), PredictionView(self.ctx)]
        self.workspace = Workspace(views, USE_STEPS, page=self.page, done=lambda s: [s.has_network, False])
        self.body.content = self.workspace.root
        self._set_actions([self.open_other_btn, self.menu_btn])
        self.refresh()
        self.page.update()
        self.toast("success", f"Modelo aberto: {os.path.basename(path)} ({state.architecture_text()}).")

    async def _on_back_to_menu(self, _e=None) -> None:
        state = self.ctx.state
        if self.workspace is None:
            return
        if state.training:
            self.toast("warning", "Pare o treinamento antes de voltar ao menu.")
            return
        if state.mode == "create" and (state.has_network or state.csv_path):
            ok = await self.confirm(
                "Voltar ao menu?",
                "A rede, os dados escolhidos e os resultados desta sessão serão descartados. "
                "Salve o modelo antes, se quiser mantê-lo.",
                "Voltar ao menu",
            )
            if not ok:
                return
        self.ctx.gateway.reset()
        self.show_menu()

    def navigate(self, index: int) -> None:
        if self.workspace:
            self.workspace.navigate(index)

    # ----------------------------------------------------------------- estado
    def refresh(self) -> None:
        state = self.ctx.state
        if self.workspace:
            self.workspace.refresh(state)
        busy = state.training
        self.menu_btn.disabled = busy
        self.template_btn.disabled = busy
        self.open_other_btn.disabled = busy
        self.save_btn.disabled = busy or not state.has_network
        self.status_text.value = self._status_line()

    def _status_line(self) -> str:
        s = self.ctx.state
        parts: list[str] = []
        if s.mode is None:
            parts.append("Escolha um modo para começar")
        elif s.mode == "use":
            parts.append(MODE_LABELS["use"])
            if s.model_path:
                parts.append(f"Modelo: {os.path.basename(s.model_path)}")
            if s.network:
                parts.append(f"Rede {s.architecture_text()} ({TASK_LABELS.get(s.task, s.task)})")
                parts.append(f"Normalização: {NORMALIZATION_SHORT[s.loaded_normalization]}")
        else:
            parts.append(MODE_LABELS["create"])
            if s.csv_path:
                parts.append(f"Dados: {os.path.basename(s.csv_path)}")
            parts.append(f"Rede {s.architecture_text()} ({TASK_LABELS.get(s.task, s.task)})" if s.network
                         else "Sem rede")
            if s.split:
                parts.append("Divisão {}/{}/{}".format(s.split_size("train"), s.split_size("val"),
                                                     s.split_size("test")))
                parts.append(f"Normalização: {NORMALIZATION_SHORT[s.normalization]}")
            if s.data_dirty:
                parts.append("alterações na aba Dados ainda não aplicadas")
            if s.training:
                parts.append(f"Treinando… ({len(s.history)} épocas)")
            elif s.trained_epochs:
                parts.append(f"Treinada: {s.trained_epochs} épocas")
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
        for btn in self.text_buttons:
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
        # Uma notificação aberta manteria o fundo do tema anterior (texto ilegível): fecha antes.
        snack = getattr(self, "_snack", None)
        if snack is not None and snack.open:
            snack.open = False
            snack.update()
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
        self._snack = ft.SnackBar(
            content=ft.Row([ft.Icon(icon, color=level_icon_color(level), semantics_label=spoken),
                            ft.Text(text, color=fg, expand=True, selectable=True)], spacing=12),
            bgcolor=bg,
            behavior=ft.SnackBarBehavior.FLOATING,
            show_close_icon=True,
            close_icon_color=fg,
            duration=ft.Duration(seconds=6),
        )
        self.page.show_dialog(self._snack)

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

    # ----------------------------------------------------------- arquivos
    async def _pick_json(self, title: str) -> str | None:
        try:
            files = await self.ctx.file_picker.pick_files(
                dialog_title=title, file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=["json"])
        except Exception as exc:  # noqa: BLE001
            self.toast("error", f"Não foi possível abrir o seletor: {exc}")
            return None
        if not files or not files[0].path:
            return None
        return files[0].path

    async def _on_open_other_model(self, _e=None) -> None:
        """Modo "Usar modelo existente": troca o modelo aberto."""
        state = self.ctx.state
        if state.mode != "use" or state.training:
            return
        path = await self._pick_json("Abrir outro modelo (JSON)")
        if not path:
            return
        try:
            info = await read_model(self.ctx.gateway, path)
        except GatewayError as err:
            self.toast("error", f"{err.title}: {err.message}")
            return
        apply_loaded_model(state, info)
        state.notify()
        self.navigate(0)
        self.toast("success", f"Modelo aberto: {os.path.basename(path)} ({state.architecture_text()}).")

    async def _on_use_template(self, _e=None) -> None:
        """Modo "Criar novo modelo": preenche o formulário com a configuração de um modelo salvo."""
        state = self.ctx.state
        if state.mode != "create" or state.training or not self.workspace:
            return
        path = await self._pick_json("Usar modelo como base (JSON)")
        if not path:
            return
        if state.has_network and not await self.confirm(
                "Usar outro modelo como base?",
                "A rede atual desta sessão será descartada (os dados escolhidos continuam). "
                "Salve-a antes, se quiser mantê-la.", "Continuar"):
            return
        try:
            info = await read_model(self.ctx.gateway, path)
        except GatewayError as err:
            self.toast("error", f"{err.title}: {err.message}")
            return
        state.forget_network()  # só a configuração é aproveitada; a rede nova nasce em "Criar rede"
        data_view, arch_view = self.workspace.views[0], self.workspace.views[1]
        data_view.apply_template(info)
        arch_view.apply_template(info)
        state.notify()
        self.navigate(1)
        self.toast("info", f"Configuração de {os.path.basename(path)} aplicada. Confira e clique em “Criar rede”.")

    async def _on_save_model(self, _e=None) -> None:
        state = self.ctx.state
        if state.mode != "create" or state.training or not state.has_network:
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
        mode = self.ctx.state.mode
        if key.isdigit() and self.workspace:
            self.navigate(int(key) - 1)
        elif key == "N" and mode is None:
            await self.start_create()
        elif key == "O":
            if mode is None:
                await self.start_use()
            elif mode == "create":
                await self._on_use_template()
            else:
                await self._on_open_other_model()
        elif key == "S":
            await self._on_save_model()
        elif key == "M":
            await self._on_back_to_menu()
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
    activations = ["linear", "sigmoid", "tanh", "relu"]
    try:
        activations = list(gateway.call_sync("list_activations")) or activations
    except GatewayError:
        pass
    ctx = AppContext(page=page, gateway=gateway, state=AppState(), file_picker=ft.FilePicker())
    shell = AppShell(page, ctx, activations)
    page.appbar = shell.appbar
    page.on_keyboard_event = shell.on_keyboard

    async def on_window_event(e: ft.WindowEvent) -> None:
        if e.type != ft.WindowEventType.CLOSE:
            return
        try:
            if ctx.state.training:  # não deixa a thread de treino órfã
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

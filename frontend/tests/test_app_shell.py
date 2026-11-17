"""Janela principal (AppShell) sem abrir janela: menu, modos e "Reiniciar tudo"."""

import asyncio

import pytest

from frontend.app import AppShell
from frontend.context import AppContext
from frontend.demo_service import DemoBackendError, DemoService
from frontend.service_gateway import ServiceGateway
from frontend.state import AppState
from frontend.tests.ui_harness import FakePage


@pytest.fixture
def csv_regressao(tmp_path):
    path = tmp_path / "dados.csv"
    lines = ["x1,x2,x3,y"] + [f"{i},{i % 7},{i * 0.5},{2 * i + 1}" for i in range(60)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def make_shell(confirm_answer: bool = True):
    page = FakePage()
    ctx = AppContext(page=page, gateway=ServiceGateway(DemoService(), DemoBackendError, demo=True),
                     state=AppState(), file_picker=None)
    shell = AppShell(page, ctx, ["linear", "sigmoid", "tanh", "relu"])
    asked = []

    async def confirm(title, text, label):
        asked.append(title)
        return confirm_answer

    shell.confirm = confirm
    ctx.confirm = confirm
    toasts = []
    shell.toast = lambda level, text: toasts.append((level, text))
    return shell, asked, toasts


def test_reiniciar_tudo_apaga_rede_dados_e_volta_para_dados(csv_regressao):
    async def flow():
        shell, asked, toasts = make_shell()
        await shell.start_create()
        assert shell.reset_btn in shell.appbar.actions  # botão no topo, no modo Criar
        data, arch = shell.workspace.views[0], shell.workspace.views[1]
        old_service = shell.ctx.gateway.service
        await data.open_csv(csv_regressao)
        await arch._on_create(None)
        shell.navigate(1)
        assert shell.ctx.state.has_network and shell.ctx.state.csv_path

        await shell._on_reset_all()
        state = shell.ctx.state
        assert asked == ["Reiniciar tudo?"]  # pediu confirmação
        assert not state.has_network and state.csv_path is None and state.mode == "create"
        assert shell.ctx.gateway.service is not old_service  # backend também recomeça
        assert shell.workspace.views[0] is not data and shell.workspace.current == 0  # telas novas, aba Dados
        assert toasts[-1][0] == "info"

    asyncio.run(flow())


def test_reiniciar_tudo_cancelado_nao_muda_nada(csv_regressao):
    async def flow():
        shell, asked, _toasts = make_shell(confirm_answer=False)
        await shell.start_create()
        data = shell.workspace.views[0]
        await data.open_csv(csv_regressao)
        await shell._on_reset_all()
        assert asked == ["Reiniciar tudo?"]
        assert shell.ctx.state.csv_path == csv_regressao and shell.workspace.views[0] is data

    asyncio.run(flow())


def test_reiniciar_tudo_bloqueado_durante_o_treino(csv_regressao):
    async def flow():
        shell, asked, toasts = make_shell()
        await shell.start_create()
        shell.ctx.state.training = True
        shell.refresh()
        assert shell.reset_btn.disabled
        await shell._on_reset_all()
        assert asked == [] and toasts[-1][0] == "warning"

    asyncio.run(flow())


def test_menu_e_botoes_por_modo():
    async def flow():
        shell, _asked, _toasts = make_shell()
        assert shell.ctx.state.mode is None and shell.workspace is None
        assert shell.reset_btn not in shell.appbar.actions  # nada de "Reiniciar tudo" no menu
        await shell.start_create()
        await shell._on_back_to_menu()
        assert shell.ctx.state.mode is None and shell.reset_btn not in shell.appbar.actions

    asyncio.run(flow())

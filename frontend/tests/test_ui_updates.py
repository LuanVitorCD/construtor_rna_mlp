"""Regressões do 3º bloco: atualização da tela, navegação sem desanexar abas e eixos dos gráficos."""

import asyncio
import random

import pytest

from frontend.app import CREATE_STEPS, Workspace
from frontend.components import ui
from frontend.components.charts import LossChart, nice_ticks
from frontend.state import AppState
from frontend.tests.ui_harness import FakePage, build_app


@pytest.fixture
def csv_regressao(tmp_path):
    path = tmp_path / "dados.csv"
    lines = ["x1,x2,x3,y"] + [f"{i},{i % 7},{i * 0.5},{2 * i + 1}" for i in range(60)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def test_handlers_terminam_com_update_da_pagina(csv_regressao, monkeypatch):
    """Bug: botões de ação só mostravam o resultado depois de trocar de aba.

    O Flet pula o auto-update se o handler chamou update() no meio; por isso toda
    mensagem mostrada por um handler precisa ser seguida de um page.update().
    """
    events: list[str] = []
    original_show = ui.Message.show

    def recording_show(self, *args, **kwargs):
        events.append("show")
        return original_show(self, *args, **kwargs)

    monkeypatch.setattr(ui.Message, "show", recording_show)

    async def flow():
        state, page, (data, arch, train, ev, pred), _t, _gw = build_app()
        page.update = lambda *c: events.append("update")
        for handler in (lambda: data.open_csv(csv_regressao), lambda: arch._on_create(None),
                        lambda: data._on_apply(None), lambda: ev._on_evaluate_all(None)):
            events.clear()
            await handler()
            assert "show" in events and events[-1] == "update", events
        pred.paste.value = "1; 2; 3"
        pred._on_fill(None)
        events.clear()
        await pred._on_predict(None)
        assert events[-1] == "update"
        train.epochs.value = "3"
        events.clear()
        await train._on_start(None)
        assert events[-1] == "update"
        await page.tasks[-1]
        assert events[-1] == "update"  # o "pump" também termina atualizando

    asyncio.run(flow())


class _DummyView:
    def __init__(self, name):
        import flet as ft

        self.root = ft.Text(name)
        self.refreshes = 0

    def refresh(self):
        self.refreshes += 1


def test_navegacao_so_alterna_visibilidade_das_abas():
    """Bug: "duas Rede atual". Desanexar/reanexar a mesma árvore confundia o diff do Flet."""
    page = FakePage()
    views = [_DummyView("a"), _DummyView("b"), _DummyView("c")]
    ws = Workspace(views, CREATE_STEPS[:3], done=lambda s: [False] * 3, page=page)
    container = ws.content.content
    mounted = list(container.controls)
    assert [p.visible for p in ws.pages] == [True, False, False]
    ws.navigate(2)
    assert [p.visible for p in ws.pages] == [False, False, True] and ws.current == 2
    ws.navigate(0)
    assert ws.content.content is container and container.controls == mounted  # nada foi desanexado
    ws.refresh(AppState())
    assert all(v.refreshes == 1 for v in views)


@pytest.mark.parametrize("seed", range(5))
def test_marcas_do_eixo_sempre_cobrem_os_dados(seed):
    """Bug: a curva saía do gráfico porque a última marca ficava abaixo do valor máximo."""
    rng = random.Random(seed)
    for _ in range(200):
        scale = 10 ** rng.uniform(-6, 6)
        lo = rng.uniform(-1, 1) * scale
        hi = lo + rng.uniform(1e-3, 10) * scale
        ticks = nice_ticks(lo, hi)
        assert ticks[0] <= lo + 1e-12 * scale and ticks[-1] >= hi - 1e-12 * scale, (lo, hi, ticks)
        assert len(ticks) == len(set(ticks)) and 2 <= len(ticks) <= 12


def test_limites_do_grafico_contem_todas_as_curvas():
    chart = LossChart()
    history = [{"epoch": i, "train_mse": 9.5 - i * 0.01, "val_mse": 9.975 - i * 0.01} for i in range(1, 51)]
    for log_scale in (False, True):
        chart.log_scale = log_scale
        chart.render(history, 50)
        ys = [p.y for series in chart.chart.data_series for p in series.points]
        xs = [p.x for series in chart.chart.data_series for p in series.points]
        assert chart.chart.min_y <= min(ys) and max(ys) <= chart.chart.max_y
        assert chart.chart.min_x <= min(xs) and max(xs) <= chart.chart.max_x

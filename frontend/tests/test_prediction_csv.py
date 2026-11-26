"""Aba Previsão, modo "Arquivo CSV" (sem abrir janela, com o DemoService).

Prever todas as linhas de um CSV, com a coluna da saída real escondida e, se o usuário
quiser, mostrada ao lado do previsto para comparar.
"""

import asyncio
import csv

import pytest

from frontend.batch_prediction import ROLE_HIDDEN, ROLE_IGNORE, ROLE_INPUT
from frontend.tests.ui_harness import build_app, build_use_app


@pytest.fixture
def csv_treino(tmp_path):
    path = tmp_path / "treino.csv"
    lines = ["x1,x2,x3,y"] + [f"{i},{i % 7},{i * 0.5},{2 * i + 1}" for i in range(60)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def csv_novos(tmp_path):
    """Outras colunas, em outra ordem, com um id e a resposta certa (y)."""
    path = tmp_path / "novos.csv"
    lines = ["id,x3,y,x1,x2"] + [f"{100 + i},{i * 0.25},{i + 1},{i / 2},{i % 3}" for i in range(150)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def header_texts(panel):
    table = panel.table_holder.controls[0].controls[0]
    return [c.label.value for c in table.columns]


async def regression_app(csv_treino):
    state, page, views, toasts, gateway = build_app()
    data, arch, _train, _ev, pred = views
    await data.open_csv(csv_treino)
    await arch._on_create(None)
    assert state.data_ready
    return state, pred, toasts, gateway


def test_esconde_a_saida_preve_e_compara_quando_pedido(csv_treino, csv_novos, tmp_path):
    async def flow():
        state, pred, toasts, gateway = await regression_app(csv_treino)
        pred.show_csv()
        assert pred.csv_panel.root.visible and not pred.manual_root.visible
        panel = pred.csv_panel
        await panel.open_csv(csv_novos)
        # Papéis pelos nomes do treino: id ignorado, y escondido, entradas na ordem do treino.
        assert panel._roles == {"id": ROLE_IGNORE, "x3": ROLE_INPUT, "y": ROLE_HIDDEN, "x1": ROLE_INPUT,
                                "x2": ROLE_INPUT}
        assert panel.plan.input_cols == ["x1", "x2", "x3"] and panel.plan_message.level == "info"
        assert not panel.predict_btn.disabled

        await panel._on_predict(None)
        assert len(panel.outputs) == 150 and panel.predict_message.level == "success"
        # O mesmo que prever a 1ª linha à mão (x1=0, x2=0, x3=0).
        single = await gateway.call("predict", [0.0, 0.0, 0.0])
        assert panel.outputs[0] == pytest.approx(single["outputs"])
        # A coluna oculta não aparece; a comparação fica disponível, mas desligada.
        assert header_texts(panel) == ["Linha", "x1", "x2", "x3", "Previsto (ŷ)"]
        assert len(panel.table_holder.controls[0].controls[0].rows) == 100  # tabela limitada
        assert "150" in panel.table_holder.controls[1].value
        assert not panel.compare_switch.disabled and not panel.compare_switch.value
        assert not panel.chart_section.visible
        assert [t.content.label for t in panel.metrics.controls] == ["Linhas previstas: 150"]

        panel.compare_switch.value = True
        panel._on_compare_toggle(None)
        assert header_texts(panel)[-2:] == ["Real (y)", "Erro (ŷ − y)"]
        assert "Coluna oculta usada como resposta esperada (y): y" in panel.compare_hint.value
        labels = [t.content.label for t in panel.metrics.controls]
        assert any(t.startswith("MSE") for t in labels) and any(t.startswith("R²") for t in labels)
        assert panel.chart_section.visible
        ys = [i + 1 for i in range(150)]
        mse = sum((p[0] - y) ** 2 for p, y in zip(panel.outputs, ys)) / 150
        assert panel.comparison["mse"] == pytest.approx(mse)

        out = str(tmp_path / "res")
        assert panel.export(out)
        with open(out + ".csv", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["x1", "x2", "x3", "previsto_y", "real_y", "erro_y"] and len(rows) == 151
        assert toasts[-1][0] == "success"

    asyncio.run(flow())


def test_mudar_papeis_e_rede_invalida_a_previsao(csv_treino, csv_novos):
    async def flow():
        state, pred, _toasts, _gw = await regression_app(csv_treino)
        panel = pred.csv_panel
        await panel.open_csv(csv_novos)
        await panel._on_predict(None)
        outputs = panel.outputs

        # Só a coluna oculta mudou: a previsão continua, a comparação some.
        panel._on_role_change("y", [ROLE_IGNORE])
        assert panel.outputs is outputs and panel.compare_switch.disabled
        assert "Nenhuma coluna" in panel.compare_hint.value
        panel._on_role_change("y", [ROLE_HIDDEN])
        assert not panel.compare_switch.disabled

        # Entrada a mais: o plano não fecha e a previsão anterior é descartada.
        panel._on_role_change("id", [ROLE_INPUT])
        assert panel.outputs is None and panel.plan is None and panel.plan_message.level == "error"
        await panel._on_predict(None)
        assert panel.outputs is None and panel.predict_message.level == "error"
        panel._on_role_change("id", [ROLE_IGNORE])
        await panel._on_predict(None)
        assert panel.outputs is not None

        # Rede treinada de novo (pesos mudaram): a previsão fica desatualizada.
        state.trained_epochs += 5
        state.notify()
        assert panel.outputs is None and panel.predict_message.level == "info"

    asyncio.run(flow())


def test_valor_invalido_aponta_a_linha(csv_treino, tmp_path):
    async def flow():
        _state, pred, _toasts, _gw = await regression_app(csv_treino)
        path = tmp_path / "ruim.csv"
        path.write_text("x1,x2,x3\n1,2,3\n4,,6\n", encoding="utf-8")
        panel = pred.csv_panel
        await panel.open_csv(str(path))
        assert panel.plan and panel.plan.hidden_cols == []
        await panel._on_predict(None)
        assert panel.outputs is None and panel.predict_message.level == "error"
        assert panel.predict_message.title == "Valor inválido no arquivo"

        await panel.open_csv(str(tmp_path / "nao_existe.csv"))
        assert panel.file_message.level == "error"

    asyncio.run(flow())


def test_sem_rede_nao_preve(csv_novos):
    async def flow():
        state, _page, views, _toasts, _gw = build_app()
        panel = views[-1].csv_panel
        await panel.open_csv(csv_novos)
        assert panel.predict_btn.disabled and not panel.columns_section.visible
        await panel._on_predict(None)
        assert panel.outputs is None

    asyncio.run(flow())


def test_classificacao_no_modo_usar_modelo(tmp_path):
    async def flow():
        train = tmp_path / "flores.csv"
        train.write_text("a,b,classe\n" + "\n".join(f"{i},{i % 5},{['x', 'y', 'z'][i % 3]}" for i in range(45)),
                         encoding="utf-8")
        state, _page, (data, arch, *_rest), _toasts, gateway = build_app()
        data.task.selected = ["classification"]
        data._on_task_change(None)
        await data.open_csv(str(train))
        await arch._on_create(None)
        model = str(tmp_path / "m.json")
        await gateway.call("save", model)

        ustate, _upage, (_info, upred), _ut, _ugw = await build_use_app(model)
        panel = upred.csv_panel
        await panel.open_csv(str(train))
        assert panel._roles["classe"] == ROLE_HIDDEN
        await panel._on_predict(None)
        assert panel.labels and len(panel.labels) == 45
        assert header_texts(panel) == ["Linha", "a", "b", "Classe prevista"]
        panel.compare_switch.value = True
        panel._on_compare_toggle(None)
        assert header_texts(panel)[-2:] == ["Classe real", "Resultado"]
        labels = [t.content.label for t in panel.metrics.controls]
        assert any(t.startswith("Taxa de acertos") for t in labels)
        expected = sum(p == t for p, t in zip(panel.labels, panel.table.column_values("classe"))) / 45
        assert panel.comparison["accuracy"] == pytest.approx(expected)

    asyncio.run(flow())

"""Fluxo completo pelas telas (sem abrir janela), usando o DemoService.

Os handlers são chamados diretamente, como se o usuário clicasse.
"""

import asyncio

import pytest

from frontend.tests.ui_harness import build_app as _build_app, table_rows


def build_app():
    state, page, views, toasts, _gateway = _build_app()
    return state, page, views, toasts


@pytest.fixture
def csv_regressao(tmp_path):
    path = tmp_path / "dados.csv"
    lines = ["x1,x2,x3,y"] + [f"{i},{i % 7},{i * 0.5},{2 * i + 1}" for i in range(60)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def csv_classes(tmp_path):
    path = tmp_path / "flores.csv"
    rows = ["a,b,classe"] + [f"{i},{i % 5},{['x', 'y', 'z'][i % 3]}" for i in range(45)]
    path.write_text("\n".join(rows), encoding="utf-8")
    return str(path)


def test_fluxo_completo_de_regressao(csv_regressao):
    async def flow():
        state, page, (arch, data, train, ev, pred), _ = build_app()
        # Antes de tudo, treinar e prever ficam bloqueados (com a lista de pendências).
        assert train.start_btn.disabled and pred.predict_btn.disabled

        arch.n_inputs.value = "3"
        await arch._on_create(None)
        assert state.has_network and state.architecture_text() == "3–3–1"
        assert not pred.predict_btn.disabled  # já dá para prever com pesos iniciais

        await data.open_csv(csv_regressao)
        assert state.columns == ["x1", "x2", "x3", "y"]
        assert data.compat_message.visible and not data.adjust_btn.visible  # compatível
        await data._on_load(None)
        assert state.has_dataset and state.dataset["rows"] == 60
        assert table_rows(data)[0] == ["0", "0", "0", "1"]  # prévia {"x", "y"} vira células

        data.train_pct.value, data.val_pct.value, data.test_pct.value = "60", "20", "20"
        await data._on_split(None)
        assert state.split_size("train") == 36 and state.split_size("test") == 12
        data.norm_group.value = "zscore"
        await data._on_norm_change(None)
        assert state.normalization == "zscore"

        assert not train.start_btn.disabled
        train.epochs.value, train.lr.value = "25", "0,05"
        await train._on_start(None)
        assert state.training and train.stop_btn.disabled is False
        await page.tasks[-1]  # espera o "pump" terminar
        assert not state.training and len(state.history) == 25 and state.trained_epochs == 25
        assert train.chart._semantics.visible and not train.chart.placeholder.visible
        assert train.insights.controls  # leitura da curva preenchida

        await ev._on_evaluate_all(None)
        assert set(state.evaluations) == {"train", "val", "test"}
        assert not ev.comparison_section.visible  # o service não devolve y_true/y_pred
        assert not ev.confusion_section.visible  # regressão: sem matriz
        tile_labels = [tile.content.label for tile in ev.metrics.controls]
        summary_cols = [c.label.value for c in ev.summary_holder.content.columns]
        assert not any("RMSE" in t for t in tile_labels + summary_cols)  # RMSE removido
        assert summary_cols == ["Subconjunto", "MSE"]

        pred.paste.value = "10; 3; 5"
        pred._on_fill(None)
        assert [f.value for f in pred.fields] == ["10", "3", "5"]
        await pred._on_predict(None)
        assert len(pred._history) == 1 and not pred.message.visible

    asyncio.run(flow())


def test_fluxo_de_classificacao(csv_classes):
    async def flow():
        state, page, (arch, data, train, ev, pred), _ = build_app()
        arch.task.selected = ["classification"]
        arch._on_task_change(None)
        assert arch.out_activation.value == "sigmoid"
        arch.n_inputs.value, arch.out_neurons.value = "2", "3"
        await arch._on_create(None)
        await data.open_csv(csv_classes)
        await data._on_load(None)
        assert state.class_labels == ["x", "y", "z"]  # lidos de service.meta
        assert table_rows(data)[0][-1] == "x"  # y one-hot decodificado para o rótulo
        await data._on_split(None)
        train.epochs.value = "5"
        await train._on_start(None)
        await page.tasks[-1]
        await ev._on_evaluate(None)
        assert ev.confusion_section.visible and not ev.comparison_section.visible
        # Matriz: acertos (diagonal) em verde, erros em vermelho, zeros sem cor.
        grid = ev.confusion_holder.controls[0].controls[0].controls  # linhas da grade
        matrix = state.evaluations["test"]["confusion_matrix"]
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                bg = grid[i + 1].controls[j + 1].bgcolor
                if value == 0:
                    assert bg is None
                else:
                    assert ("tertiaryfixeddim" if i == j else "error") in str(bg)
        for f, v in zip(pred.fields, ("1", "2")):
            f.value = v
        await pred._on_predict(None)
        assert pred._history[0][1]["label"] in {"x", "y", "z"}

    asyncio.run(flow())


def test_grafico_desejado_estimado_aparece_se_o_service_devolver_y(csv_regressao):
    """Extensão opcional: com y_true/y_pred no evaluate, a aba mostra o gráfico e MAE/R²."""
    async def flow():
        state, page, views, _toasts, gateway = _build_app()
        arch, data, _train, ev, _pred = views
        original = gateway.service.evaluate

        def evaluate_with_y(subset="test"):
            result = original(subset)
            result.update(y_true=[[1.0], [2.0], [3.0]], y_pred=[[1.1], [1.9], [3.2]])
            return result

        gateway.service.evaluate = evaluate_with_y
        arch.n_inputs.value = "3"
        await arch._on_create(None)
        await data.open_csv(csv_regressao)
        await data._on_load(None)
        await data._on_split(None)
        await ev._on_evaluate(None)
        assert ev.comparison_section.visible
        labels = [tile.content.label for tile in ev.metrics.controls]
        assert any(label.startswith("R²") for label in labels)

    asyncio.run(flow())


def test_erros_aparecem_na_tela(csv_regressao):
    async def flow():
        state, _page, (arch, data, train, _ev, pred), _ = build_app()
        # formulário inválido: nada é enviado ao backend
        arch.hidden_rows[0].neurons.value = "abc"
        await arch._on_create(None)
        assert not state.has_network and arch.hidden_rows[0].neurons.error
        assert arch.message.visible

        arch.hidden_rows[0].neurons.value = "4"
        await arch._on_create(None)  # rede 2-4-1
        await data.open_csv(csv_regressao)  # CSV com 3 entradas
        assert data.adjust_btn.visible  # oferece ajustar a arquitetura
        await data._on_load(None)
        assert not state.has_dataset  # DatasetError do backend...
        assert data.load_message.level == "error" and data.load_message.title == "Problema nos dados"

        data._on_adjust(None)
        assert arch.n_inputs.value == "3" and arch.out_neurons.value == "1"

        data.train_pct.value = "90"
        await data._on_split(None)  # soma 120% (e sem dataset) -> erro
        assert not state.has_split

        pred.fields[0].value = "x"
        await pred._on_predict(None)
        assert pred.fields[0].error and pred.message.visible

    asyncio.run(flow())


def test_recriar_rede_reseta_etapas(csv_regressao):
    async def flow():
        state, page, (arch, data, train, ev, _pred), _ = build_app()
        arch.n_inputs.value = "3"
        await arch._on_create(None)
        await data.open_csv(csv_regressao)
        await data._on_load(None)
        await data._on_split(None)
        train.epochs.value = "3"
        await train._on_start(None)
        await page.tasks[-1]
        assert state.is_trained
        await arch._on_create(None)  # confirm() devolve True no teste
        assert not state.has_dataset and not state.has_split and not state.is_trained
        assert train.start_btn.disabled and ev.eval_btn.disabled

    asyncio.run(flow())


def test_parar_treinamento(csv_regressao):
    async def flow():
        state, page, (arch, data, train, _ev, _pred), _ = build_app()
        arch.n_inputs.value = "3"
        await arch._on_create(None)
        await data.open_csv(csv_regressao)
        await data._on_load(None)
        await data._on_split(None)
        train.epochs.value = "2000"
        await train._on_start(None)
        await asyncio.sleep(0.3)
        train._on_stop(None)
        await page.tasks[-1]
        assert not state.training and 0 < len(state.history) < 2000
        assert train.message.title == "Treinamento interrompido"

    asyncio.run(flow())

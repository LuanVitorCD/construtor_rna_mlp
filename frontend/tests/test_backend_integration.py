"""Frontend × backend real: as telas usando o ``backend.service.MLPService`` de verdade.

Confere se o que a UI envia e lê bate com o service implementado (formatos de retorno,
erros, threads de treino), nos dois modos da aplicação. Pulado se o pacote ``backend``
não estiver disponível.
"""

import asyncio

import pytest

service_module = pytest.importorskip("backend.service")
from backend.errors import BackendError  # noqa: E402

from frontend.model_files import read_model  # noqa: E402
from frontend.tests.ui_harness import build_app, build_use_app, table_rows  # noqa: E402
from frontend.views.architecture import IO_CUSTOM  # noqa: E402


def real_app():
    return build_app(service_module.MLPService(), BackendError)


@pytest.fixture
def csv_regressao(tmp_path):
    path = tmp_path / "casas.csv"
    lines = ["area,quartos,idade,preco"] + [
        f"{40 + i * 1.5},{1 + i % 4},{i % 30},{100 + 2 * i + (i % 4) * 10}" for i in range(80)
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def csv_classes(tmp_path):
    path = tmp_path / "flores.csv"
    centers = {"setosa": (1.0, 1.0), "versicolor": (4.0, 4.0), "virginica": (8.0, 1.0)}
    rows = ["a,b,especie"]
    for k in range(20):
        for name, (cx, cy) in centers.items():
            rows.append(f"{cx + (k % 5) * 0.1},{cy + (k % 3) * 0.1},{name}")
    path.write_text("\n".join(rows), encoding="utf-8")
    return str(path)


def test_criar_modelo_de_regressao_ponta_a_ponta(csv_regressao, tmp_path):
    async def flow():
        state, page, views, _toasts, gateway = real_app()
        data, arch, train, ev, pred = views

        await data.open_csv(csv_regressao)
        assert state.columns == ["area", "quartos", "idade", "preco"]
        assert (arch.n_inputs.value, arch.n_outputs.value) == ("3", "1")  # automático pelo CSV
        data.norm_group.value = "zscore"
        await arch._on_create(None)  # backend: create_network + load_dataset + split + normalização
        assert state.network["total_parameters"] == state.n_params == 16  # 3-3-1
        assert [s["neurons"] for s in state.layer_specs] == [3, 1]  # backend usa "n_neurons"
        assert state.data_ready and state.dataset["rows"] == 80
        assert table_rows(data)[0] == ["40", "1", "0", "100"]  # prévia {"x","y"} do backend
        assert (state.split_size("train"), state.split_size("val"), state.split_size("test")) == (56, 12, 12)
        assert state.normalization == "zscore" and gateway.read_norm_kind() == "zscore"

        train.epochs.value, train.lr.value, train.shuffle_seed.value = "15", "0,01", "1"
        await train._on_start(None)
        await asyncio.wait_for(page.tasks[-1], timeout=60)
        assert not state.training and [h["epoch"] for h in state.history] == list(range(1, 16))
        assert set(state.history[0]) >= {"epoch", "train_mse", "val_mse"}
        assert train.chart._semantics.visible

        await ev._on_evaluate_all(None)
        assert set(state.evaluations) == {"train", "val", "test"}
        assert ev.message.level == "success"
        # R² da tela = 1 − SQ_res/SQ_tot calculado aqui com as previsões da própria rede do backend
        service = gateway.service
        test = service.split.test
        y = [float(row[0]) for row in test.Y]
        y_hat = [float(service.network.forward(x)[0]) for x in test.X]
        mean = sum(y) / len(y)
        ss_res = sum((a - b) ** 2 for a, b in zip(y, y_hat))
        ss_tot = sum((a - mean) ** 2 for a in y)
        assert state.evaluations["test"]["r2"] == pytest.approx(1 - ss_res / ss_tot)
        assert [c.label.value for c in ev.summary_holder.content.columns] == ["Subconjunto", "MSE", "R²"]
        assert not ev.comparison_section.visible and not ev.confusion_section.visible

        pred.paste.value = "100; 2; 10"
        pred._on_fill(None)
        await pred._on_predict(None)
        assert not pred.message.visible and len(pred._history[0][1]["outputs"]) == 1

        # Salvar e abrir no modo "Usar modelo existente"
        model = str(tmp_path / "modelo.json")
        await gateway.call("save", model)
        ustate, _upage, (info, upred), _t, ugw = await build_use_app(model, service_module.MLPService(),
                                                                     BackendError)
        assert ustate.input_names() == ["area", "quartos", "idade"]
        assert ustate.output_names() == ["preco"]
        assert ustate.loaded_normalization == "zscore"
        assert [f.label for f in upred.fields] == ["area", "quartos", "idade"]
        assert info.file_name.value == "modelo.json"
        # Mesma rede, mesma normalização -> mesma saída
        same = await ugw.call("predict", [100.0, 2.0, 10.0])
        assert same["outputs"] == pytest.approx(pred._history[0][1]["outputs"])

    asyncio.run(flow())


def test_classificacao_com_backend_real(csv_classes):
    async def flow():
        state, page, views, _toasts, _gateway = real_app()
        data, arch, train, ev, pred = views
        data.task.selected = ["classification"]
        data._on_task_change(None)
        await data.open_csv(csv_classes)
        assert state.suggested_io() == (2, 3)  # 3 classes contadas antes da rede existir
        await arch._on_create(None)
        assert state.data_ready
        assert state.class_labels == ["setosa", "versicolor", "virginica"]  # via service.meta
        assert table_rows(data)[0][-1] == "setosa"  # one-hot decodificado
        train.epochs.value, train.lr.value = "30", "0,1"
        await train._on_start(None)
        await asyncio.wait_for(page.tasks[-1], timeout=60)
        await ev._on_evaluate(None)
        result = state.evaluations["test"]
        assert {"mse", "accuracy", "confusion_matrix", "classes"} <= set(result)
        assert ev.confusion_section.visible
        for f, v in zip(pred.fields, ("8", "1")):
            f.value = v
        await pred._on_predict(None)
        assert pred._history[0][1]["label"] in state.class_labels

    asyncio.run(flow())


def test_modelo_como_base_com_backend_real(csv_classes, tmp_path):
    async def flow():
        state, _page, views, _toasts, gateway = real_app()
        data, arch, _train, _ev, _pred = views
        data.task.selected = ["classification"]
        data._on_task_change(None)
        await data.open_csv(csv_classes)
        await arch._on_create(None)
        model = str(tmp_path / "base.json")
        await gateway.call("save", model)

        state2, _p2, views2, _t2, gateway2 = real_app()
        data2, arch2 = views2[0], views2[1]
        await data2.open_csv(csv_classes)
        info = await read_model(gateway2, model)
        state2.forget_network()
        data2.apply_template(info)
        arch2.apply_template(info)
        state2.notify()
        assert state2.task_choice == "classification"
        assert (arch2.n_inputs.value, arch2.n_outputs.value) == ("2", "3")
        await arch2._on_create(None)
        assert state2.data_ready and state2.architecture_text() == "2–3–3"

    asyncio.run(flow())


def test_erros_do_backend_viram_mensagens(csv_regressao):
    async def flow():
        state, page, views, _toasts, _gateway = real_app()
        data, arch, train, _ev, _pred = views
        await data.open_csv(csv_regressao)
        arch.io_mode.selected = [IO_CUSTOM]
        arch._on_io_mode_change(None)
        arch.n_inputs.value = "2"  # CSV tem 3 entradas
        await arch._on_create(None)
        assert state.has_network and not state.has_dataset
        assert data.apply_message.level == "error" and data.apply_message.title == "Problema nos dados"
        assert arch.message.level == "error"

        arch.n_inputs.value = "3"
        data.train_pct.value, data.val_pct.value, data.test_pct.value = "0,5", "0,5", "99"
        await arch._on_create(None)  # 0,5% de 80 linhas = 0 amostras -> DatasetError do backend
        assert not state.has_split and data.apply_message.level == "error"

        data.train_pct.value, data.val_pct.value, data.test_pct.value = "70", "15", "15"
        await data._on_apply(None)
        assert state.data_ready
        train.epochs.value = "100000"
        await train._on_start(None)
        await asyncio.sleep(0.2)
        train._on_stop(None)  # Trainer.stop() + on_finish no finally do backend
        await asyncio.wait_for(page.tasks[-1], timeout=30)
        assert not state.training and train.message.title == "Treinamento interrompido"

    asyncio.run(flow())


def test_previsao_por_csv_com_modelo_salvo(csv_regressao, tmp_path):
    """Modo "Usar modelo existente" + arquivo CSV: a coluna "preco" fica oculta e só entra na
    comparação. A normalização do modelo salvo é aplicada pelo backend em cada linha."""
    async def flow():
        state, page, views, _toasts, gateway = real_app()
        data, arch, train, _ev, _pred = views
        await data.open_csv(csv_regressao)
        data.norm_group.value = "zscore"
        await arch._on_create(None)
        train.epochs.value, train.lr.value = "20", "0,01"
        await train._on_start(None)
        await asyncio.wait_for(page.tasks[-1], timeout=60)
        model = str(tmp_path / "casas.json")
        await gateway.call("save", model)

        _ustate, _upage, (_info, upred), _ut, ugw = await build_use_app(
            model, service_module.MLPService(), BackendError)
        panel = upred.csv_panel
        await panel.open_csv(csv_regressao)
        assert panel.plan.input_cols == ["area", "quartos", "idade"] and panel.plan.hidden_cols == ["preco"]
        await panel._on_predict(None)
        assert len(panel.outputs) == 80
        first = await ugw.call("predict", [40.0, 1.0, 0.0])
        assert panel.outputs[0] == pytest.approx(first["outputs"])
        panel.compare_switch.value = True
        panel._on_compare_toggle(None)
        precos = [100 + 2 * i + (i % 4) * 10 for i in range(80)]
        mse = sum((p[0] - y) ** 2 for p, y in zip(panel.outputs, precos)) / 80
        assert panel.comparison["mse"] == pytest.approx(mse)
        mean = sum(precos) / 80
        var = sum((y - mean) ** 2 for y in precos) / 80
        assert panel.comparison["r2"] == pytest.approx(1 - mse / var)

    asyncio.run(flow())

"""Frontend × backend real: as telas usando o ``backend.service.MLPService`` de verdade.

Confere se o que a UI envia e lê bate com o service implementado (formatos de retorno,
erros, threads de treino). Pulado se o pacote ``backend`` não estiver disponível.
"""

import asyncio

import pytest

service_module = pytest.importorskip("backend.service")
from backend.errors import BackendError  # noqa: E402

from frontend.tests.ui_harness import build_app, table_rows  # noqa: E402


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


def test_regressao_ponta_a_ponta_com_backend_real(csv_regressao, tmp_path):
    async def flow():
        state, page, views, _toasts, gateway = real_app()
        arch, data, train, ev, pred = views
        assert state.activations == ["linear", "sigmoid", "tanh", "relu"] or state.activations

        arch.n_inputs.value = "3"
        await arch._on_create(None)
        assert state.network["total_parameters"] == state.n_params == 16  # 3-3-1
        assert [s["neurons"] for s in state.layer_specs] == [3, 1]  # backend usa "n_neurons"

        await data.open_csv(csv_regressao)
        assert state.columns == ["area", "quartos", "idade", "preco"]
        await data._on_load(None)
        assert state.has_dataset and state.dataset["rows"] == 80
        assert table_rows(data)[0] == ["40", "1", "0", "100"]  # prévia {"x","y"} do backend

        await data._on_split(None)
        assert (state.split_size("train"), state.split_size("val"), state.split_size("test")) == (56, 12, 12)
        data.norm_group.value = "zscore"
        await data._on_norm_change(None)
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
        assert not ev.comparison_section.visible and not ev.confusion_section.visible

        pred.paste.value = "100; 2; 10"
        pred._on_fill(None)
        await pred._on_predict(None)
        assert not pred.message.visible and len(pred._history[0][1]["outputs"]) == 1

        # Salvar e abrir de novo (mesma lógica do botão "Abrir modelo")
        model = str(tmp_path / "modelo.json")
        await gateway.call("save", model)
        fresh_state, _page2, fresh_views, _t, fresh_gw = real_app()
        await fresh_gw.call("load", model)
        description = await fresh_gw.call("describe_network")
        fresh_state.loaded_meta = description.get("meta") or fresh_gw.read_meta()
        fresh_state.on_model_loaded(description, model)
        fresh_state.loaded_normalization = fresh_gw.read_norm_kind()
        fresh_state.notify()
        assert fresh_state.input_names() == ["area", "quartos", "idade"]
        assert fresh_state.output_names() == ["preco"]
        assert fresh_state.loaded_normalization == "zscore"
        assert [f.label for f in fresh_views[4].fields] == ["area", "quartos", "idade"]
        assert fresh_views[1].model_note.level == "warning"  # aviso sobre a normalização salva
        # Mesma rede, mesma normalização -> mesma saída
        same = await fresh_gw.call("predict", [100.0, 2.0, 10.0])
        assert same["outputs"] == pytest.approx(pred._history[0][1]["outputs"])

    asyncio.run(flow())


def test_classificacao_com_backend_real(csv_classes):
    async def flow():
        state, page, views, _toasts, _gateway = real_app()
        arch, data, train, ev, pred = views
        arch.task.selected = ["classification"]
        arch._on_task_change(None)
        arch.n_inputs.value, arch.out_neurons.value = "2", "3"
        await arch._on_create(None)
        await data.open_csv(csv_classes)
        await data._on_load(None)
        assert state.has_dataset
        assert state.class_labels == ["setosa", "versicolor", "virginica"]  # via service.meta
        assert table_rows(data)[0][-1] == "setosa"  # one-hot decodificado
        await data._on_split(None)
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


def test_erros_do_backend_viram_mensagens(csv_regressao):
    async def flow():
        state, page, views, _toasts, _gateway = real_app()
        arch, data, train, _ev, _pred = views
        await arch._on_create(None)  # rede 2-3-1, CSV tem 3 entradas
        await data.open_csv(csv_regressao)
        await data._on_load(None)
        assert not state.has_dataset
        assert data.load_message.level == "error" and data.load_message.title == "Problema nos dados"

        arch.n_inputs.value = "3"
        await arch._on_create(None)
        await data._on_load(None)
        data.train_pct.value, data.val_pct.value, data.test_pct.value = "0,5", "0,5", "99"
        await data._on_split(None)  # 0,5% de 80 linhas = 0 amostras -> DatasetError do backend
        assert not state.has_split and data.split_message.level == "error"

        data.train_pct.value, data.val_pct.value, data.test_pct.value = "70", "15", "15"
        await data._on_split(None)
        train.epochs.value = "100000"
        await train._on_start(None)
        await asyncio.sleep(0.2)
        train._on_stop(None)  # Trainer.stop() + on_finish no finally do backend
        await asyncio.wait_for(page.tasks[-1], timeout=30)
        assert not state.training and train.message.title == "Treinamento interrompido"

    asyncio.run(flow())

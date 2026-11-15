import builtins
import json

import pytest

from frontend.demo_service import ConfigError, DemoBackendError
from frontend.service_gateway import GatewayError, ServiceGateway, create_gateway
from frontend.state import AppState

DESC = {"n_inputs": 2, "layers": [{"neurons": 3, "activation": "tanh"}, {"n_neurons": 1, "activation": "linear"}]}


def test_state_aceita_neurons_ou_n_neurons_e_calcula_parametros():
    s = AppState()
    s.on_network_created(DESC, "regression")
    assert s.layer_specs[-1] == {"neurons": 1, "activation": "linear"}
    assert s.n_outputs == 1 and s.n_params == 13
    assert s.architecture_text() == "2–3–1"
    assert s.input_names() == ["x₁", "x₂"]


def test_recriar_rede_invalida_dataset_divisao_e_historico():
    s = AppState()
    s.on_network_created(DESC, "regression")
    s.on_dataset_loaded({"rows": 10, "inputs": ["a", "b"], "outputs": ["y"]})
    s.on_split({"train": 7, "val": 2, "test": 1})
    s.history = [{"epoch": 1, "train_mse": 1, "val_mse": 1}]
    s.trained_epochs = 1
    assert s.input_names() == ["a", "b"] and s.split_size("val") == 2
    s.on_network_created(DESC, "regression")
    assert not s.has_dataset and not s.has_split and s.history == [] and not s.is_trained


def test_modelo_carregado_conta_como_treinado_e_usa_meta():
    s = AppState()
    s.loaded_meta = {"input_cols": ["a", "b"], "output_cols": ["y"]}
    s.on_model_loaded(DESC, "modelo.json")
    assert s.is_trained and s.input_names() == ["a", "b"] and s.output_names() == ["y"]


def test_split_size_aceita_lista_de_indices():
    s = AppState()
    s.split = {"train": [0, 1, 2], "val": [3], "test": []}
    assert (s.split_size("train"), s.split_size("val"), s.split_size("test")) == (3, 1, 0)


def test_gateway_traduz_erros():
    class Svc:
        def boom(self):
            raise ConfigError("lr deve ser > 0")

        def crash(self):
            raise RuntimeError("bug")

    gw = ServiceGateway(Svc(), DemoBackendError, demo=True)
    with pytest.raises(GatewayError) as e:
        gw.call_sync("boom")
    assert e.value.title == "Configuração inválida" and "lr" in e.value.message
    with pytest.raises(GatewayError) as e:
        gw.call_sync("crash")
    assert e.value.kind == "Unexpected"


def test_create_gateway_usa_demo_sem_backend(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("backend"):
            raise ModuleNotFoundError(f"No module named '{name}'", name="backend")
        return real_import(name, *args, **kwargs)

    monkeypatch.delenv("MLP_FRONTEND_DEMO", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert create_gateway().demo is True


def test_create_gateway_nao_esconde_erro_dentro_do_backend(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("backend"):
            raise ModuleNotFoundError("No module named 'numpyy'", name="numpyy")
        return real_import(name, *args, **kwargs)

    monkeypatch.delenv("MLP_FRONTEND_DEMO", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ModuleNotFoundError):
        create_gateway()


def test_demo_service_respeita_contrato(tmp_path):
    """Retornos serializáveis em JSON e erros do tipo certo, como o MLPService real."""
    from frontend.demo_service import DatasetError, DemoService, StateError

    csv_path = tmp_path / "d.csv"
    csv_path.write_text("a,b,y\n" + "\n".join(f"{i},{i * 2},{i * 3}" for i in range(20)), encoding="utf-8")
    svc = DemoService()
    with pytest.raises(StateError):
        svc.split_dataset(70, 15, 15)
    json.dumps(svc.create_network(2, [{"neurons": 3, "activation": "tanh"},
                                      {"neurons": 1, "activation": "linear"}], seed=1))
    with pytest.raises(DatasetError):
        svc.load_dataset(str(csv_path), ["a"], ["y"])  # 1 coluna para 2 entradas
    json.dumps(svc.load_dataset(str(csv_path), ["a", "b"], ["y"]))
    with pytest.raises(DatasetError):
        svc.split_dataset(70, 20, 20)
    assert svc.split_dataset(70, 15, 15, seed=1) == {"train": 14, "val": 3, "test": 3}
    json.dumps(svc.evaluate("test"))
    json.dumps(svc.predict([1.0, 2.0]))
    model = tmp_path / "m.json"
    svc.save(str(model))
    svc.load(str(model))
    assert svc.describe_network()["n_inputs"] == 2

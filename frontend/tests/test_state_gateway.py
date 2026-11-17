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


def test_suggested_io_regressao_e_classificacao():
    s = AppState()
    assert s.suggested_io() is None  # sem CSV
    s.csv_path, s.input_cols, s.output_cols = "d.csv", ["a", "b", "c"], ["y"]
    assert s.suggested_io() == (3, 1)
    s.output_cols = ["y", "z"]
    assert s.suggested_io() == (3, 2)
    s.task_choice = "classification"
    assert s.suggested_io() is None  # classificação exige 1 coluna de rótulo
    s.output_cols = ["classe"]
    s.class_labels_preview = ["a", "b"]
    assert s.suggested_io() == (3, 1)  # binária: 1 neurônio
    s.class_labels_preview = ["a", "b", "c", "d"]
    assert s.suggested_io() == (3, 4)  # one-hot
    s.class_labels_preview = ["a"]
    assert s.suggested_io() is None


def test_alterar_dados_aplicados_marca_como_desatualizado():
    s = AppState()
    s.mark_data_dirty()
    assert not s.data_dirty  # nada aplicado ainda: não há o que desatualizar
    s.on_network_created(DESC, "regression")
    s.on_dataset_loaded({"rows": 10})
    s.on_split({"train": 7, "val": 2, "test": 1})
    assert s.data_ready
    s.mark_data_dirty()
    assert s.data_dirty and not s.data_ready
    s.on_split({"train": 6, "val": 3, "test": 1})  # reaplicado
    assert s.data_ready


def test_dados_reaplicados_mantem_a_contagem_de_epocas():
    s = AppState()
    s.on_network_created(DESC, "regression")
    s.trained_epochs = 10
    s.on_dataset_loaded({"rows": 10})  # os pesos continuam na rede
    assert s.trained_epochs == 10
    s.on_network_created(DESC, "regression")  # pesos novos
    assert s.trained_epochs == 0


def test_forget_network_mantem_as_escolhas_de_dados():
    s = AppState()
    s.csv_path, s.input_cols, s.output_cols = "d.csv", ["a"], ["y"]
    s.on_network_created(DESC, "regression")
    s.forget_network()
    assert not s.has_network and s.csv_path == "d.csv" and s.input_cols == ["a"]


def test_distinct_values_conta_classes_como_o_backend(tmp_path):
    from frontend.csv_utils import distinct_values

    path = tmp_path / "c.csv"
    path.write_text("x,classe\n1, b\n2,a\n3,b \n4,c\n", encoding="utf-8")
    assert distinct_values(str(path), "classe") == ["a", "b", "c"]  # strip + ordem alfabética
    assert distinct_values(str(path), "nao_existe") is None
    assert distinct_values(str(tmp_path / "sumiu.csv"), "classe") is None
    many = tmp_path / "m.csv"
    many.write_text("y\n" + "\n".join(str(i) for i in range(50)), encoding="utf-8")
    assert len(distinct_values(str(many), "y", limit=10)) == 11  # para de contar acima do limite


def test_gateway_reset_cria_service_novo():
    from frontend.demo_service import DemoService

    gw = ServiceGateway(DemoService(), DemoBackendError, demo=True)
    old = gw.service
    gw.call_sync("create_network", 2, [{"neurons": 1, "activation": "linear"}])
    gw.reset()
    assert gw.service is not old and gw.service.network is None


def test_profile_columns_detecta_texto_e_conta_valores(tmp_path):
    from frontend.csv_utils import profile_columns

    path = tmp_path / "p.csv"
    path.write_text("x,y,classe\n1,2.5, a\n2,,b\n3,1e3,a\n", encoding="utf-8")
    prof = profile_columns(str(path), ["x", "y", "classe"])
    assert prof["x"].numeric and prof["x"].distinct == ["1", "2", "3"]
    assert prof["y"].non_numeric == "(vazio)"  # célula vazia não é número (como no backend)
    assert prof["classe"].non_numeric == "a" and prof["classe"].distinct == ["a", "b"]
    assert profile_columns(str(path), ["nao_existe"]) is None


def test_selection_problem_bloqueia_a_sugestao():
    s = AppState()
    s.csv_path, s.input_cols, s.output_cols = "d.csv", ["a"], ["y"]
    assert s.suggested_io() == (1, 1)
    s.selection_problem = "A saída “y” não é numérica."
    assert s.suggested_io() is None

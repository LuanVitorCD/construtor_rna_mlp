import json
import threading
import pytest
from backend.errors import DatasetError, StateError
from backend.service import MLPService


def treinar_e_esperar(svc: MLPService, **kw):
    pronto = threading.Event()
    svc.start_training(**kw, on_finish=pronto.set)
    assert pronto.wait(timeout=30)


def montar_fluxo_regressao(csv_path) -> MLPService:
    svc = MLPService()
    svc.create_network(
        n_inputs=2,
        layers=[{"neurons": 4, "activation": "tanh"}, {"neurons": 1, "activation": "linear"}],
        seed=1,
    )
    svc.load_dataset(csv_path, ["a", "b"], ["y"])
    svc.split_dataset(70, 15, 15, seed=1)
    svc.set_normalization("minmax")
    return svc


def test_fluxo_completo_de_regressao(csv_regressao, tmp_path):
    svc = montar_fluxo_regressao(csv_regressao)
    treinar_e_esperar(svc, epochs=30, lr=0.05, shuffle_seed=1)

    assert len(svc.get_history()) == 30
    assert svc.evaluate("test")["mse"] >= 0.0
    y = svc.predict([1.0, 2.0])["outputs"]

    model_path = tmp_path / "m.json"
    svc.save(model_path)

    svc2 = MLPService()
    svc2.load(model_path)
    assert svc2.predict([1.0, 2.0])["outputs"] == y


def test_historico_com_chaves_esperadas_pela_ui(csv_regressao):
    svc = montar_fluxo_regressao(csv_regressao)
    treinar_e_esperar(svc, epochs=5, lr=0.05, shuffle_seed=1)
    assert set(svc.get_history()[0].keys()) == {"epoch", "train_mse", "val_mse"}


def test_retornos_sao_serializaveis_em_json(csv_regressao):
    svc = montar_fluxo_regressao(csv_regressao)
    treinar_e_esperar(svc, epochs=5, lr=0.05, shuffle_seed=1)

    for retorno in (
        svc.describe_network(),
        svc.get_history(),
        svc.evaluate("test"),
        svc.predict([1.0, 2.0]),
        svc.list_activations(),
    ):
        json.dumps(retorno)


def test_stop_training_encerra_antes_do_fim(csv_regressao):
    svc = montar_fluxo_regressao(csv_regressao)
    pronto = threading.Event()
    svc.start_training(
        epochs=100000,
        lr=0.01,
        on_epoch=lambda r: svc.stop_training() if r["epoch"] == 2 else None,
        on_finish=pronto.set,
    )
    assert pronto.wait(timeout=10)
    assert len(svc.get_history()) < 100000


def test_on_epoch_recebe_dict_simples(csv_regressao):
    svc = montar_fluxo_regressao(csv_regressao)
    recebidos = []
    treinar_e_esperar(svc, epochs=3, lr=0.01, on_epoch=recebidos.append)
    assert len(recebidos) == 3
    assert isinstance(recebidos[0], dict)


def test_treinar_sem_rede():
    with pytest.raises(StateError):
        MLPService().start_training(10, 0.1)


def test_treinar_sem_split(csv_regressao):
    svc = MLPService()
    svc.create_network(2, [{"neurons": 1, "activation": "linear"}])
    svc.load_dataset(csv_regressao, ["a", "b"], ["y"])
    with pytest.raises(StateError):
        svc.start_training(10, 0.1)


def test_normalizar_antes_do_split(csv_regressao):
    svc = MLPService()
    svc.create_network(2, [{"neurons": 1, "activation": "linear"}])
    svc.load_dataset(csv_regressao, ["a", "b"], ["y"])
    with pytest.raises(StateError):
        svc.set_normalization("minmax")


def test_dataset_incompativel(csv_regressao):
    svc = MLPService()
    svc.create_network(3, [{"neurons": 1, "activation": "linear"}])
    with pytest.raises(DatasetError):
        svc.load_dataset(csv_regressao, ["a", "b"], ["y"])


def test_percentuais_invalidos(csv_regressao):
    svc = MLPService()
    svc.create_network(2, [{"neurons": 1, "activation": "linear"}])
    svc.load_dataset(csv_regressao, ["a", "b"], ["y"])
    with pytest.raises(DatasetError):
        svc.split_dataset(70, 20, 20)


def test_avaliar_teste_sem_split():
    svc = MLPService()
    svc.create_network(2, [{"neurons": 1, "activation": "linear"}])
    with pytest.raises(StateError):
        svc.evaluate("test")


def test_predict_sem_modelo():
    with pytest.raises(StateError):
        MLPService().predict([1.0, 2.0])


def test_trocar_normalizacao_reaplica_a_partir_dos_dados_originais(csv_regressao):
    svc = montar_fluxo_regressao(csv_regressao)
    svc.set_normalization("minmax")
    svc.set_normalization("zscore")
    assert svc.split.train.X.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)
    assert svc.split.train.X.std(axis=0) == pytest.approx([1.0, 1.0])


def test_criar_nova_rede_reseta_historico_e_split(csv_regressao):
    svc = montar_fluxo_regressao(csv_regressao)
    treinar_e_esperar(svc, epochs=3, lr=0.01)
    svc.create_network(2, [{"neurons": 1, "activation": "linear"}])
    assert svc.get_history() == []
    assert svc.split is None


def test_fluxo_completo_de_classificacao(csv_classificacao):
    svc = MLPService()
    svc.create_network(
        n_inputs=2,
        layers=[{"neurons": 5, "activation": "tanh"}, {"neurons": 3, "activation": "sigmoid"}],
        seed=1,
        task="classification",
    )
    svc.load_dataset(csv_classificacao, ["f1", "f2"], ["label"])
    svc.split_dataset(70, 15, 15, seed=1)
    svc.set_normalization("zscore")
    treinar_e_esperar(svc, epochs=20, lr=0.1)

    r = svc.evaluate("test")
    assert 0.0 <= r["accuracy"] <= 1.0
    assert "confusion_matrix" in r
    assert svc.predict([0.3, 0.7])["label"] in {"A", "B", "C"}
    assert svc.evaluate("val")["mse"] == pytest.approx(svc.get_history()[-1]["val_mse"])


def test_classificacao_com_numero_errado_de_saidas(csv_classificacao):
    svc = MLPService()
    svc.create_network(
        n_inputs=2,
        layers=[{"neurons": 1, "activation": "sigmoid"}],
        task="classification",
    )
    with pytest.raises(DatasetError):
        svc.load_dataset(csv_classificacao, ["f1", "f2"], ["label"])


def test_r2_multiplas_saidas_calcula_media_dos_r2_individuais(tmp_path):
    csv_path = tmp_path / "multi.csv"
    linhas = ["x,y1,y2"] + [f"{i},{1000.0 * i},{0.1 * i}" for i in range(1, 21)]
    csv_path.write_text("\n".join(linhas), encoding="utf-8")

    svc = MLPService()
    svc.create_network(
        n_inputs=1,
        layers=[{"neurons": 4, "activation": "tanh"}, {"neurons": 2, "activation": "linear"}],
        seed=1,
    )
    svc.load_dataset(csv_path, ["x"], ["y1", "y2"])
    svc.split_dataset(70, 15, 15, seed=1)
    svc.set_normalization("zscore")
    treinar_e_esperar(svc, epochs=30, lr=0.05, shuffle_seed=1)

    res = svc.evaluate("test")
    assert "r2" in res and isinstance(res["r2"], float)
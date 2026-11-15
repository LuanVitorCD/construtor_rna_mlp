import numpy as np
import pytest
from backend.dataset import Dataset, DataSplit
from backend.errors import DatasetError
from backend.network import NeuralNetwork
from backend.normalization import apply_normalization
from backend.persistence import LoadedModel, load_model, save_model
from backend.predictor import predict
from tests.helpers import params_iguais, snapshot_params


def _modelo_com_norm() -> LoadedModel:
    net = NeuralNetwork.from_config(
        2,
        [{"neurons": 3, "activation": "tanh"}, {"neurons": 1, "activation": "linear"}],
        seed=4,
    )
    ds = Dataset(np.array([[0.0, 0.0], [10.0, 20.0]]), np.array([[0.0], [1.0]]), ["a", "b"], ["y"])
    _, norm = apply_normalization(DataSplit(ds, ds, ds), "minmax")
    return LoadedModel(network=net, normalizer=norm, task="regression", meta={})


def test_previsao_regressao_formato():
    modelo = _modelo_com_norm()
    r = predict(modelo, [1.0, 2.0])
    assert len(r["outputs"]) == modelo.network.n_outputs


def test_entrada_com_tamanho_errado():
    modelo = _modelo_com_norm()
    with pytest.raises(DatasetError):
        predict(modelo, [1.0])


def test_previsao_nao_altera_os_parametros():
    modelo = _modelo_com_norm()
    antes = snapshot_params(modelo.network)
    predict(modelo, [1.0, 2.0])
    assert params_iguais(antes, snapshot_params(modelo.network))


def test_previsao_aplica_a_normalizacao_do_treino():
    modelo = _modelo_com_norm()
    esperado = modelo.network.forward(modelo.normalizer.transform(np.array([[10.0, 3.0]]))[0])
    assert predict(modelo, [10.0, 3.0])["outputs"] == pytest.approx(esperado.tolist())


def test_sem_normalizador_usa_o_valor_cru():
    net = NeuralNetwork.from_config(2, [{"neurons": 1, "activation": "linear"}], seed=2)
    modelo = LoadedModel(network=net, normalizer=None, task="regression", meta={})
    x = [3.0, -4.0]
    assert predict(modelo, x)["outputs"] == pytest.approx(net.forward(x).tolist())


def test_previsao_apos_salvar_e_carregar_e_igual(tmp_path):
    modelo = _modelo_com_norm()
    path = tmp_path / "pred.json"
    save_model(path, modelo.network, modelo.normalizer, modelo.task, modelo.meta)
    recarregado = load_model(path)
    assert predict(modelo, [4.5, 9.0]) == predict(recarregado, [4.5, 9.0])
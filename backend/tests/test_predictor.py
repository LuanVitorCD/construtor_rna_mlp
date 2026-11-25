import numpy as np
import pytest
from backend.data.dataset import Dataset, DataSplit
from backend.errors import DatasetError
from backend.neural_network.network import NeuralNetwork
from backend.data.normalization import apply_normalization
from backend.persistence import LoadedModel, load_model, save_model
from backend.predictor import predict
from backend.tests.helpers import params_iguais, snapshot_params


def _modelo_com_norm() -> LoadedModel:
    net = NeuralNetwork.from_config(
        2,
        [{"neurons": 3, "activation": "tanh"}, {"neurons": 1, "activation": "linear"}],
        seed=4,
    )
    ds = Dataset(np.array([[0.0, 0.0], [10.0, 20.0]]), np.array([[100.0], [500.0]]), ["a", "b"], ["y"])
    _, norm, y_norm = apply_normalization(DataSplit(ds, ds, ds), "minmax", normalize_y=True)
    return LoadedModel(network=net, normalizer=norm, task="regression", meta={}, y_normalizer=y_norm)


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
    raw_out = modelo.network.forward(modelo.normalizer.transform(np.array([[10.0, 3.0]]))[0])
    esperado = modelo.y_normalizer.inverse_transform(raw_out[None, :])[0]
    assert predict(modelo, [10.0, 3.0])["outputs"] == pytest.approx(esperado.tolist())


def test_sem_normalizador_usa_o_valor_cru():
    net = NeuralNetwork.from_config(2, [{"neurons": 1, "activation": "linear"}], seed=2)
    modelo = LoadedModel(network=net, normalizer=None, task="regression", meta={})
    x = [3.0, -4.0]
    assert predict(modelo, x)["outputs"] == pytest.approx(net.forward(x).tolist())


def test_previsao_apos_salvar_e_carregar_e_igual(tmp_path):
    modelo = _modelo_com_norm()
    path = tmp_path / "pred.json"
    save_model(
        path,
        modelo.network,
        modelo.normalizer,
        modelo.task,
        modelo.meta,
        y_normalizer=modelo.y_normalizer,
    )
    recarregado = load_model(path)
    assert predict(modelo, [4.5, 9.0]) == predict(recarregado, [4.5, 9.0])
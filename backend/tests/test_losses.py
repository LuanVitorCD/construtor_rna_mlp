import numpy as np
import pytest
from backend.dataset import load_csv, split
from backend.losses import evaluate_mse, mse
from backend.network import NeuralNetwork
from tests.helpers import params_iguais, snapshot_params


def test_mse_manual():
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[1.0], [2.0], [6.0]])
    assert mse(y_true, y_pred) == pytest.approx(3.0)


def test_mse_zero_quando_perfeito():
    Y = np.array([[0.5, 1.5], [-1.0, 2.0]])
    assert mse(Y, Y) == 0.0


def test_mse_com_varias_saidas_e_media_sobre_amostras_e_saidas():
    assert mse(np.array([[0.0, 0.0]]), np.array([[1.0, 3.0]])) == pytest.approx(5.0)


def test_evaluate_mse_nao_altera_a_rede():
    net = NeuralNetwork.from_config(2, [{"neurons": 1, "activation": "linear"}], seed=1)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Y = np.array([[5.0], [6.0]])
    antes = snapshot_params(net)
    evaluate_mse(net, X, Y)
    assert params_iguais(antes, snapshot_params(net))


def test_evaluate_mse_funciona_para_treino_validacao_e_teste(csv_regressao):
    ds = load_csv(csv_regressao, ["a", "b"], ["y"])
    s = split(ds, 70, 15, 15, seed=1)
    net = NeuralNetwork.from_config(2, [{"neurons": 1, "activation": "linear"}], seed=1)
    for subconjunto in (s.train, s.val, s.test):
        assert evaluate_mse(net, subconjunto.X, subconjunto.Y) >= 0.0
import numpy as np
import pytest
from backend.initializers import init_layer_params
from backend.network import NeuralNetwork
from backend.tests.helpers import params_iguais, snapshot_params


def test_formatos_e_vies_zero():
    W, b = init_layer_params(n_in=5, n_out=3, activation_name="tanh", rng=np.random.default_rng(0))
    assert W.shape == (3, 5)
    assert b.shape == (3,)
    assert np.all(b == 0.0)


def test_xavier_respeita_o_limite():
    W, _ = init_layer_params(50, 30, "tanh", np.random.default_rng(0))
    assert np.abs(W).max() <= np.sqrt(6.0 / (50 + 30))


def test_he_tem_desvio_padrao_esperado():
    W, _ = init_layer_params(n_in=500, n_out=500, activation_name="relu", rng=np.random.default_rng(0))
    assert W.std() == pytest.approx(np.sqrt(2.0 / 500), rel=0.05)


def test_he_nao_e_limitado_como_xavier():
    W, _ = init_layer_params(500, 500, "relu", np.random.default_rng(0))
    assert np.abs(W).max() > np.sqrt(6.0 / (500 + 500))


@pytest.mark.parametrize("ativacao", ["sigmoid", "tanh", "linear"])
def test_xavier_para_sigmoid_tanh_linear(ativacao):
    W, _ = init_layer_params(40, 20, ativacao, np.random.default_rng(1))
    assert np.abs(W).max() <= np.sqrt(6.0 / (40 + 20))


def test_mesma_seed_gera_mesma_rede():
    specs = [{"neurons": 4, "activation": "tanh"}, {"neurons": 2, "activation": "linear"}]
    a = NeuralNetwork.from_config(3, specs, seed=7)
    b = NeuralNetwork.from_config(3, specs, seed=7)
    assert params_iguais(snapshot_params(a), snapshot_params(b))


def test_seeds_diferentes_geram_redes_diferentes():
    specs = [{"neurons": 4, "activation": "tanh"}, {"neurons": 2, "activation": "linear"}]
    a = NeuralNetwork.from_config(3, specs, seed=1)
    b = NeuralNetwork.from_config(3, specs, seed=2)
    assert not params_iguais(snapshot_params(a), snapshot_params(b))


def test_sem_seed_nao_quebra():
    specs = [{"neurons": 3, "activation": "relu"}]
    net = NeuralNetwork.from_config(3, specs, seed=None)
    assert net.n_outputs == 3
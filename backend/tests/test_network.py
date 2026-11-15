import numpy as np
import pytest
from backend.activations import get_activation
from backend.errors import ConfigError, DatasetError
from backend.network import NeuralNetwork
from tests.helpers import params_iguais, snapshot_params


def test_construcao_dinamica_encadeia_as_entradas():
    net = NeuralNetwork(n_inputs=4)
    net.add_layer(6, "relu")
    net.add_layer(3, "tanh")
    net.add_layer(2, "linear")

    assert [l.n_inputs for l in net.layers] == [4, 6, 3]
    assert [len(l.neurons) for l in net.layers] == [6, 3, 2]
    assert net.n_outputs == 2
    assert all(len(n.weights) == l.n_inputs for l in net.layers for n in l.neurons)


def test_camadas_com_ativacoes_diferentes():
    net = NeuralNetwork.from_config(
        2,
        [{"neurons": 3, "activation": "relu"}, {"neurons": 1, "activation": "linear"}],
    )
    assert [l.activation_name for l in net.layers] == ["relu", "linear"]


def test_rede_sem_camadas_e_invalida():
    with pytest.raises(ConfigError):
        NeuralNetwork.from_config(2, [])


def test_camada_com_zero_neuronios_e_invalida():
    net = NeuralNetwork(2)
    with pytest.raises(ConfigError):
        net.add_layer(0, "relu")


def test_forward_manual_linear():
    net = NeuralNetwork(2)
    net.add_layer(1, "linear")
    net.layers[0].set_parameters([[1.0, 1.0]], [-1.5])
    assert net.forward([1, 1])[0] == pytest.approx(0.5)
    assert net.forward([0, 0])[0] == pytest.approx(-1.5)


def test_forward_com_relu_zera_negativos():
    net = NeuralNetwork(2)
    net.add_layer(1, "relu")
    net.layers[0].set_parameters([[1.0, 1.0]], [-1.5])
    assert net.forward([0, 0])[0] == pytest.approx(0.0)
    assert net.forward([1, 1])[0] == pytest.approx(0.5)


def test_forward_guarda_valores_intermediarios():
    net = NeuralNetwork.from_config(2, [{"neurons": 2, "activation": "tanh"}], seed=1)
    x = np.array([0.5, -1.2])
    net.forward(x)
    n = net.layers[0].neurons[0]
    act = get_activation("tanh")
    assert n.z == pytest.approx(np.dot(n.weights, x) + n.bias)
    assert n.output == pytest.approx(act.fn(n.z))


def test_forward_com_numero_errado_de_entradas():
    net = NeuralNetwork.from_config(2, [{"neurons": 1, "activation": "linear"}])
    with pytest.raises(DatasetError):
        net.forward([1, 2, 3])


def test_forward_funciona_com_qualquer_profundidade():
    for depth in (1, 2, 5):
        specs = [{"neurons": 3, "activation": "tanh"} for _ in range(depth - 1)]
        specs.append({"neurons": 2, "activation": "linear"})
        net = NeuralNetwork.from_config(4, specs, seed=depth)
        out = net.forward([0.1, 0.2, 0.3, 0.4])
        assert len(out) == 2


def test_forward_nao_altera_parametros():
    net = NeuralNetwork.from_config(2, [{"neurons": 3, "activation": "sigmoid"}], seed=3)
    antes = snapshot_params(net)
    net.forward([0.4, -0.7])
    assert params_iguais(antes, snapshot_params(net))
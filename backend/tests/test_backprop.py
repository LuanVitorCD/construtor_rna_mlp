import numpy as np
import pytest
from backend.losses import mse
from backend.network import NeuralNetwork
from tests.helpers import numeric_gradients, params_iguais, snapshot_params


def _analiticos(net: NeuralNetwork):
    return {
        (l, j): (n.delta * layer.last_inputs, n.delta)
        for l, layer in enumerate(net.layers)
        for j, n in enumerate(layer.neurons)
    }


@pytest.mark.parametrize(
    "ativacoes",
    [
        ("sigmoid", "sigmoid"),
        ("tanh", "linear"),
        ("sigmoid", "tanh", "linear"),
    ],
)
def test_gradiente_analitico_bate_com_numerico(ativacoes):
    specs = [{"neurons": 4, "activation": act} for act in ativacoes[:-1]]
    specs.append({"neurons": 2, "activation": ativacoes[-1]})
    net = NeuralNetwork.from_config(3, specs, seed=10)

    x = np.array([0.4, -0.8, 1.2])
    y = np.array([1.0, -0.5])

    num = numeric_gradients(net, lambda: mse(y, net.forward(x)))
    net.forward(x)
    net.backward(y, lr=0.0)
    ana = _analiticos(net)

    for chave in num:
        assert ana[chave][0] == pytest.approx(num[chave][0], rel=1e-4, abs=1e-8)
        assert ana[chave][1] == pytest.approx(num[chave][1], rel=1e-4, abs=1e-8)


def test_gradiente_com_relu():
    specs = [{"neurons": 4, "activation": "relu"}, {"neurons": 2, "activation": "linear"}]
    net = NeuralNetwork.from_config(3, specs, seed=7)
    x = np.array([1.1, -0.7, 0.9])
    y = np.array([0.5, 1.5])

    net.forward(x)
    # Garante distância do ponto não diferenciável z = 0 para a diferença finita ser precisa
    assert all(abs(n.z) > 1e-3 for n in net.layers[0].neurons)

    num = numeric_gradients(net, lambda: mse(y, net.forward(x)))
    net.forward(x)
    net.backward(y, lr=0.0)
    ana = _analiticos(net)

    for chave in num:
        assert ana[chave][0] == pytest.approx(num[chave][0], rel=1e-4, abs=1e-8)
        assert ana[chave][1] == pytest.approx(num[chave][1], rel=1e-4, abs=1e-8)


def test_todos_os_deltas_sao_calculados_antes_de_atualizar_os_pesos():
    specs = [
        {"neurons": 4, "activation": "tanh"},
        {"neurons": 3, "activation": "sigmoid"},
        {"neurons": 2, "activation": "linear"},
    ]
    net = NeuralNetwork.from_config(3, specs, seed=12)
    x = np.array([0.6, -0.4, 0.9])
    y = np.array([1.2, -0.3])

    antes = snapshot_params(net)
    num = numeric_gradients(net, lambda: mse(y, net.forward(x)))

    net.forward(x)
    net.backward(y, lr=0.5)

    for (l, j), (gw, gb) in num.items():
        n = net.layers[l].neurons[j]
        w0, b0 = antes[l][j]
        assert n.weights == pytest.approx(w0 - 0.5 * gw, rel=1e-4, abs=1e-8)
        assert n.bias == pytest.approx(b0 - 0.5 * gb, rel=1e-4, abs=1e-8)


def test_convencao_da_perda_sgd_um_passo():
    net = NeuralNetwork(1)
    net.add_layer(1, "linear")
    net.layers[0].set_parameters([[1.0]], [0.0])

    net.forward([2.0])
    net.backward([5.0], lr=0.1)

    n = net.layers[0].neurons[0]
    assert n.weights[0] == pytest.approx(2.2)
    assert n.bias == pytest.approx(0.6)


def test_atualizacao_reduz_o_erro_do_exemplo():
    net = NeuralNetwork.from_config(2, [{"neurons": 3, "activation": "tanh"}, {"neurons": 1, "activation": "linear"}], seed=5)
    x = np.array([1.0, -1.0])
    y = np.array([0.8])

    e0 = mse(y, net.forward(x))
    net.backward(y, lr=0.01)
    e1 = mse(y, net.forward(x))
    assert e1 < e0


def test_backward_com_lr_zero_nao_altera_parametros():
    net = NeuralNetwork.from_config(2, [{"neurons": 2, "activation": "sigmoid"}], seed=1)
    antes = snapshot_params(net)
    net.forward([0.5, 0.5])
    net.backward([1.0, 0.0], lr=0.0)
    assert params_iguais(antes, snapshot_params(net))


def test_deltas_ficam_guardados_nos_neuronios():
    net = NeuralNetwork.from_config(2, [{"neurons": 3, "activation": "tanh"}, {"neurons": 1, "activation": "sigmoid"}], seed=2)
    net.forward([0.3, -0.7])
    net.backward([1.0], lr=0.1)
    assert all(n.delta != 0.0 for layer in net.layers for n in layer.neurons)
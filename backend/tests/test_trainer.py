import numpy as np
import pytest
from backend.dataset import Dataset, DataSplit
from backend.errors import ConfigError, DatasetError
from backend.losses import mse
from backend.network import NeuralNetwork
from backend.normalization import apply_normalization
from backend.trainer import Trainer
from backend.tests.helpers import params_iguais, snapshot_params


def _dados_linha() -> tuple[Dataset, Dataset]:
    x_tr = np.linspace(0.0, 4.0, 20).reshape(-1, 1)
    y_tr = 2.0 * x_tr + 1.0
    x_va = np.array([[0.5], [1.5], [2.5], [3.5]])
    y_va = 2.0 * x_va + 1.0
    return Dataset(x_tr, y_tr, ["x"], ["y"]), Dataset(x_va, y_va, ["x"], ["y"])


def test_historico_tem_uma_entrada_por_epoca():
    train, val = _dados_linha()
    net = NeuralNetwork.from_config(1, [{"neurons": 1, "activation": "linear"}], seed=1)
    hist = Trainer(net, train, val, lr=0.01, epochs=7, shuffle_seed=1).run()
    assert [h.epoch for h in hist] == [1, 2, 3, 4, 5, 6, 7]
    assert all(h.train_mse >= 0.0 and h.val_mse >= 0.0 for h in hist)


def test_callback_chamado_a_cada_epoca():
    train, val = _dados_linha()
    net = NeuralNetwork.from_config(1, [{"neurons": 1, "activation": "linear"}], seed=1)
    recebidos = []
    Trainer(net, train, val, lr=0.01, epochs=5, shuffle_seed=1).run(on_epoch=recebidos.append)
    assert len(recebidos) == 5


def test_sgd_atualiza_a_cada_exemplo_e_nao_em_lote():
    train, val = _dados_linha()
    net = NeuralNetwork(1)
    net.add_layer(1, "linear")
    net.layers[0].set_parameters([[0.5]], [-0.2])

    lr = 0.02
    ordem = np.random.default_rng(5).permutation(len(train))
    w, b = 0.5, -0.2
    for i in ordem:
        erro = 2.0 * ((w * train.X[i][0] + b) - train.Y[i][0])
        w -= lr * erro * train.X[i][0]
        b -= lr * erro

    Trainer(net, train, val, lr=lr, epochs=1, shuffle_seed=5).run()
    assert net.layers[0].neurons[0].weights[0] == pytest.approx(w)
    assert net.layers[0].neurons[0].bias == pytest.approx(b)


def test_validacao_nao_influencia_os_pesos():
    train, val_a = _dados_linha()
    val_b = Dataset(val_a.X.copy(), val_a.Y + 50.0, ["x"], ["y"])

    net1 = NeuralNetwork.from_config(1, [{"neurons": 1, "activation": "linear"}], seed=3)
    net2 = NeuralNetwork.from_config(1, [{"neurons": 1, "activation": "linear"}], seed=3)

    t1 = Trainer(net1, train, val_a, lr=0.01, epochs=4, shuffle_seed=2)
    t2 = Trainer(net2, train, val_b, lr=0.01, epochs=4, shuffle_seed=2)
    t1.run()
    t2.run()

    assert params_iguais(snapshot_params(net1), snapshot_params(net2))
    assert t1.history[-1].val_mse != t2.history[-1].val_mse


def test_treinamento_reprodutivel():
    train, val = _dados_linha()
    net1 = NeuralNetwork.from_config(1, [{"neurons": 2, "activation": "tanh"}, {"neurons": 1, "activation": "linear"}], seed=9)
    net2 = NeuralNetwork.from_config(1, [{"neurons": 2, "activation": "tanh"}, {"neurons": 1, "activation": "linear"}], seed=9)

    h1 = Trainer(net1, train, val, lr=0.01, epochs=5, shuffle_seed=4).run()
    h2 = Trainer(net2, train, val, lr=0.01, epochs=5, shuffle_seed=4).run()

    assert params_iguais(snapshot_params(net1), snapshot_params(net2))
    assert [r.train_mse for r in h1] == [r.train_mse for r in h2]


def test_shuffle_seeds_diferentes_dao_trajetorias_diferentes():
    train, val = _dados_linha()
    net1 = NeuralNetwork.from_config(1, [{"neurons": 1, "activation": "linear"}], seed=1)
    net2 = NeuralNetwork.from_config(1, [{"neurons": 1, "activation": "linear"}], seed=1)

    h1 = Trainer(net1, train, val, lr=0.01, epochs=5, shuffle_seed=1).run()
    h2 = Trainer(net2, train, val, lr=0.01, epochs=5, shuffle_seed=99).run()
    assert h1[-1].train_mse != h2[-1].train_mse


def test_stop_interrompe_o_treino():
    train, val = _dados_linha()
    net = NeuralNetwork.from_config(1, [{"neurons": 1, "activation": "linear"}], seed=1)
    trainer = Trainer(net, train, val, lr=0.01, epochs=100, shuffle_seed=1)

    def cb(r):
        if r.epoch == 3:
            trainer.stop()

    trainer.run(on_epoch=cb)
    assert len(trainer.history) == 3


@pytest.mark.parametrize("lr, epochs", [(0, 10), (-0.1, 10), (0.1, 0)])
def test_hiperparametros_invalidos(lr, epochs):
    train, val = _dados_linha()
    net = NeuralNetwork.from_config(1, [{"neurons": 1, "activation": "linear"}])
    with pytest.raises(ConfigError):
        Trainer(net, train, val, lr=lr, epochs=epochs)


def test_validacao_vazia_e_recusada():
    train, _ = _dados_linha()
    val_vazio = Dataset(np.zeros((0, 1)), np.zeros((0, 1)), ["x"], ["y"])
    net = NeuralNetwork.from_config(1, [{"neurons": 1, "activation": "linear"}])
    with pytest.raises(DatasetError):
        Trainer(net, train, val_vazio, lr=0.01, epochs=5)


def test_trainer_calcula_mse_na_escala_original():
    train_raw, val_raw = _dados_linha()
    split_norm, _, y_norm = apply_normalization(
        DataSplit(train=train_raw, val=val_raw, test=val_raw),
        "zscore",
        normalize_y=True,
    )
    net = NeuralNetwork.from_config(1, [{"neurons": 1, "activation": "linear"}], seed=1)
    hist = Trainer(
        net,
        split_norm.train,
        split_norm.val,
        lr=0.01,
        epochs=3,
        shuffle_seed=1,
        y_normalizer=y_norm,
    ).run()

    preds_norm = np.array([net.forward(x) for x in split_norm.val.X], dtype=float)
    preds_real = y_norm.inverse_transform(preds_norm)
    assert hist[-1].val_mse == pytest.approx(mse(val_raw.Y, preds_real))


@pytest.mark.slow
def test_regressao_aprende_a_reta():
    train, val = _dados_linha()
    net = NeuralNetwork.from_config(1, [{"neurons": 1, "activation": "linear"}], seed=0)
    Trainer(net, train, val, lr=0.01, epochs=1500, shuffle_seed=0).run()
    n = net.layers[0].neurons[0]
    assert n.weights[0] == pytest.approx(2.0, abs=0.05)
    assert n.bias == pytest.approx(1.0, abs=0.05)


@pytest.mark.slow
def test_mse_de_treino_diminui_em_regressao_nao_linear():
    x = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y = x ** 2
    ds = Dataset(x, y, ["x"], ["y"])
    net = NeuralNetwork.from_config(
        1,
        [{"neurons": 8, "activation": "tanh"}, {"neurons": 1, "activation": "linear"}],
        seed=2,
    )
    hist = Trainer(net, ds, ds, lr=0.05, epochs=600, shuffle_seed=2).run()
    assert hist[-1].train_mse < 0.1 * hist[0].train_mse


@pytest.mark.slow
def test_xor_converge(dados_xor):
    X, Y = dados_xor
    ds = Dataset(X, Y, ["x1", "x2"], ["y"])
    net = NeuralNetwork.from_config(
        2,
        [{"neurons": 4, "activation": "tanh"}, {"neurons": 1, "activation": "sigmoid"}],
        seed=42,
    )
    hist = Trainer(net, ds, ds, lr=0.5, epochs=3000, shuffle_seed=42).run()
    assert hist[-1].train_mse < 0.05
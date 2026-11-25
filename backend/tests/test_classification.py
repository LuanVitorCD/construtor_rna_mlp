import numpy as np
import pytest
from backend.classification import LabelEncoder, accuracy, confusion_matrix
from backend.data.dataset import Dataset
from backend.errors import DatasetError
from backend.neural_network.network import NeuralNetwork
from backend.persistence import load_model, save_model
from backend.predictor import predict
from backend.trainer import Trainer


def test_binario_usa_uma_saida_e_limiar_05():
    enc = LabelEncoder().fit(["sim", "nao", "sim", "nao"])
    assert enc.classes_ == ["nao", "sim"]
    assert enc.n_outputs == 1
    assert enc.encode(["nao", "sim"]).tolist() == [[0.0], [1.0]]
    assert enc.decode([0.49]) == "nao"
    assert enc.decode([0.50]) == "sim"


def test_multiclasse_one_hot_e_argmax():
    enc = LabelEncoder().fit(["A", "B", "C"])
    assert enc.n_outputs == 3
    assert enc.encode(["B"]).tolist() == [[0.0, 1.0, 0.0]]
    assert enc.decode([0.1, 0.2, 0.7]) == "C"


def test_rotulo_desconhecido():
    enc = LabelEncoder().fit(["A", "B"])
    with pytest.raises(DatasetError):
        enc.encode(["Z"])


def test_accuracy():
    assert accuracy(["A", "B", "A", "B"], ["A", "B", "B", "B"]) == pytest.approx(0.75)


def test_matriz_de_confusao():
    cm = confusion_matrix(["A", "A", "B", "B"], ["A", "B", "B", "B"], classes=["A", "B"])
    assert cm.tolist() == [[1, 1], [0, 2]]


def test_classes_persistidas_no_modelo(tmp_path):
    net = NeuralNetwork.from_config(2, [{"neurons": 3, "activation": "sigmoid"}], seed=1)
    path = tmp_path / "cls.json"
    save_model(path, net, normalizer=None, task="classification", meta={"class_labels": ["A", "B", "C"]})
    modelo = load_model(path)
    assert modelo.meta["class_labels"] == ["A", "B", "C"]
    assert predict(modelo, [0.5, 0.5])["label"] in {"A", "B", "C"}


@pytest.mark.slow
def test_classificacao_binaria_simples_aprende():
    rng = np.random.default_rng(1)
    X0 = rng.normal(-2.0, 0.4, size=(25, 2))
    X1 = rng.normal(2.0, 0.4, size=(25, 2))
    X = np.vstack([X0, X1])
    y_raw = ["neg"] * 25 + ["pos"] * 25

    enc = LabelEncoder().fit(y_raw)
    ds = Dataset(X, enc.encode(y_raw), ["x1", "x2"], ["label"])
    net = NeuralNetwork.from_config(
        2,
        [{"neurons": 4, "activation": "tanh"}, {"neurons": 1, "activation": "sigmoid"}],
        seed=1,
    )
    Trainer(net, ds, ds, lr=0.1, epochs=150, shuffle_seed=1).run()
    preds = [enc.decode(net.forward(x)) for x in X]
    assert accuracy(y_raw, preds) > 0.95
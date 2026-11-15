import numpy as np
import pytest
from backend.dataset import Dataset, check_compatibility, load_csv, read_columns, split
from backend.errors import DatasetError
from backend.network import NeuralNetwork


def test_read_columns(csv_regressao):
    assert read_columns(csv_regressao) == ["a", "b", "y"]


def test_load_csv_separa_entradas_e_saidas(csv_regressao):
    ds = load_csv(csv_regressao, ["a", "b"], ["y"])
    assert ds.X.shape == (60, 2)
    assert ds.Y.shape == (60, 1)
    assert ds.Y[:, 0] == pytest.approx(2.0 * ds.X[:, 0] - ds.X[:, 1] + 1.0)


def test_coluna_inexistente(csv_regressao):
    with pytest.raises(DatasetError):
        load_csv(csv_regressao, ["a", "zzz"], ["y"])


def test_coluna_nao_pode_ser_entrada_e_saida(csv_regressao):
    with pytest.raises(DatasetError):
        load_csv(csv_regressao, ["a", "y"], ["y"])


def test_compatibilidade_entradas():
    net = NeuralNetwork.from_config(3, [{"neurons": 1, "activation": "linear"}])
    ds = Dataset(X=np.zeros((5, 2)), Y=np.zeros((5, 1)), input_cols=["a", "b"], output_cols=["y"])
    with pytest.raises(DatasetError):
        check_compatibility(ds, net)


def test_compatibilidade_saidas():
    net = NeuralNetwork.from_config(2, [{"neurons": 2, "activation": "linear"}])
    ds = Dataset(X=np.zeros((5, 2)), Y=np.zeros((5, 1)), input_cols=["a", "b"], output_cols=["y"])
    with pytest.raises(DatasetError):
        check_compatibility(ds, net)


def test_compatibilidade_ok():
    net = NeuralNetwork.from_config(2, [{"neurons": 1, "activation": "linear"}])
    ds = Dataset(X=np.zeros((5, 2)), Y=np.zeros((5, 1)), input_cols=["a", "b"], output_cols=["y"])
    check_compatibility(ds, net)


def test_split_tamanhos():
    ds = Dataset(X=np.zeros((100, 2)), Y=np.zeros((100, 1)), input_cols=["a", "b"], output_cols=["y"])
    s = split(ds, 70, 15, 15, seed=1)
    assert (len(s.train), len(s.val), len(s.test)) == (70, 15, 15)


def test_split_com_arredondamento_nao_perde_linhas():
    ds = Dataset(X=np.zeros((10, 2)), Y=np.zeros((10, 1)), input_cols=["a", "b"], output_cols=["y"])
    s = split(ds, 70, 20, 10, seed=1)
    assert len(s.train) + len(s.val) + len(s.test) == 10


def test_split_e_disjunto_e_completo():
    X = np.arange(30, dtype=float).reshape(30, 1)
    ds = Dataset(X=X, Y=X * 2, input_cols=["x"], output_cols=["y"])
    s = split(ds, 60, 20, 20, seed=4)

    ids = lambda d: set(d.X[:, 0].tolist())
    assert ids(s.train).isdisjoint(ids(s.val))
    assert ids(s.train).isdisjoint(ids(s.test))
    assert ids(s.val).isdisjoint(ids(s.test))
    assert ids(s.train) | ids(s.val) | ids(s.test) == ids(ds)


def test_split_mantem_pares_entrada_saida_alinhados(csv_regressao):
    ds = load_csv(csv_regressao, ["a", "b"], ["y"])
    s = split(ds, 70, 15, 15, seed=9)
    for sub in (s.train, s.val, s.test):
        assert sub.Y[:, 0] == pytest.approx(2.0 * sub.X[:, 0] - sub.X[:, 1] + 1.0)


def test_split_reprodutivel(csv_regressao):
    ds = load_csv(csv_regressao, ["a", "b"], ["y"])
    a = split(ds, 70, 15, 15, seed=3)
    b = split(ds, 70, 15, 15, seed=3)
    assert np.array_equal(a.train.X, b.train.X)


def test_split_com_seeds_diferentes(csv_regressao):
    ds = load_csv(csv_regressao, ["a", "b"], ["y"])
    assert not np.array_equal(split(ds, 70, 15, 15, seed=1).train.X, split(ds, 70, 15, 15, seed=2).train.X)


@pytest.mark.parametrize("tr, va, te", [(70, 20, 20), (50, 20, 20), (-10, 60, 50)])
def test_percentuais_invalidos(csv_regressao, tr, va, te):
    ds = load_csv(csv_regressao, ["a", "b"], ["y"])
    with pytest.raises(DatasetError):
        split(ds, tr, va, te, seed=1)


def test_treino_e_validacao_nao_podem_ficar_vazios():
    ds = Dataset(X=np.zeros((3, 1)), Y=np.zeros((3, 1)), input_cols=["x"], output_cols=["y"])
    with pytest.raises(DatasetError):
        split(ds, 100, 0, 0, seed=1)
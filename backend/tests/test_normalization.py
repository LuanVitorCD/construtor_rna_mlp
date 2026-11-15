import numpy as np
import pytest
from backend.dataset import Dataset, DataSplit
from backend.errors import ConfigError
from backend.normalization import Normalizer, apply_normalization


def _make_split() -> DataSplit:
    train = Dataset(np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]]), np.array([[1.0], [2.0], [3.0]]), ["a", "b"], ["y"])
    val = Dataset(np.array([[2.5, 15.0]]), np.array([[1.5]]), ["a", "b"], ["y"])
    test = Dataset(np.array([[20.0, 50.0]]), np.array([[5.0]]), ["a", "b"], ["y"])
    return DataSplit(train=train, val=val, test=test)


def test_minmax_treino_vai_de_0_a_1():
    s2, _ = apply_normalization(_make_split(), "minmax")
    assert s2.train.X.min(axis=0) == pytest.approx([0.0, 0.0])
    assert s2.train.X.max(axis=0) == pytest.approx([1.0, 1.0])


def test_zscore_treino_tem_media_0_e_desvio_1():
    s2, _ = apply_normalization(_make_split(), "zscore")
    assert s2.train.X.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)
    assert s2.train.X.std(axis=0) == pytest.approx([1.0, 1.0])


def test_parametros_vem_so_do_treino():
    sp = _make_split()
    _, norm = apply_normalization(sp, "minmax")
    assert norm.params["min"] == pytest.approx(sp.train.X.min(axis=0))
    assert norm.params["max"] == pytest.approx(sp.train.X.max(axis=0))


def test_sem_vazamento_valores_de_fora_do_treino_ultrapassam_0_1():
    s2, _ = apply_normalization(_make_split(), "minmax")
    assert s2.test.X.max() > 1.0


def test_normalizacao_aplicada_aos_tres_subconjuntos():
    sp = _make_split()
    s2, norm = apply_normalization(sp, "zscore")
    assert np.allclose(s2.val.X, norm.transform(sp.val.X))
    assert np.allclose(s2.test.X, norm.transform(sp.test.X))


def test_saidas_nao_sao_normalizadas():
    sp = _make_split()
    s2, _ = apply_normalization(sp, "zscore")
    assert np.array_equal(s2.train.Y, sp.train.Y)
    assert np.array_equal(s2.test.Y, sp.test.Y)


def test_sem_normalizacao():
    sp = _make_split()
    s2, norm = apply_normalization(sp, None)
    assert norm is None
    assert np.array_equal(s2.train.X, sp.train.X)


def test_coluna_constante_nao_gera_nan_nem_inf():
    train = Dataset(np.array([[5.0, 1.0], [5.0, 3.0]]), np.array([[1.0], [2.0]]), ["a", "b"], ["y"])
    sp = DataSplit(train=train, val=train, test=train)
    for tipo in ("minmax", "zscore"):
        s2, _ = apply_normalization(sp, tipo)
        assert np.isfinite(s2.train.X).all()


def test_to_dict_from_dict_preservam_a_transformacao():
    sp = _make_split()
    _, norm = apply_normalization(sp, "zscore")
    norm2 = Normalizer.from_dict(norm.to_dict())
    assert np.allclose(norm2.transform(sp.test.X), norm.transform(sp.test.X))


def test_tipo_desconhecido():
    with pytest.raises(ConfigError):
        apply_normalization(_make_split(), "robust")
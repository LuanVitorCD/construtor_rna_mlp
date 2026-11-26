import pytest
from backend.neural_network.activations import get_activation, list_activations
from backend.errors import ConfigError


@pytest.mark.parametrize(
    "nome, z, esperado",
    [
        ("linear", -2.0, -2.0),
        ("linear", 3.0, 3.0),
        ("sigmoid", 0.0, 0.5),
        ("tanh", 0.0, 0.0),
        ("relu", -2.0, 0.0),
        ("relu", 3.0, 3.0),
    ],
)
def test_valor_da_funcao(nome, z, esperado):
    f = get_activation(nome).fn
    assert f(z) == pytest.approx(esperado)


@pytest.mark.parametrize(
    "nome, z, derivada",
    [
        ("linear", 5.0, 1.0),
        ("sigmoid", 0.0, 0.25),
        ("tanh", 0.0, 1.0),
        ("relu", 2.0, 1.0),
        ("relu", -2.0, 0.0),
    ],
)
def test_derivada(nome, z, derivada):
    act = get_activation(nome)
    a = act.fn(z)
    assert act.dfn(z, a) == pytest.approx(derivada)


@pytest.mark.parametrize("nome", ["sigmoid", "tanh", "relu", "linear"])
def test_derivada_confere_com_diferenca_finita(nome):
    act = get_activation(nome)
    for z in (-2.0, -0.5, 0.7, 1.9):
        num = (act.fn(z + 1e-6) - act.fn(z - 1e-6)) / 2e-6
        assert act.dfn(z, act.fn(z)) == pytest.approx(num, rel=1e-4, abs=1e-6)


def test_sigmoid_nao_estoura_em_valores_extremos():
    sig = get_activation("sigmoid").fn
    assert 0.0 <= sig(-1000.0) < 1e-100
    assert sig(1000.0) == pytest.approx(1.0)


def test_ativacao_desconhecida():
    with pytest.raises(ConfigError):
        get_activation("foo")


def test_lista_contem_as_quatro_obrigatorias():
    assert {"linear", "sigmoid", "tanh", "relu"} <= set(list_activations())
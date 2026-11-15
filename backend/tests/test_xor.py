import pytest
from backend.losses import evaluate_mse
from tests.helpers import numeric_gradients

SAIDAS_SLIDE = {
    (0, 0): 0.4036,
    (0, 1): 0.4322,
    (1, 0): 0.4351,
    (1, 1): 0.4583,
}


def test_forward_xor_bate_com_os_slides(rede_xor_slide):
    for (x1, x2), esperado in SAIDAS_SLIDE.items():
        assert rede_xor_slide.forward([x1, x2])[0] == pytest.approx(esperado, abs=1e-4)


def test_z_da_saida_bate_com_o_slide(rede_xor_slide):
    rede_xor_slide.forward([0, 0])
    assert rede_xor_slide.layers[-1].neurons[0].z == pytest.approx(-0.3905, abs=1e-3)
    rede_xor_slide.forward([0, 1])
    assert rede_xor_slide.layers[-1].neurons[0].z == pytest.approx(-0.2730, abs=1e-3)


def test_ativacoes_ocultas_do_slide(rede_xor_slide):
    rede_xor_slide.forward([0, 1])
    saidas = [n.output for n in rede_xor_slide.layers[0].neurons]
    assert saidas == pytest.approx([0.5893, 0.3505, 0.3818], abs=1e-4)


def test_mse_xor_bate_com_o_slide(rede_xor_slide, dados_xor):
    X, Y = dados_xor
    assert evaluate_mse(rede_xor_slide, X, Y) == pytest.approx(0.253632, abs=1e-4)


def test_gradiente_em_lote_da_camada_de_saida_bate_com_o_slide(rede_xor_slide, dados_xor):
    X, Y = dados_xor
    grads = numeric_gradients(rede_xor_slide, lambda: evaluate_mse(rede_xor_slide, X, Y))
    gw, _ = grads[(1, 0)]
    assert gw == pytest.approx([-0.01668882, -0.01096593, -0.01075111], abs=1e-4)
"""Contraste das cores (WCAG 2.1 AA) nos dois temas."""

import pytest

from frontend import theme


@pytest.mark.parametrize("name, tokens", [("claro", theme.LIGHT), ("escuro", theme.DARK)])
@pytest.mark.parametrize("fg, bg, minimum", theme.CONTRAST_PAIRS)
def test_pares_de_contraste(name, tokens, fg, bg, minimum):
    ratio = theme.contrast_ratio(tokens[fg], tokens[bg])
    assert ratio >= minimum, f"tema {name}: {fg} sobre {bg} = {ratio:.2f}:1 (mínimo {minimum}:1)"


def test_cores_das_logos_sao_usadas():
    assert theme.LIGHT["primary"] == theme.BRAND_CC_BLUE
    assert theme.LIGHT["secondary"] == theme.BRAND_SW_NAVY
    assert theme.DARK["primary"] == theme.BRAND_SW_CYAN
    assert theme.LIGHT["surface"] == "#FFFFFF"  # fundo branco


def test_ciano_original_nao_serve_para_texto_sobre_branco():
    # Motivo de o tema claro usar o ciano escurecido como "tertiary".
    assert theme.contrast_ratio(theme.BRAND_SW_CYAN, "#FFFFFF") < 3
    assert theme.contrast_ratio(theme.LIGHT["tertiary"], "#FFFFFF") >= 4.5


def test_contrast_ratio_extremos():
    assert theme.contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert theme.contrast_ratio("#777777", "#777777") == pytest.approx(1.0)


def test_mesmas_chaves_nos_dois_temas_e_temas_constroem():
    assert theme.LIGHT.keys() == theme.DARK.keys()
    assert theme.light_theme().color_scheme.primary == theme.BRAND_CC_BLUE
    assert theme.dark_theme().color_scheme.primary == theme.BRAND_SW_CYAN

"""Paleta e temas (claro/escuro) da aplicação.

As cores de marca foram extraídas das logos do enunciado (Trabalho Prático 02):

- Ciência da Computação (Unijuí): azul ``#1362AD``
- Engenharia de Software (Unijuí): azul-marinho ``#115D8F`` e ciano ``#18A8DA``

Regras de acessibilidade (WCAG 2.1, nível AA), conferidas em ``tests/test_theme.py``:

- texto normal: contraste >= 4,5:1;
- bordas de campos, linhas de gráfico e ícones: contraste >= 3:1.

O ciano original tem só 2,74:1 sobre branco. No tema claro ele aparece
escurecido (``#127EA4``, 4,62:1). No tema escuro o ciano original vira a cor
primária, porque ali o contraste é alto (5,94:1).

Os componentes usam *tokens* do ColorScheme (``ft.Colors.PRIMARY``, ``ft.Colors.ON_SURFACE`` ...)
em vez de hexadecimais. Assim, trocar o tema recolore tudo sem reconstruir a interface.

Verde de sucesso: o Material 3 não tem um papel "success". Por isso os papéis
``tertiary_fixed*`` (que nenhum componente padrão usa) guardam o verde de
"concluído/sucesso". O vermelho de erro é o papel ``error``. Use os apelidos
``SUCCESS*`` abaixo em vez dos nomes originais.
"""

from __future__ import annotations

import flet as ft

BRAND_CC_BLUE = "#1362AD"
BRAND_SW_NAVY = "#115D8F"
BRAND_SW_CYAN = "#18A8DA"

# Verde de sucesso (papéis tertiary_fixed* reaproveitados; ver docstring).
SUCCESS = ft.Colors.TERTIARY_FIXED_DIM  # ícones, bordas e textos curtos
SUCCESS_CONTAINER = ft.Colors.TERTIARY_FIXED  # fundo de mensagens de sucesso
ON_SUCCESS_CONTAINER = ft.Colors.ON_TERTIARY_FIXED  # texto sobre esse fundo

# Chaves = nomes dos campos de ft.ColorScheme.
LIGHT: dict[str, str] = {
    "primary": BRAND_CC_BLUE,
    "on_primary": "#FFFFFF",
    "primary_container": "#D7E5F6",
    "on_primary_container": "#0B3A66",
    "secondary": BRAND_SW_NAVY,
    "on_secondary": "#FFFFFF",
    "secondary_container": "#D5E8F5",
    "on_secondary_container": "#0B3C5D",
    "tertiary": "#127EA4",  # ciano da logo escurecido 25% (4,62:1 sobre branco)
    "on_tertiary": "#FFFFFF",
    "tertiary_container": "#D3F0FA",
    "on_tertiary_container": "#06384A",
    "error": "#B3261E",
    "on_error": "#FFFFFF",
    "error_container": "#F9DEDC",
    "on_error_container": "#410E0B",
    "tertiary_fixed": "#D7F3DE",  # sucesso: fundo
    "on_tertiary_fixed": "#0B3D1A",  # sucesso: texto
    "tertiary_fixed_dim": "#1E7A34",  # sucesso: ícone/borda/texto curto
    "surface": "#FFFFFF",
    "on_surface": "#1A1C1E",
    "on_surface_variant": "#4A5560",
    "surface_container_lowest": "#FFFFFF",
    "surface_container_low": "#F6F8FB",
    "surface_container": "#EFF3F8",
    "surface_container_high": "#E8EEF5",
    "surface_container_highest": "#E1E8F0",
    "outline": "#6B7785",
    "outline_variant": "#C5CFDA",
    "inverse_surface": "#2E3238",
    "on_inverse_surface": "#F0F2F5",
    "inverse_primary": "#9CC3EE",
    "surface_tint": BRAND_CC_BLUE,
    "shadow": "#000000",
    "scrim": "#000000",
}

DARK: dict[str, str] = {
    "primary": BRAND_SW_CYAN,
    "on_primary": "#00263A",
    "primary_container": "#0D4F6E",
    "on_primary_container": "#CBEFFC",
    "secondary": "#89B0D6",  # azul CC clareado 50%
    "on_secondary": "#0B2742",
    "secondary_container": "#1D4466",
    "on_secondary_container": "#D5E6F7",
    "tertiary": "#A1C0DE",
    "on_tertiary": "#102A43",
    "tertiary_container": "#24435F",
    "on_tertiary_container": "#D9E8F6",
    "error": "#FFB4AB",
    "on_error": "#690005",
    "error_container": "#93000A",
    "on_error_container": "#FFDAD6",
    "tertiary_fixed": "#1B4227",  # sucesso: fundo
    "on_tertiary_fixed": "#CDEFD6",  # sucesso: texto
    "tertiary_fixed_dim": "#7DD99A",  # sucesso: ícone/borda/texto curto
    "surface": "#0F1720",
    "on_surface": "#E6EDF3",
    "on_surface_variant": "#A9B6C3",
    "surface_container_lowest": "#0A1118",
    "surface_container_low": "#131C26",
    "surface_container": "#16212C",
    "surface_container_high": "#1C2935",
    "surface_container_highest": "#23313F",
    "outline": "#8696A7",
    "outline_variant": "#34434F",
    "inverse_surface": "#E6EDF3",
    "on_inverse_surface": "#1A222B",
    "inverse_primary": BRAND_CC_BLUE,
    "surface_tint": BRAND_SW_CYAN,
    "shadow": "#000000",
    "scrim": "#000000",
}

# Pares (frente, fundo, contraste mínimo) que a interface realmente usa.
# Testados para os dois temas em tests/test_theme.py.
TEXT = 4.5
GRAPHIC = 3.0
CONTRAST_PAIRS: list[tuple[str, str, float]] = [
    ("on_surface", "surface", TEXT),
    ("on_surface", "surface_container_low", TEXT),
    ("on_surface", "surface_container_high", TEXT),
    ("on_surface_variant", "surface", TEXT),
    ("on_surface_variant", "surface_container_low", TEXT),
    ("on_surface_variant", "surface_container_high", TEXT),
    ("primary", "surface", TEXT),  # botões de texto, links, valores destacados
    ("primary", "surface_container_low", TEXT),
    ("on_primary", "primary", TEXT),  # botões preenchidos, barra superior
    ("on_primary_container", "primary_container", TEXT),
    ("on_secondary_container", "secondary_container", TEXT),
    ("on_tertiary_container", "tertiary_container", TEXT),
    ("on_error_container", "error_container", TEXT),
    ("error", "surface", TEXT),  # mensagens de erro dos campos
    ("error", "surface_container_low", TEXT),
    ("outline", "surface", GRAPHIC),  # bordas de campos de formulário
    ("outline", "surface_container_low", GRAPHIC),
    ("tertiary", "surface", GRAPHIC),  # linha de validação no gráfico
    ("tertiary", "surface_container_low", GRAPHIC),
    ("secondary", "surface_container_low", GRAPHIC),  # marcador da melhor época
    # Verde (sucesso) e vermelho (erro)
    ("on_tertiary_fixed", "tertiary_fixed", TEXT),  # texto da mensagem de sucesso
    ("tertiary_fixed_dim", "tertiary_fixed", GRAPHIC),  # ícone/borda verde na mensagem
    ("tertiary_fixed_dim", "surface", TEXT),  # "✓ soma 100%"
    ("tertiary_fixed_dim", "surface_container_low", TEXT),
    ("tertiary_fixed_dim", "primary_container", GRAPHIC),  # ✓ na aba selecionada
    ("error", "error_container", GRAPHIC),  # ícone/borda vermelha na mensagem
]


def _relative_luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"cor inválida: {hex_color!r}")
    channels = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """Razão de contraste WCAG 2.1 entre duas cores ``#RRGGBB`` (1 a 21)."""
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def build_theme(tokens: dict[str, str], *, dark: bool) -> ft.Theme:
    scheme = ft.ColorScheme(**tokens)
    if dark:
        appbar = ft.AppBarTheme(
            bgcolor=tokens["surface_container_high"], color=tokens["on_surface"], elevation=0
        )
    else:
        appbar = ft.AppBarTheme(bgcolor=tokens["primary"], color=tokens["on_primary"], elevation=0)
    return ft.Theme(
        color_scheme_seed=tokens["primary"],
        color_scheme=scheme,
        appbar_theme=appbar,
        scaffold_bgcolor=tokens["surface"],
        visual_density=ft.VisualDensity.STANDARD,
        # Foco do teclado bem visível (WCAG 2.4.7)
        focus_color=ft.Colors.with_opacity(0.24, tokens["primary"]),
    )


def light_theme() -> ft.Theme:
    return build_theme(LIGHT, dark=False)


def dark_theme() -> ft.Theme:
    return build_theme(DARK, dark=True)

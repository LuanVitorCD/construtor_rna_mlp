"""Leitura e formatação de números digitados pelo usuário (sem dependência do Flet).

Aceita o padrão brasileiro (vírgula decimal: ``0,01``) e o padrão do Python
(``0.01``, ``1e-3``). As mensagens de erro já saem prontas para a interface.
"""

from __future__ import annotations

import math
import re


class ParseError(ValueError):
    """Valor digitado inválido; ``str(erro)`` é a mensagem para o usuário."""


def _normalize_decimal(text: str, field: str) -> str:
    cleaned = text.strip().replace(" ", "")
    if "," in cleaned and "." in cleaned:
        raise ParseError(f"{field}: use só um separador decimal (vírgula ou ponto).")
    return cleaned.replace(",", ".")


def parse_float(
    text: str | None,
    *,
    field: str,
    minimum: float | None = None,
    min_inclusive: bool = True,
    maximum: float | None = None,
    allow_empty: bool = False,
) -> float | None:
    raw = (text or "").strip()
    if not raw:
        if allow_empty:
            return None
        raise ParseError(f"{field}: campo obrigatório.")
    try:
        value = float(_normalize_decimal(raw, field))
    except ValueError:
        raise ParseError(f"{field}: “{raw}” não é um número.") from None
    if not math.isfinite(value):
        raise ParseError(f"{field}: informe um número finito.")
    if minimum is not None:
        if min_inclusive and value < minimum:
            raise ParseError(f"{field}: deve ser maior ou igual a {format_number(minimum)}.")
        if not min_inclusive and value <= minimum:
            raise ParseError(f"{field}: deve ser maior que {format_number(minimum)}.")
    if maximum is not None and value > maximum:
        raise ParseError(f"{field}: deve ser menor ou igual a {format_number(maximum)}.")
    return value


def parse_int(
    text: str | None,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
    allow_empty: bool = False,
) -> int | None:
    raw = (text or "").strip()
    if not raw:
        if allow_empty:
            return None
        raise ParseError(f"{field}: campo obrigatório.")
    if not re.fullmatch(r"[+-]?\d+", raw):
        raise ParseError(f"{field}: informe um número inteiro (ex.: 10).")
    value = int(raw)
    if minimum is not None and value < minimum:
        raise ParseError(f"{field}: deve ser no mínimo {minimum}.")
    if maximum is not None and value > maximum:
        raise ParseError(f"{field}: deve ser no máximo {maximum}.")
    return value


_STRONG_SEPARATORS = re.compile(r"[;\t\n]+")


def parse_values_line(text: str, *, expected: int | None = None) -> list[float]:
    """Converte uma linha colada (ex.: de uma planilha ou CSV) numa lista de floats.

    Separadores aceitos, por prioridade:
      1. ``;``, tabulação ou quebra de linha (as vírgulas viram decimais);
      2. espaços;
      3. vírgulas, quando a linha também tem pontos (``1.5,2.0``) ou quando o
         número de vírgulas bate com ``expected - 1`` (``1,2,3`` para 3 entradas).
    """
    raw = (text or "").strip()
    if not raw:
        raise ParseError("Cole ou digite os valores.")
    if _STRONG_SEPARATORS.search(raw):
        parts = _STRONG_SEPARATORS.split(raw)
    elif re.search(r"\s", raw):
        parts = raw.split()
    elif "," in raw and ("." in raw or (expected is not None and raw.count(",") == expected - 1)):
        parts = raw.split(",")
    else:
        parts = [raw]
    parts = [p.strip() for p in parts if p.strip()]
    values = [parse_float(p, field=f"Valor {i}") for i, p in enumerate(parts, start=1)]
    if expected is not None and len(values) != expected:
        raise ParseError(
            f"Foram encontrados {len(values)} valores, mas a rede espera {expected}. "
            "Separe os valores com ponto e vírgula (;)."
        )
    return [float(v) for v in values]  # type: ignore[arg-type]


def format_number(value: float | int | None, digits: int = 4) -> str:
    """Formata no padrão brasileiro com ``digits`` algarismos significativos."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "∞" if value > 0 else "−∞"
    if 10**digits <= abs(value) < 1e15:  # ex.: 12430,4 -> "12.430" (em vez de "1,243e4")
        return format_number(int(round(value))).replace("-", "−")
    text = f"{value:.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}e{int(exponent)}"
    return text.replace(".", ",").replace("-", "−")


def format_percent(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}%".replace(".", ",")


def to_input_text(value: float | int | None) -> str:
    """Valor para preencher um campo editável (sem sinal tipográfico, para poder ser relido)."""
    if value is None:
        return ""
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return str(int(value))
    return repr(float(value)).replace(".", ",")

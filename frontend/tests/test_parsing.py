import pytest

from frontend.parsing import (
    ParseError,
    format_fixed,
    format_number,
    format_percent,
    parse_float,
    parse_int,
    parse_values_line,
    to_input_text,
)


@pytest.mark.parametrize("text, expected", [("0,01", 0.01), ("0.01", 0.01), ("1e-3", 0.001), (" 2 ", 2.0),
                                            ("-1,5", -1.5)])
def test_parse_float_aceita_virgula_e_ponto(text, expected):
    assert parse_float(text, field="x") == pytest.approx(expected)


@pytest.mark.parametrize("text", ["abc", "1,000.5", "nan", "inf", "1..2"])
def test_parse_float_rejeita(text):
    with pytest.raises(ParseError):
        parse_float(text, field="x")


def test_parse_float_limites_e_vazio():
    with pytest.raises(ParseError, match="maior que 0"):
        parse_float("0", field="η", minimum=0, min_inclusive=False)
    with pytest.raises(ParseError, match="obrigatório"):
        parse_float("", field="η")
    assert parse_float("", field="η", allow_empty=True) is None
    with pytest.raises(ParseError):
        parse_float("101", field="%", maximum=100)


def test_parse_int():
    assert parse_int("42", field="n") == 42
    assert parse_int(" ", field="n", allow_empty=True) is None
    for bad in ("1.5", "3,0", "dez"):
        with pytest.raises(ParseError, match="inteiro"):
            parse_int(bad, field="n")
    with pytest.raises(ParseError, match="no mínimo 1"):
        parse_int("0", field="n", minimum=1)


@pytest.mark.parametrize(
    "text, expected, n",
    [
        ("5,1; 3,5; 1,4", [5.1, 3.5, 1.4], 3),  # padrão brasileiro com ;
        ("5.1\t3.5\t1.4", [5.1, 3.5, 1.4], 3),  # colado do Excel
        ("5.1 3.5 1.4", [5.1, 3.5, 1.4], 3),
        ("5.1,3.5,1.4", [5.1, 3.5, 1.4], 3),  # linha de CSV
        ("1,2,3", [1.0, 2.0, 3.0], 3),  # vírgulas = separadores quando batem com n-1
        ("1,5", [1.5], 1),  # vírgula decimal
    ],
)
def test_parse_values_line(text, expected, n):
    assert parse_values_line(text, expected=n) == pytest.approx(expected)


def test_parse_values_line_quantidade_errada():
    with pytest.raises(ParseError, match="espera 2"):
        parse_values_line("1; 2; 3", expected=2)


def test_format_number_padrao_brasileiro():
    assert format_number(0.123456) == "0,1235"
    assert format_number(1234567) == "1.234.567"
    assert format_number(-0.5) == "−0,5"
    assert format_number(1.5e-7) == "1,5e−7"
    assert format_number(12430.4) == "12.430"  # grandes: milhar, não notação científica
    assert format_number(-30000.0, 3) == "−30.000"
    assert format_number(2500.0, 3) == "2.500" and format_number(250.0, 3) == "250"
    assert format_number(None) == "—"
    assert format_number(float("nan")) == "NaN"
    assert format_percent(12.345) == "12,3%"


def test_to_input_text_pode_ser_relido():
    for v in (0.1, 2.0, 1e-5, -3.25):
        assert parse_float(to_input_text(v), field="x") == pytest.approx(v)


def test_format_fixed_para_r2():
    assert format_fixed(0.89912) == "0,899"
    assert format_fixed(-0.0111) == "−0,011"
    assert format_fixed(-2.42e-6) == "0,000"  # sem "−0,000"
    assert format_fixed(None) == "—" and format_fixed(float("nan")) == "—"

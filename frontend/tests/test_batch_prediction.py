"""Previsão de várias linhas por CSV: a parte sem Flet (frontend/batch_prediction.py)."""

import csv

import pytest

from frontend.batch_prediction import (
    ROLE_HIDDEN,
    ROLE_IGNORE,
    ROLE_INPUT,
    BatchError,
    compare_classification,
    compare_regression,
    comparison_problem,
    default_roles,
    export_rows,
    parse_inputs,
    plan_columns,
    read_table,
    write_csv,
)


def write(tmp_path, text, name="dados.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------- leitura
def test_le_o_csv_pulando_linhas_em_branco_e_guardando_a_linha_do_arquivo(tmp_path):
    path = write(tmp_path, " a , b ,y\n1,2,3\n\n4,5\n7,8,9,10\n")
    table = read_table(path)
    assert table.columns == ["a", "b", "y"]  # nomes sem espaços nas pontas
    assert table.rows == [["1", "2", "3"], ["4", "5", ""], ["7", "8", "9"]]  # curta completa; longa corta
    assert table.lines == [2, 4, 5]  # a linha em branco (3) conta na numeração do arquivo
    assert table.column_values("y") == ["3", "", "9"]


@pytest.mark.parametrize("text, trecho", [
    ("", "cabeçalho"),
    ("a,b,a\n1,2,3\n", "mesmo nome: a"),
    ("a,b\n\n", "não tem linhas de dados"),
])
def test_arquivos_que_nao_servem(tmp_path, text, trecho):
    with pytest.raises(BatchError, match=trecho):
        read_table(write(tmp_path, text))


def test_limite_de_linhas_e_arquivo_inexistente(tmp_path):
    path = write(tmp_path, "a\n" + "\n".join(str(i) for i in range(6)))
    assert len(read_table(path, max_rows=6)) == 6
    with pytest.raises(BatchError, match="mais de 5 linhas"):
        read_table(path, max_rows=5)
    with pytest.raises(BatchError, match="Não foi possível abrir"):
        read_table(str(tmp_path / "nao_existe.csv"))


# --------------------------------------------------------------------- colunas
def test_papeis_sugeridos_pelos_nomes_do_treino():
    roles = default_roles(["id", "idade", "area", "preco"], 2, ["area", "idade"], ["preco"], 1)
    assert roles == {"id": ROLE_IGNORE, "idade": ROLE_INPUT, "area": ROLE_INPUT, "preco": ROLE_HIDDEN}


def test_papeis_sugeridos_pela_posicao_quando_os_nomes_nao_batem():
    # Sobram exatamente as saídas esperadas: ficam ocultas.
    assert default_roles(["c1", "c2", "c3"], 2, ["x", "y"], ["z"], 1) == {
        "c1": ROLE_INPUT, "c2": ROLE_INPUT, "c3": ROLE_HIDDEN}
    # Sobra coluna demais (ou o modelo não guarda nomes): o resto é ignorado.
    assert default_roles(["c1", "c2", "c3", "c4"], 2, None, None, 1) == {
        "c1": ROLE_INPUT, "c2": ROLE_INPUT, "c3": ROLE_IGNORE, "c4": ROLE_IGNORE}


def test_plano_usa_a_ordem_do_treinamento():
    roles = {"idade": ROLE_INPUT, "area": ROLE_INPUT, "preco": ROLE_HIDDEN, "id": ROLE_IGNORE}
    plan = plan_columns(["id", "idade", "area", "preco"], roles, 2, ["area", "idade"], ["preco"])
    assert plan.input_cols == ["area", "idade"] and plan.reordered and plan.names_match
    assert plan.hidden_cols == ["preco"]


def test_plano_com_nomes_diferentes_ou_contagem_errada():
    roles = {"a": ROLE_INPUT, "b": ROLE_INPUT, "c": ROLE_INPUT}
    plan = plan_columns(["a", "b", "c"], roles, 3, ["x", "y", "z"], None)
    assert plan.input_cols == ["a", "b", "c"] and plan.names_match is False and not plan.reordered
    assert plan_columns(["a", "b", "c"], roles, 3, None, None).names_match is None
    with pytest.raises(BatchError, match="espera 2 entrada"):
        plan_columns(["a", "b", "c"], roles, 2, None, None)


def test_entradas_invalidas_apontam_linha_e_coluna(tmp_path):
    table = read_table(write(tmp_path, "a,b\n1,2\n3,abc\n"))
    assert parse_inputs(table, ["a"]) == [[1.0], [3.0]]
    with pytest.raises(BatchError, match="Linha 3 do arquivo, coluna “b”: “abc”"):
        parse_inputs(table, ["a", "b"])
    table = read_table(write(tmp_path, "a,b\n1,\n", "vazio.csv"))
    with pytest.raises(BatchError, match="“\\(vazio\\)” não é um número"):
        parse_inputs(table, ["a", "b"])


# ------------------------------------------------------------------- comparar
def test_quando_da_para_comparar():
    assert "Nenhuma coluna" in comparison_problem("regression", [], 1)
    assert "marque 2 coluna" in comparison_problem("regression", ["y"], 2)
    assert comparison_problem("regression", ["y1", "y2"], 2) is None
    assert "só a coluna do rótulo" in comparison_problem("classification", ["a", "b"], 3)
    assert comparison_problem("classification", ["especie"], 3) is None


def test_comparacao_de_regressao_mse_e_r2(tmp_path):
    table = read_table(write(tmp_path, "x,y\n0,1\n0,2\n0,3\n"))
    result = compare_regression(table, ["y"], [[1.5], [2.0], [2.5]])
    assert result["y_true"] == [[1.0], [2.0], [3.0]] and result["n"] == 3
    assert result["mse"] == pytest.approx((0.25 + 0 + 0.25) / 3)
    # Var(y) = 2/3 -> R² = 1 − MSE/Var = 1 − (1/6)/(2/3) = 0,75
    assert result["r2"] == pytest.approx(0.75)
    bad = read_table(write(tmp_path, "x,y\n0,1\n0,?\n", "ruim.csv"))
    with pytest.raises(BatchError, match="Linha 3 .* não dá para comparar"):
        compare_regression(bad, ["y"], [[1.0], [1.0]])


def test_comparacao_de_classificacao(tmp_path):
    table = read_table(write(tmp_path, "x,classe\n0,A\n0,B\n0,C\n0,A\n"))
    result = compare_classification(table, "classe", ["A", "A", "C", "A"], known=["A", "B"])
    assert result["hits"] == [True, False, True, True]
    assert result["correct"] == 3 and result["accuracy"] == 0.75
    assert result["unknown"] == 1  # "C" não apareceu no treino


# --------------------------------------------------------------------- salvar
def test_exporta_previsto_e_comparacao(tmp_path):
    table = read_table(write(tmp_path, "id,b,a,y\n7,2,1,3\n8,4,3,5\n"))
    roles = {"id": ROLE_IGNORE, "a": ROLE_INPUT, "b": ROLE_INPUT, "y": ROLE_HIDDEN}
    plan = plan_columns(table.columns, roles, 2, ["a", "b"], ["y"])
    outputs = [[2.5], [5.5]]
    header, rows = export_rows(table, plan, outputs, None, ["y"])
    assert header == ["a", "b", "previsto_y"]  # sem a coluna oculta e sem a ignorada
    assert rows == [["1", "2", "2.5"], ["3", "4", "5.5"]]
    cmp = compare_regression(table, plan.hidden_cols, outputs)
    header, rows = export_rows(table, plan, outputs, None, ["y"], cmp)
    assert header == ["a", "b", "previsto_y", "real_y", "erro_y"]
    assert rows[0] == ["1", "2", "2.5", "3.0", "-0.5"]
    out = str(tmp_path / "res.csv")
    write_csv(out, header, rows)
    with open(out, newline="", encoding="utf-8") as f:
        assert list(csv.reader(f))[0] == header


def test_exporta_classificacao(tmp_path):
    table = read_table(write(tmp_path, "a,classe\n1,A\n2,B\n"))
    plan = plan_columns(table.columns, {"a": ROLE_INPUT, "classe": ROLE_HIDDEN}, 1, ["a"], ["classe"])
    cmp = compare_classification(table, "classe", ["A", "A"])
    header, rows = export_rows(table, plan, [[0.9], [0.8]], ["A", "A"], ["Classe A"], cmp)
    assert header == ["a", "classe_prevista", "classe_real", "acertou"]
    assert rows == [["1", "A", "A", "sim"], ["2", "A", "B", "não"]]

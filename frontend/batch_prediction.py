"""Previsão de várias linhas a partir de um CSV (sem Flet).

O ``MLPService`` só prevê uma linha por vez (``predict(values)``); aqui ficam as partes
que a interface precisa em volta disso:

- ler o CSV e sugerir o papel de cada coluna (entrada, saída oculta, ignorar);
- montar as entradas na ordem em que a rede foi treinada;
- comparar o previsto com a saída real, quando o arquivo a tiver.

A coluna da saída real pode ficar no arquivo. Marcada como "Saída (oculta)", ela não é
enviada para a rede e só aparece se o usuário pedir a comparação. Assim não é preciso
apagar a coluna numa planilha antes de prever.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

from frontend.analysis import r2_from_mse
from frontend.parsing import format_number

ROLE_INPUT, ROLE_HIDDEN, ROLE_IGNORE = "in", "hidden", "skip"
MAX_ROWS = 50_000  # acima disso a tela fica lenta (uma chamada ao service por linha)


class BatchError(Exception):
    """Problema no arquivo ou nas colunas escolhidas, com mensagem pronta para a tela."""


@dataclass
class CsvTable:
    path: str
    columns: list[str]
    rows: list[list[str]]  # células como texto, na ordem de ``columns``
    lines: list[int] = field(default_factory=list)  # linha de cada registro no arquivo (cabeçalho = 1)

    def __len__(self) -> int:
        return len(self.rows)

    def column_values(self, name: str) -> list[str]:
        j = self.columns.index(name)
        return [row[j] for row in self.rows]


@dataclass
class ColumnPlan:
    input_cols: list[str]  # na ordem que a rede espera
    hidden_cols: list[str]  # saídas reais, na ordem das saídas da rede quando os nomes batem
    names_match: bool | None  # None: o modelo não guarda os nomes das colunas
    reordered: bool  # a ordem das entradas é diferente da ordem do arquivo


def read_table(path: str, *, max_rows: int = MAX_ROWS) -> CsvTable:
    """Lê o CSV inteiro como texto. Linhas em branco são puladas."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or not any(h.strip() for h in header):
                raise BatchError("O arquivo não tem cabeçalho (nomes das colunas na primeira linha).")
            columns = [h.strip() for h in header]
            repeated = sorted({c for c in columns if columns.count(c) > 1})
            if repeated:
                raise BatchError("Há colunas com o mesmo nome: " + ", ".join(repeated) + ".")
            rows: list[list[str]] = []
            lines: list[int] = []
            for row in reader:
                if not any(cell.strip() for cell in row):
                    continue
                if len(rows) >= max_rows:
                    raise BatchError(f"O arquivo tem mais de {format_number(max_rows)} linhas de dados. "
                                     "Divida-o em partes menores.")
                cells = [c.strip() for c in row[: len(columns)]]
                cells += [""] * (len(columns) - len(cells))
                rows.append(cells)
                lines.append(reader.line_num)
    except UnicodeDecodeError:
        raise BatchError("Não foi possível ler o arquivo como texto UTF-8.") from None
    except csv.Error as exc:
        raise BatchError(f"CSV malformado: {exc}") from None
    except OSError as exc:
        raise BatchError(f"Não foi possível abrir o arquivo: {exc.strerror or exc}") from None
    if not rows:
        raise BatchError("O arquivo não tem linhas de dados (só o cabeçalho).")
    return CsvTable(path=path, columns=columns, rows=rows, lines=lines)


def default_roles(
    columns: list[str],
    n_inputs: int,
    model_inputs: list[str] | None,
    model_outputs: list[str] | None,
    n_hidden_expected: int,
) -> dict[str, str]:
    """Papel sugerido para cada coluna.

    Se o arquivo tem as colunas de entrada do treino, usa os nomes: entradas do treino
    viram "Entrada", saídas do treino viram "Saída (oculta)" e o resto é ignorado.
    Senão, as ``n_inputs`` primeiras colunas são as entradas e, se sobrarem exatamente
    as colunas de saída esperadas, elas ficam ocultas.
    """
    if model_inputs and all(c in columns for c in model_inputs):
        outs = set(model_outputs or [])
        return {c: ROLE_INPUT if c in model_inputs else ROLE_HIDDEN if c in outs else ROLE_IGNORE
                for c in columns}
    rest = columns[n_inputs:]
    hidden = set(rest) if rest and len(rest) == n_hidden_expected else set()
    return {c: ROLE_INPUT if i < n_inputs else ROLE_HIDDEN if c in hidden else ROLE_IGNORE
            for i, c in enumerate(columns)}


def plan_columns(
    columns: list[str],
    roles: dict[str, str],
    n_inputs: int,
    model_inputs: list[str] | None,
    model_outputs: list[str] | None,
) -> ColumnPlan:
    """Entradas e saídas ocultas escolhidas, na ordem certa. BatchError se não fecha."""
    chosen = [c for c in columns if roles.get(c) == ROLE_INPUT]
    hidden = [c for c in columns if roles.get(c) == ROLE_HIDDEN]
    if len(chosen) != n_inputs:
        raise BatchError(f"A rede espera {n_inputs} entrada(s), mas {len(chosen)} coluna(s) estão marcadas "
                         "como Entrada.")
    names_match: bool | None = None
    inputs = chosen
    if model_inputs and len(model_inputs) == n_inputs:
        names_match = set(chosen) == set(model_inputs)
        if names_match:
            inputs = list(model_inputs)  # mesma ordem do treinamento
    if model_outputs and set(hidden) == set(model_outputs):
        hidden = list(model_outputs)
    return ColumnPlan(input_cols=inputs, hidden_cols=hidden, names_match=names_match,
                      reordered=inputs != chosen)


def parse_inputs(table: CsvTable, input_cols: list[str]) -> list[list[float]]:
    """Valores de entrada como números (mesma regra do backend: ``float``)."""
    idx = [table.columns.index(c) for c in input_cols]
    values: list[list[float]] = []
    for k, row in enumerate(table.rows):
        parsed = []
        for j, col in zip(idx, input_cols):
            text = row[j]
            try:
                v = float(text)
            except ValueError:
                v = math.nan
            if not math.isfinite(v):
                raise BatchError(f"Linha {table.lines[k]} do arquivo, coluna “{col}”: "
                                 f"“{text or '(vazio)'}” não é um número.")
            parsed.append(v)
        values.append(parsed)
    return values


def comparison_problem(task: str, hidden_cols: list[str], n_outputs: int) -> str | None:
    """Por que não dá para comparar com a saída real (None = dá)."""
    if not hidden_cols:
        return "Nenhuma coluna está marcada como “Saída (oculta)”."
    if task == "classification":
        if len(hidden_cols) != 1:
            return "Na classificação, marque só a coluna do rótulo (classe) como “Saída (oculta)”."
        return None
    if len(hidden_cols) != n_outputs:
        return (f"A rede tem {n_outputs} saída(s); marque {n_outputs} coluna(s) como “Saída (oculta)” "
                f"(marcadas: {len(hidden_cols)}).")
    return None


def compare_regression(table: CsvTable, hidden_cols: list[str], outputs: list[list[float]]) -> dict:
    """MSE e R² do previsto contra as colunas ocultas (mesmo MSE do backend)."""
    idx = [table.columns.index(c) for c in hidden_cols]
    y_true: list[list[float]] = []
    for k, row in enumerate(table.rows):
        values = []
        for j, col in zip(idx, hidden_cols):
            try:
                v = float(row[j])
            except ValueError:
                v = math.nan
            if not math.isfinite(v):
                raise BatchError(f"Linha {table.lines[k]} do arquivo, coluna “{col}”: "
                                 f"“{row[j] or '(vazio)'}” não é um número; não dá para comparar.")
            values.append(v)
        y_true.append(values)
    sq = [(p - t) ** 2 for pred, true in zip(outputs, y_true) for p, t in zip(pred, true)]
    mse = sum(sq) / len(sq) if sq else None
    return {"y_true": y_true, "mse": mse, "r2": r2_from_mse(mse, y_true), "n": len(y_true)}


def compare_classification(table: CsvTable, hidden_col: str, labels: list[str],
                           known: list[str] | None = None) -> dict:
    """Acertos do rótulo previsto contra a coluna oculta (comparação de texto)."""
    true = table.column_values(hidden_col)
    hits = [str(p).strip() == t for p, t in zip(labels, true)]
    unknown = sum(1 for t in true if known and t not in known)
    n = len(hits)
    return {"y_true": true, "hits": hits, "correct": sum(hits), "n": n,
            "accuracy": (sum(hits) / n) if n else None, "unknown": unknown}


def export_rows(
    table: CsvTable,
    plan: ColumnPlan,
    outputs: list[list[float]],
    labels: list[str] | None,
    output_names: list[str],
    comparison: dict | None = None,
) -> tuple[list[str], list[list[str]]]:
    """Cabeçalho e linhas do CSV de resultados: entradas, previsto e, se comparado, real e erro."""
    in_idx = [table.columns.index(c) for c in plan.input_cols]
    header = list(plan.input_cols)
    if labels is not None:
        header.append("classe_prevista")
    else:
        header += [f"previsto_{name}" for name in output_names]
    if comparison:
        if labels is not None:
            header += ["classe_real", "acertou"]
        else:
            for name, col in zip(output_names, plan.hidden_cols):
                header += [f"real_{col}", f"erro_{name}"]
    rows = []
    for k, row in enumerate(table.rows):
        line = [row[j] for j in in_idx]
        if labels is not None:
            line.append(str(labels[k]))
        else:
            line += [repr(float(v)) for v in outputs[k]]
        if comparison:
            if labels is not None:
                line += [comparison["y_true"][k], "sim" if comparison["hits"][k] else "não"]
            else:
                for p, t in zip(outputs[k], comparison["y_true"][k]):
                    line += [repr(t), repr(float(p) - t)]
        rows.append(line)
    return header, rows


def write_csv(path: str, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

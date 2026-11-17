"""Análises feitas só na interface, sobre os dados que o backend devolve (sem Flet).

Nada aqui treina ou altera a rede. São leituras do histórico e das métricas,
usadas para orientar o usuário. Os critérios seguem a bibliografia da disciplina:

- parada antecipada: a época de menor MSE de validação (Haykin, *Redes Neurais*, Fig. 4.20;
  Ivan Nunes da Silva, *Redes Neurais Artificiais*, Fig. 5.41);
- cerca de 10 amostras de treinamento por parâmetro livre (Haykin, seção 4.12);
- nº de neurônios ocultos pela regra de Kolmogorov, 2n + 1 (Ivan Nunes, seção 5.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from frontend.parsing import format_number

Level = Literal["info", "success", "warning", "error"]


@dataclass(frozen=True)
class Insight:
    level: Level
    title: str
    message: str


# --------------------------------------------------------------------------- arquitetura


def count_parameters(n_inputs: int, layer_sizes: Sequence[int]) -> int:
    """Pesos + vieses de uma MLP totalmente conectada."""
    total, fan_in = 0, n_inputs
    for size in layer_sizes:
        total += (fan_in + 1) * size
        fan_in = size
    return total


def suggest_hidden_neurons(n_inputs: int) -> int:
    """Regra de Kolmogorov: 2n + 1 neurônios na camada oculta."""
    return 2 * max(1, n_inputs) + 1


def suggest_output_neurons(task: str, n_output_columns: int, n_classes: int | None = None) -> int:
    if task == "classification" and n_classes:
        return 1 if n_classes == 2 else n_classes
    return max(1, n_output_columns)


def output_activation_warning(task: str, activation: str) -> str | None:
    if task == "regression":
        if activation == "sigmoid":
            return "Sigmoid limita a saída ao intervalo (0, 1). Para regressão, prefira Linear."
        if activation == "tanh":
            return "Tanh limita a saída ao intervalo (−1, 1). Para regressão, prefira Linear."
        if activation == "relu":
            return "ReLU não produz valores negativos. Para regressão, prefira Linear."
    elif task == "classification" and activation == "relu":
        return "ReLU na saída não fica limitada a [0, 1]. Para classificação, prefira Sigmoid."
    return None


def samples_per_parameter(n_train: int | None, n_params: int | None) -> Insight | None:
    if not n_train or not n_params:
        return None
    ratio = n_train / n_params
    text = (
        f"{n_train} amostras de treinamento para {n_params} parâmetros "
        f"({format_number(ratio, 2)} amostras por parâmetro)."
    )
    if ratio < 10:
        return Insight(
            "warning",
            "Poucos dados para o tamanho da rede",
            text + " Haykin recomenda cerca de 10 por parâmetro. Considere uma rede menor "
            "ou acompanhe de perto o MSE de validação.",
        )
    return Insight("info", "Proporção dados/parâmetros adequada", text)


# --------------------------------------------------------------------------- treinamento


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def best_epoch(history: Sequence[dict]) -> dict | None:
    """Época de menor MSE de validação, ou seja, o ponto de parada antecipada."""
    valid = [h for h in history if _finite(h.get("val_mse"))]
    if not valid:
        return None
    return min(valid, key=lambda h: h["val_mse"])


def analyze_history(history: Sequence[dict], *, stopped_early: bool = False) -> list[Insight]:
    """Interpreta a curva de aprendizagem (treinamento × validação)."""
    if not history:
        return []
    insights: list[Insight] = []
    last = history[-1]
    first = history[0]

    if not _finite(last.get("train_mse")) or not _finite(last.get("val_mse")):
        return [
            Insight(
                "error",
                "O treinamento divergiu",
                "O erro virou infinito ou NaN. Reduza a taxa de aprendizagem (η) "
                "ou normalize as entradas.",
            )
        ]

    if len(history) >= 3 and last["train_mse"] > first["train_mse"] * 1.5:
        insights.append(
            Insight(
                "error",
                "O erro de treinamento aumentou",
                f"O MSE de treinamento foi de {format_number(first['train_mse'])} para "
                f"{format_number(last['train_mse'])}. A taxa de aprendizagem (η) "
                "provavelmente está alta demais.",
            )
        )

    best = best_epoch(history)
    if best is not None:
        epochs_since_best = last["epoch"] - best["epoch"]
        patience = max(5, len(history) // 10)
        rising = last["val_mse"] > best["val_mse"] * 1.05
        train_falling = last["train_mse"] <= history[max(0, len(history) - patience - 1)]["train_mse"]
        if epochs_since_best >= patience and rising and train_falling:
            insights.append(
                Insight(
                    "warning",
                    "Possível sobreajuste (overfitting)",
                    f"O MSE de validação está subindo desde a época {best['epoch']} "
                    f"({format_number(best['val_mse'])}), enquanto o de treinamento continua caindo. "
                    "A rede pode estar memorizando o ruído dos dados. A época "
                    f"{best['epoch']} seria o ponto de parada antecipada.",
                )
            )
        elif best is last:
            insights.append(
                Insight(
                    "success",
                    "Validação ainda melhorando",
                    f"O menor MSE de validação ({format_number(best['val_mse'])}) foi obtido "
                    "na última época. Mais épocas podem reduzir o erro.",
                )
            )
        else:
            insights.append(
                Insight(
                    "info",
                    "Menor erro de validação",
                    f"Época {best['epoch']}: MSE de validação = {format_number(best['val_mse'])}.",
                )
            )

    if len(history) >= 20:
        window = max(5, len(history) // 5)
        before = history[-window - 1]["train_mse"]
        if before > 0 and abs(before - last["train_mse"]) / before < 0.01:
            insights.append(
                Insight(
                    "info",
                    "Curva estabilizada",
                    f"O MSE de treinamento variou menos de 1% nas últimas {window} épocas. "
                    "Mais épocas trarão pouco ganho com esta configuração.",
                )
            )

    if stopped_early:
        insights.append(
            Insight("info", "Treinamento interrompido", f"Parado na época {last['epoch']}.")
        )
    return insights


def downsample_indices(n: int, max_points: int, keep: Sequence[int] = ()) -> list[int]:
    """Índices espaçados para desenhar no máximo ``max_points`` pontos.

    Sempre mantém o primeiro, o último e os índices de ``keep``.
    """
    if n <= max_points:
        return list(range(n))
    step = (n - 1) / (max_points - 1)
    chosen = {round(i * step) for i in range(max_points)}
    chosen.update(k for k in keep if 0 <= k < n)
    chosen.update((0, n - 1))
    return sorted(chosen)


# --------------------------------------------------------------------------- avaliação


def flatten(values: Any) -> list[float]:
    """Achata a saída do backend (lista simples ou lista de listas) em floats."""
    flat: list[float] = []
    for v in values or []:
        if isinstance(v, (list, tuple)):
            flat.extend(float(x) for x in v)
        else:
            flat.append(float(v))
    return flat


def column(values: Any, index: int) -> list[float]:
    """Seleciona uma das saídas (coluna ``index``) de y_true/y_pred."""
    out: list[float] = []
    for v in values or []:
        if isinstance(v, (list, tuple)):
            out.append(float(v[index]))
        else:
            out.append(float(v))
    return out


def r2_from_mse(mse: float | None, y_true: Sequence[Sequence[float]] | None) -> float | None:
    """Coeficiente de determinação a partir do MSE e das saídas esperadas.

    R² = 1 − SQ_res / SQ_tot = 1 − MSE / Var(y), com Var = variância populacional
    (média dos desvios ao quadrado). Com várias saídas, usa a variância média das
    saídas, que é coerente com o MSE do backend (média sobre amostras e saídas).
    None quando não dá para calcular (sem dados ou y constante).
    """
    if mse is None or not _finite(mse) or not y_true:
        return None
    rows = [list(r) if isinstance(r, (list, tuple)) else [r] for r in y_true]
    n, k = len(rows), len(rows[0])
    if k == 0 or any(len(r) != k for r in rows):
        return None
    variances = []
    for j in range(k):
        col = [float(r[j]) for r in rows]
        mean = sum(col) / n
        variances.append(sum((v - mean) ** 2 for v in col) / n)
    var = sum(variances) / k
    if var <= 0:
        return None
    return 1 - mse / var


def accuracy_from_confusion(matrix: Sequence[Sequence[int]]) -> float | None:
    total = sum(sum(row) for row in matrix)
    if not total:
        return None
    return sum(matrix[i][i] for i in range(min(len(matrix), len(matrix[0])))) / total

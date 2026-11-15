import math

import pytest

from frontend.analysis import (
    accuracy_from_confusion,
    analyze_history,
    best_epoch,
    column,
    count_parameters,
    downsample_indices,
    flatten,
    output_activation_warning,
    regression_metrics,
    rmse,
    samples_per_parameter,
    suggest_hidden_neurons,
    suggest_output_neurons,
)


def hist(train, val):
    return [{"epoch": i + 1, "train_mse": t, "val_mse": v} for i, (t, v) in enumerate(zip(train, val))]


def test_count_parameters_rede_do_slide():
    # 2 entradas -> 3 ocultos -> 1 saída: (2+1)*3 + (3+1)*1 = 13
    assert count_parameters(2, [3, 1]) == 13
    assert count_parameters(4, [1]) == 5  # ADALINE: 4 pesos + 1 viés


def test_sugestoes():
    assert suggest_hidden_neurons(3) == 7  # Kolmogorov 2n+1
    assert suggest_output_neurons("regression", 2) == 2
    assert suggest_output_neurons("classification", 1, n_classes=2) == 1
    assert suggest_output_neurons("classification", 1, n_classes=3) == 3


def test_aviso_de_ativacao_na_saida():
    assert output_activation_warning("regression", "linear") is None
    assert "(0, 1)" in output_activation_warning("regression", "sigmoid")
    assert "negativos" in output_activation_warning("regression", "relu")
    assert output_activation_warning("classification", "sigmoid") is None


def test_best_epoch_ignora_nan():
    h = hist([1, 0.5, 0.4], [1.0, float("nan"), 0.6])
    assert best_epoch(h)["epoch"] == 3
    assert best_epoch([]) is None


def test_detecta_sobreajuste():
    n = 60
    train = [1 / (i + 1) for i in range(n)]
    val = [1 / (i + 1) + (0.002 * (i - 30) if i > 30 else 0) for i in range(n)]
    insights = analyze_history(hist(train, val))
    titles = [i.title for i in insights]
    assert any("sobreajuste" in t for t in titles)
    warning = next(i for i in insights if "sobreajuste" in i.title)
    assert warning.level == "warning"
    assert "parada antecipada" in warning.message


def test_validacao_ainda_melhorando():
    train = [1 / (i + 1) for i in range(30)]
    insights = analyze_history(hist(train, [t * 1.1 for t in train]))
    assert insights[0].level == "success"


def test_detecta_divergencia():
    insights = analyze_history(hist([1, 10, float("inf")], [1, 10, float("inf")]))
    assert insights[0].level == "error" and "divergiu" in insights[0].title
    insights = analyze_history(hist([1, 3, 9], [1, 3, 9]))
    assert any("aumentou" in i.title for i in insights)


def test_downsample_mantem_extremos_e_indice_pedido():
    idx = downsample_indices(10_000, 400, keep=[1234])
    assert idx[0] == 0 and idx[-1] == 9_999 and 1234 in idx
    assert len(idx) <= 402
    assert downsample_indices(5, 400) == [0, 1, 2, 3, 4]


def test_regression_metrics():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert m["mse"] == pytest.approx(4 / 3)
    assert m["rmse"] == pytest.approx(math.sqrt(4 / 3))
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["r2"] == pytest.approx(1 - 4 / 2)
    assert m["mean_rel_error_pct"] == pytest.approx(100 * (2 / 3) / 3)
    assert regression_metrics([0.0, 1.0], [0.0, 1.0])["mean_rel_error_pct"] is None
    assert regression_metrics([], []) == {}


def test_column_e_flatten_aceitam_listas_de_listas():
    y = [[1, 10], [2, 20]]
    assert column(y, 1) == [10.0, 20.0]
    assert flatten(y) == [1.0, 10.0, 2.0, 20.0]
    assert column([1, 2], 0) == [1.0, 2.0]


def test_rmse_e_acuracia():
    assert rmse(4.0) == 2.0
    assert rmse(None) is None and rmse(float("nan")) is None
    assert accuracy_from_confusion([[5, 1], [0, 4]]) == pytest.approx(0.9)


def test_samples_per_parameter():
    assert samples_per_parameter(50, 13).level == "warning"
    assert samples_per_parameter(500, 13).level == "info"
    assert samples_per_parameter(None, 13) is None

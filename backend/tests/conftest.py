import csv
from pathlib import Path
import numpy as np
import pytest
from backend.network import NeuralNetwork

SEED = 42


@pytest.fixture
def rede_xor_slide() -> NeuralNetwork:
    """Rede 2-3-1 sigmoid/sigmoid com os pesos exatos dos slides da disciplina (bias = 0)."""
    net = NeuralNetwork.from_config(
        2,
        [
            {"neurons": 3, "activation": "sigmoid"},
            {"neurons": 1, "activation": "sigmoid"},
        ],
        seed=SEED,
    )
    oculta, saida = net.layers
    oculta.set_parameters(
        weights=[
            [-0.420, 0.361],
            [-0.730, -0.617],
            [-0.941, -0.482],
        ],
        biases=[0.0, 0.0, 0.0],
    )
    saida.set_parameters(weights=[[-0.015, -0.902, 0.136]], biases=[0.0])
    return net


@pytest.fixture
def dados_xor() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)
    return X, Y


@pytest.fixture
def csv_regressao(tmp_path: Path) -> Path:
    """Gera CSV sintético y = 2a - b + 1 sem ruído (60 amostras)."""
    rng = np.random.default_rng(SEED)
    a = rng.uniform(-5.0, 5.0, size=60)
    b = rng.uniform(-5.0, 5.0, size=60)
    y = 2.0 * a - b + 1.0

    file_path = tmp_path / "reg.csv"
    with file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "y"])
        for ai, bi, yi in zip(a, b, y):
            writer.writerow([f"{ai}", f"{bi}", f"{yi}"])
    return file_path


@pytest.fixture
def csv_classificacao(tmp_path: Path) -> Path:
    """Gera CSV sintético com 3 classes (A, B, C) e 60 amostras."""
    rng = np.random.default_rng(SEED)
    classes = ["A", "B", "C"]
    centers = {"A": (0.0, 0.0), "B": (3.0, 3.0), "C": (-3.0, 3.0)}

    file_path = tmp_path / "cls.csv"
    with file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["f1", "f2", "label"])
        for i in range(60):
            lbl = classes[i % 3]
            cx, cy = centers[lbl]
            f1 = cx + rng.normal(0.0, 0.2)
            f2 = cy + rng.normal(0.0, 0.2)
            writer.writerow([f"{f1:.6f}", f"{f2:.6f}", lbl])
    return file_path
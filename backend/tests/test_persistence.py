import json
import numpy as np
import pytest
from backend.dataset import Dataset, DataSplit
from backend.errors import PersistenceError
from backend.network import NeuralNetwork
from backend.normalization import apply_normalization
from backend.persistence import load_model, save_model
from backend.tests.helpers import params_iguais, snapshot_params


def _rede_exemplo() -> NeuralNetwork:
    return NeuralNetwork.from_config(
        3,
        [
            {"neurons": 5, "activation": "tanh"},
            {"neurons": 4, "activation": "relu"},
            {"neurons": 2, "activation": "linear"},
        ],
        seed=11,
    )


def test_roundtrip_gera_exatamente_as_mesmas_saidas(tmp_path):
    net = _rede_exemplo()
    path = tmp_path / "m.json"
    save_model(path, net, normalizer=None, task="regression", meta={"input_cols": ["a", "b", "c"]})
    modelo = load_model(path)

    rng = np.random.default_rng(0)
    for x in rng.normal(size=(10, 3)):
        assert np.array_equal(modelo.network.forward(x), net.forward(x))


def test_arquitetura_restaurada(tmp_path):
    net = _rede_exemplo()
    path = tmp_path / "m.json"
    save_model(path, net, normalizer=None)
    modelo = load_model(path)

    assert modelo.network.n_inputs == net.n_inputs
    assert [len(l.neurons) for l in modelo.network.layers] == [len(l.neurons) for l in net.layers]
    assert [l.activation_name for l in modelo.network.layers] == [l.activation_name for l in net.layers]


def test_pesos_e_vieses_identicos(tmp_path):
    net = _rede_exemplo()
    path = tmp_path / "m.json"
    save_model(path, net, normalizer=None)
    modelo = load_model(path)
    assert params_iguais(snapshot_params(net), snapshot_params(modelo.network))


def test_normalizacao_salva_e_restaurada(tmp_path):
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    Y = np.array([[10.0, 20.0], [30.0, 40.0]])
    ds = Dataset(X, Y, ["a", "b", "c"], ["y1", "y2"])
    _, norm, y_norm = apply_normalization(DataSplit(ds, ds, ds), "zscore", normalize_y=True)

    path = tmp_path / "m.json"
    save_model(path, _rede_exemplo(), normalizer=norm, y_normalizer=y_norm)
    modelo = load_model(path)

    assert modelo.normalizer.kind == "zscore"
    assert np.allclose(modelo.normalizer.params["mean"], norm.params["mean"])
    assert np.allclose(modelo.normalizer.transform(X), norm.transform(X))
    assert modelo.y_normalizer is not None
    assert modelo.y_normalizer.kind == "zscore"
    assert np.allclose(modelo.y_normalizer.inverse_transform(Y), y_norm.inverse_transform(Y))


def test_sem_normalizacao_grava_null(tmp_path):
    path = tmp_path / "m.json"
    save_model(path, _rede_exemplo(), normalizer=None)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["normalization"] is None
    assert d["y_normalization"] is None
    loaded = load_model(path)
    assert loaded.normalizer is None
    assert loaded.y_normalizer is None


def test_json_tem_os_campos_exigidos(tmp_path):
    path = tmp_path / "m.json"
    save_model(path, _rede_exemplo(), normalizer=None)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert {"n_inputs", "layers", "normalization"} <= set(d)
    camada = d["layers"][0]
    assert {"n_neurons", "activation", "neurons"} <= set(camada)
    assert {"weights", "bias"} <= set(camada["neurons"][0])


def test_task_e_meta_preservados(tmp_path):
    path = tmp_path / "m.json"
    save_model(path, _rede_exemplo(), normalizer=None, task="regression", meta={"input_cols": ["a", "b", "c"]})
    modelo = load_model(path)
    assert modelo.task == "regression"
    assert modelo.meta["input_cols"] == ["a", "b", "c"]


def test_arquivo_inexistente():
    with pytest.raises(PersistenceError):
        load_model("nao_existe_12345.json")


def test_json_invalido(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{{{", encoding="utf-8")
    with pytest.raises(PersistenceError):
        load_model(path)


def test_campo_obrigatorio_ausente(tmp_path):
    path = tmp_path / "missing.json"
    path.write_text(json.dumps({"format_version": 1, "n_inputs": 2}), encoding="utf-8")
    with pytest.raises(PersistenceError):
        load_model(path)


def test_versao_nao_suportada(tmp_path):
    path = tmp_path / "ver.json"
    path.write_text(
        json.dumps({"format_version": 99, "n_inputs": 2, "layers": [], "normalization": None}),
        encoding="utf-8",
    )
    with pytest.raises(PersistenceError):
        load_model(path)


def test_dimensoes_inconsistentes(tmp_path):
    path = tmp_path / "dim.json"
    payload = {
        "format_version": 1,
        "n_inputs": 2,
        "normalization": None,
        "layers": [
            {
                "n_neurons": 1,
                "activation": "linear",
                "neurons": [{"weights": [0.1, 0.2, 0.3], "bias": 0.0}],  # esperava 2 pesos
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PersistenceError):
        load_model(path)
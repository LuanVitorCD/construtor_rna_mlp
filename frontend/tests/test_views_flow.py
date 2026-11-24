"""Fluxo pelas telas (sem abrir janela), usando o DemoService.

Os handlers são chamados diretamente, como se o usuário clicasse.
Modo "Criar novo modelo": Dados → Arquitetura → Treinamento → Avaliação → Previsão.
"""

import asyncio

import flet as ft
import pytest

from frontend.tests.ui_harness import build_app, build_use_app, table_rows
from frontend.views.architecture import IO_AUTO, IO_CUSTOM


@pytest.fixture
def csv_regressao(tmp_path):
    path = tmp_path / "dados.csv"
    lines = ["x1,x2,x3,y"] + [f"{i},{i % 7},{i * 0.5},{2 * i + 1}" for i in range(60)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def csv_classes(tmp_path):
    path = tmp_path / "flores.csv"
    rows = ["a,b,classe"] + [f"{i},{i % 5},{['x', 'y', 'z'][i % 3]}" for i in range(45)]
    path.write_text("\n".join(rows), encoding="utf-8")
    return str(path)


def test_fluxo_completo_de_regressao_csv_primeiro(csv_regressao):
    async def flow():
        state, page, (data, arch, train, ev, pred), _toasts, _gw = build_app()
        # Antes de tudo: criar rede (modo Automático), treinar e prever ficam bloqueados.
        assert arch.create_btn.disabled and train.start_btn.disabled and pred.predict_btn.disabled
        assert arch.io_message.action and arch.io_message.action[0] == "Ir para a aba Dados"

        await data.open_csv(csv_regressao)
        assert state.columns == ["x1", "x2", "x3", "y"]
        assert state.input_cols == ["x1", "x2", "x3"] and state.output_cols == ["y"]
        assert data.selection_message.level == "success"
        # Arquitetura já sabe as entradas/saídas (modo Automático) sem a rede existir.
        assert state.suggested_io() == (3, 1)
        assert (arch.n_inputs.value, arch.n_outputs.value) == ("3", "1") and arch.n_inputs.read_only
        assert not arch.create_btn.disabled

        data.norm_group.value = "zscore"
        data.train_pct.value, data.val_pct.value, data.test_pct.value = "60", "20", "20"
        await arch._on_create(None)  # cria a rede E aplica os dados (carregar, dividir, normalizar)
        assert state.has_network and state.architecture_text() == "3–3–1"
        assert state.data_ready and state.dataset["rows"] == 60 and state.normalization == "zscore"
        assert state.split_size("train") == 36 and state.split_size("test") == 12
        assert arch.message.title == "Rede criada e dados aplicados"
        assert table_rows(data)[0] == ["0", "0", "0", "1"]  # prévia {"x", "y"} vira células

        assert not train.start_btn.disabled
        train.epochs.value, train.lr.value = "25", "0,05"
        await train._on_start(None)
        assert state.training and train.stop_btn.disabled is False
        await page.tasks[-1]  # espera o "pump" terminar
        assert not state.training and len(state.history) == 25 and state.trained_epochs == 25
        assert train.chart._semantics.visible and not train.chart.placeholder.visible
        assert train.insights.controls  # leitura da curva preenchida

        await ev._on_evaluate_all(None)
        assert set(state.evaluations) == {"train", "val", "test"}
        assert ev.comparison_section.visible  # o evaluate devolve y_true/y_pred: gráfico desejado × estimado
        assert not ev.confusion_section.visible  # regressão: sem matriz
        tile_labels = [tile.content.label for tile in ev.metrics.controls]
        summary_cols = [c.label.value for c in ev.summary_holder.content.columns]
        # Só métricas vistas em aula: nada de RMSE, MAE ou erro relativo
        assert [t.split(":")[0] for t in tile_labels] == ["MSE", "Amostras", "R²"]
        assert summary_cols == ["Subconjunto", "MSE", "R²"]  # regressão: R² no lugar de "Acertos"
        for subset, result in state.evaluations.items():  # R² = 1 − MSE / Var(y) do subconjunto
            assert result["n_samples"] == state.split_size(subset) == len(result["y_true"])
            ys = [row[0] for row in result["y_true"]]
            mean = sum(ys) / len(ys)
            var = sum((y - mean) ** 2 for y in ys) / len(ys)
            assert result["r2"] == pytest.approx(1 - result["mse"] / var)

        pred.paste.value = "10; 3; 5"
        pred._on_fill(None)
        assert [f.value for f in pred.fields] == ["10", "3", "5"]
        assert [f.label for f in pred.fields] == ["x1", "x2", "x3"]
        await pred._on_predict(None)
        assert len(pred._history) == 1 and not pred.message.visible

    asyncio.run(flow())


def test_personalizado_diferente_do_csv_avisa_e_nao_aplica_dados(csv_regressao):
    async def flow():
        state, _page, (data, arch, train, _ev, _pred), _toasts, _gw = build_app()
        await data.open_csv(csv_regressao)
        arch.io_mode.selected = [IO_CUSTOM]
        arch._on_io_mode_change(None)
        assert not arch.n_inputs.read_only
        arch.n_inputs.value = "2"
        arch._on_draft_change(object())  # digitação
        assert arch.io_message.level == "warning"  # diferente do CSV
        await arch._on_create(None)
        assert state.has_network and not state.data_ready
        assert arch.message.level == "error" and "não foram aplicados" in arch.message.title
        assert data.apply_message.title == "Problema nos dados"
        assert train.start_btn.disabled

        arch.io_mode.selected = [IO_AUTO]  # volta ao automático e recria
        arch._on_io_mode_change(None)
        await arch._on_create(None)
        assert state.data_ready and not train.start_btn.disabled

    asyncio.run(flow())


def test_personalizado_sem_csv_cria_a_rede_sem_dados():
    async def flow():
        state, _page, (data, arch, train, _ev, pred), _toasts, _gw = build_app()
        arch.io_mode.selected = [IO_CUSTOM]
        arch._on_io_mode_change(None)
        assert not arch.create_btn.disabled and arch.io_message.level == "info"
        arch.n_inputs.value, arch.n_outputs.value = "2", "1"
        await arch._on_create(None)
        assert state.has_network and not state.has_dataset and arch.message.title == "Rede criada"
        assert train.start_btn.disabled and not pred.predict_btn.disabled  # já dá para prever
        assert data.apply_btn.visible and data.apply_btn.disabled  # falta o CSV

    asyncio.run(flow())


def test_alterar_dados_depois_de_aplicar_exige_reaplicar(csv_regressao):
    async def flow():
        state, _page, (data, arch, train, ev, _pred), _toasts, _gw = build_app()
        await data.open_csv(csv_regressao)
        await arch._on_create(None)
        assert state.data_ready and not train.start_btn.disabled

        data.train_pct.value = "60"
        data._on_pct_change(object())
        data.val_pct.value = "25"
        data._on_pct_change(object())
        assert state.data_dirty and not state.data_ready
        assert train.start_btn.disabled and ev.eval_all_btn.disabled
        assert data.apply_message.level == "warning" and not data.apply_btn.disabled

        await data._on_apply(None)
        assert state.data_ready and state.split_size("val") == 15 and not train.start_btn.disabled

    asyncio.run(flow())


def test_fluxo_de_classificacao_conta_classes_antes_da_rede(csv_classes):
    async def flow():
        state, page, (data, arch, train, ev, pred), _toasts, _gw = build_app()
        data.task.selected = ["classification"]
        data._on_task_change(None)
        assert state.task_choice == "classification"
        assert arch.out_activation.value == "sigmoid"  # padrão da classificação
        await data.open_csv(csv_classes)
        assert state.class_labels_preview == ["x", "y", "z"]  # lido do CSV, antes da rede
        assert state.suggested_io() == (2, 3)  # 3 classes -> one-hot
        assert (arch.n_inputs.value, arch.n_outputs.value) == ("2", "3")
        await arch._on_create(None)
        assert state.data_ready and state.class_labels == ["x", "y", "z"]
        assert table_rows(data)[0][-1] == "x"  # y one-hot decodificado para o rótulo
        train.epochs.value = "5"
        await train._on_start(None)
        await page.tasks[-1]
        await ev._on_evaluate_all(None)
        assert ev.confusion_section.visible and not ev.comparison_section.visible
        assert state.evaluations["test"].get("r2") is None  # R² só na regressão
        assert [c.label.value for c in ev.summary_holder.content.columns] == ["Subconjunto", "MSE", "Acertos"]
        # Matriz: acertos (diagonal) em verde, erros em vermelho, zeros sem cor.
        grid = ev.confusion_holder.controls[0].controls[0].controls  # linhas da grade
        matrix = state.evaluations["test"]["confusion_matrix"]
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                bg = grid[i + 1].controls[j + 1].bgcolor
                if value == 0:
                    assert bg is None
                else:
                    assert ("tertiaryfixeddim" if i == j else "error") in str(bg)
        for f, v in zip(pred.fields, ("1", "2")):
            f.value = v
        await pred._on_predict(None)
        assert pred._history[0][1]["label"] in {"x", "y", "z"}

    asyncio.run(flow())


def test_classificacao_com_rotulo_continuo_e_bloqueada(csv_regressao):
    async def flow():
        state, _page, (data, arch, _train, _ev, _pred), _toasts, _gw = build_app()
        data.task.selected = ["classification"]
        data._on_task_change(None)
        await data.open_csv(csv_regressao)  # "y" tem 60 valores: ainda vale como rótulo (< 1000)
        assert state.suggested_io() == (3, 60)
        # ...mas são 60 "classes" numéricas: avisa e oferece voltar para Regressão (sem bloquear)
        assert data.selection_message.level == "warning"
        assert data.selection_message.action[0] == "Mudar para Regressão"
        data._on_role_change("x3", ["out"])  # duas colunas de saída: não pode na classificação
        assert data.selection_message.level == "warning" and state.suggested_io() is None
        assert arch.create_btn.disabled

    asyncio.run(flow())


def test_trocar_tarefa_depois_de_criar_pede_para_recriar(csv_regressao):
    async def flow():
        state, _page, (data, arch, train, _ev, _pred), _toasts, _gw = build_app()
        await data.open_csv(csv_regressao)
        await arch._on_create(None)
        data.task.selected = ["classification"]
        data._on_task_change(None)
        assert data.task_message.level == "warning" and data.apply_btn.disabled
        assert state.data_dirty and train.start_btn.disabled

    asyncio.run(flow())


def test_grafico_desejado_estimado_com_duas_saidas(tmp_path):
    """Com várias saídas, o gráfico desejado × estimado mostra uma saída por vez (seletor)."""
    async def flow():
        path = tmp_path / "duas.csv"
        path.write_text("x1,x2,y1,y2\n" + "\n".join(f"{i},{i % 4},{2 * i},{i % 9}" for i in range(50)),
                        encoding="utf-8")
        _state, _page, views, _toasts, _gateway = build_app()
        data, arch, _train, ev, _pred = views
        await data.open_csv(str(path))
        data._on_role_change("y1", ["out"])
        await arch._on_create(None)
        await ev._on_evaluate_all(None)
        assert ev.comparison_section.visible and ev.output_select.visible
        assert [o.text for o in ev.output_select.options] == ["y1", "y2"]
        ev.output_select.value = "1"
        ev._on_output_change(None)
        assert "y2" in ev.comparison._semantics.label

    asyncio.run(flow())


def test_erros_aparecem_na_tela(csv_regressao):
    async def flow():
        state, _page, (data, arch, _train, _ev, pred), _toasts, _gw = build_app()
        await data.open_csv(csv_regressao)
        # formulário inválido: nada é enviado ao backend
        arch.hidden_rows[0].neurons.value = "abc"
        await arch._on_create(None)
        assert not state.has_network and arch.hidden_rows[0].neurons.error
        assert arch.message.level == "error"

        arch.hidden_rows[0].neurons.value = "4"
        data.train_pct.value = "90"  # soma 120%
        data._on_pct_change(object())
        await arch._on_create(None)
        assert state.has_network and not state.has_split  # rede criada, divisão recusada
        assert data.apply_message.title == "Divisão inválida"

        pred.fields[0].value = "x"
        await pred._on_predict(None)
        assert pred.fields[0].error and pred.message.visible

    asyncio.run(flow())


def test_recriar_rede_reseta_etapas(csv_regressao):
    async def flow():
        state, page, (data, arch, train, ev, _pred), _toasts, _gw = build_app()
        await data.open_csv(csv_regressao)
        await arch._on_create(None)
        train.epochs.value = "3"
        await train._on_start(None)
        await page.tasks[-1]
        assert state.is_trained
        await arch._on_create(None)  # confirm() devolve True no teste; dados reaplicados
        assert not state.is_trained and state.history == [] and state.data_ready
        assert ev.current is None

    asyncio.run(flow())


def test_parar_treinamento(csv_regressao):
    async def flow():
        state, page, (data, arch, train, _ev, _pred), _toasts, _gw = build_app()
        await data.open_csv(csv_regressao)
        await arch._on_create(None)
        train.epochs.value = "2000"
        await train._on_start(None)
        await asyncio.sleep(0.3)
        train._on_stop(None)
        await page.tasks[-1]
        assert not state.training and 0 < len(state.history) < 2000
        assert train.message.title == "Treinamento interrompido"

    asyncio.run(flow())


def _history(train_mse, val_mse):
    return [{"epoch": i + 1, "train_mse": t, "val_mse": v} for i, (t, v) in enumerate(zip(train_mse, val_mse))]


def test_sobreajuste_aparece_em_vermelho_logo_abaixo_do_botao():
    # A "Leitura da curva" fica abaixo da dobra; o aviso precisa aparecer na mensagem final.
    state, _page, (_data, _arch, train, _ev, _pred), _toasts, _gw = build_app()
    n = 60
    train_mse = [1 / (i + 1) for i in range(n)]
    state.history = _history(train_mse, [t + (0.002 * (i - 30) if i > 30 else 0) for i, t in enumerate(train_mse)])
    train._finish(forced=False)
    assert train.message.visible and train.message.level == "error"
    assert train.message.title == "Possível sobreajuste (overfitting)"
    text = train.message._holder.content.content.controls[1].controls[1].value
    assert text.startswith("Treinamento concluído após 60 época(s). Melhor época: 31 ")
    assert "treine com 31 épocas" in text
    boxes = [c for c in train.insights.controls if c.bgcolor == ft.Colors.ERROR_CONTAINER]
    assert len(boxes) == 1  # o cartão da leitura da curva também fica vermelho

    # Sem problema na curva, continua a mensagem verde de sempre.
    state.history = _history(train_mse, [t * 1.1 for t in train_mse])
    train._finish(forced=False)
    assert train.message.level == "success" and train.message.title == "Treinamento concluído"


def test_usar_modelo_como_base_preenche_o_formulario(csv_classes, tmp_path):
    async def flow():
        # 1) cria e salva um modelo de classificação 2-5-3 com Min-Max
        state, _page, (data, arch, _train, _ev, _pred), _toasts, gateway = build_app()
        data.task.selected = ["classification"]
        data._on_task_change(None)
        await data.open_csv(csv_classes)
        arch.hidden_rows[0].neurons.value = "5"
        data.norm_group.value = "minmax"
        await arch._on_create(None)
        model = str(tmp_path / "base.json")
        await gateway.call("save", model)

        # 2) nova sessão: usa como base (só a configuração)
        from frontend.model_files import read_model

        state2, _p2, (data2, arch2, _t2, _e2, pred2), _toasts2, gateway2 = build_app()
        await data2.open_csv(csv_classes)
        info = await read_model(gateway2, model)
        state2.forget_network()
        data2.apply_template(info)
        arch2.apply_template(info)
        state2.notify()
        assert state2.task_choice == "classification" and data2.norm_group.value == "minmax"
        assert [r.neurons.value for r in arch2.hidden_rows] == ["5"]
        assert arch2.out_activation.value == "sigmoid"
        assert arch2._io_mode() == IO_AUTO and (arch2.n_inputs.value, arch2.n_outputs.value) == ("2", "3")
        assert not state2.has_network and pred2.predict_btn.disabled  # pesos não reaproveitados
        await arch2._on_create(None)
        assert state2.has_network and state2.data_ready and state2.architecture_text() == "2–5–3"

    asyncio.run(flow())


def test_modo_usar_modelo_existente(csv_regressao, tmp_path):
    async def flow():
        state, _page, (data, arch, _train, _ev, _pred), _toasts, gateway = build_app()
        await data.open_csv(csv_regressao)
        data.norm_group.value = "zscore"
        await arch._on_create(None)
        model = str(tmp_path / "m.json")
        await gateway.call("save", model)

        ustate, _upage, (info, upred), _ut, _ugw = await build_use_app(model)
        assert ustate.mode == "use" and ustate.model_origin == "loaded" and ustate.is_trained
        assert info.file_name.value == "m.json"
        labels = [tile.content.label for tile in info.metrics.controls]
        assert "Entradas: 3" in labels and "Normalização: Z-score" in labels
        assert [f.label for f in upred.fields] == ["x1", "x2", "x3"]  # nomes vindos do modelo
        assert not upred.requirements.content and not upred.predict_btn.disabled
        upred.paste.value = "1; 2; 3"
        upred._on_fill(None)
        await upred._on_predict(None)
        assert len(upred._history) == 1

    asyncio.run(flow())


def test_rotulo_de_texto_com_regressao_e_detectado_na_aba_dados(csv_classes):
    """Cenário do vídeo: esquecer de mudar para Classificação. O problema aparece na aba Dados,
    com correção rápida, antes de criar a rede (sem vai e vem entre as abas)."""
    async def flow():
        state, _page, (data, arch, _train, _ev, _pred), _toasts, _gw = build_app()
        await data.open_csv(csv_classes)  # tarefa ainda em Regressão; saída "classe" = x/y/z
        msg = data.selection_message
        assert msg.level == "error" and msg.title == "A saída “classe” não é numérica"
        assert msg.action[0] == "Mudar para Classificação"
        assert state.selection_problem and state.suggested_io() is None
        # Arquitetura (automático) não deixa criar e explica o porquê, com atalho para Dados
        assert arch.create_btn.disabled
        assert arch.io_message.title == "Os dados escolhidos têm um problema"
        assert arch.io_message.action[0] == "Ir para a aba Dados"

        msg.action[1](None)  # clique em "Mudar para Classificação"
        assert state.task_choice == "classification" and data.task.selected == ["classification"]
        assert state.selection_problem is None and state.suggested_io() == (2, 3)
        assert arch.out_activation.value == "sigmoid" and not arch.create_btn.disabled
        await arch._on_create(None)
        assert state.data_ready and arch.message.title == "Rede criada e dados aplicados"

    asyncio.run(flow())


def test_entrada_nao_numerica_e_bloqueada(csv_classes):
    async def flow():
        state, _page, (data, arch, _train, _ev, _pred), _toasts, _gw = build_app()
        await data.open_csv(csv_classes)
        data._on_role_change("classe", ["in"])  # texto marcado como entrada
        data._on_role_change("b", ["out"])
        assert data.selection_message.title == "A entrada “classe” não é numérica"
        assert state.suggested_io() is None and arch.create_btn.disabled
        data._on_role_change("classe", ["skip"])
        assert state.selection_problem is None and state.suggested_io() == (1, 1)

    asyncio.run(flow())

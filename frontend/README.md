# Frontend: MLP Configurável

Interface gráfica desktop do Trabalho 02 (Sistema Configurável de Redes Neurais Artificiais). Com ela é
possível montar, treinar, avaliar, salvar e usar redes Multilayer Perceptron (MLP) sem alterar o
código-fonte. O foco é **regressão**; a **classificação** está disponível como modo extra.

O frontend não faz cálculos da rede: tudo passa por `backend/service.py` (`MLPService`). Enquanto o
backend não existir, a interface abre em **modo demonstração**, com resultados simulados e um aviso na tela.

## Funcionalidades

- **Arquitetura:** escolher a tarefa (regressão ou classificação), o nº de entradas, as camadas ocultas
  (neurônios e função de ativação), a camada de saída e a semente. O diagrama da rede atualiza ao vivo.
- **Dados:** carregar um CSV, marcar as colunas de entrada e de saída, conferir a compatibilidade com a
  rede, dividir em treinamento/validação/teste e normalizar as entradas (Min-Max ou Z-score).
- **Treinamento:** definir épocas e taxa de aprendizagem (η) e acompanhar ao vivo a curva do MSE de
  treinamento e de validação. A tela marca a melhor época, avisa sobre sobreajuste e exporta o histórico em CSV.
- **Avaliação:** MSE e RMSE por subconjunto. Na classificação, também acurácia e matriz de confusão.
- **Previsão:** calcular a saída da rede para novos valores de entrada.
- **Modelos:** abrir e salvar redes em arquivo JSON.
- **Acessibilidade:** tema claro e escuro com as cores das logos do curso e contraste WCAG AA, atalhos de
  teclado, e mensagens com ícone e texto (não só cor).

## Tecnologias Utilizadas

- **Python** 3.10 ou superior (testado no 3.14)
- **Flet** 1.1.2: interface gráfica
- **flet-charts** 1.1.2: gráficos
- **pytest**: testes automatizados

## Instalação e Execução Local

Na raiz do projeto:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m frontend
```

Na primeira execução, o Flet baixa o cliente desktop (só uma vez).

Para usar o modo demonstração mesmo com o backend pronto, defina a variável `MLP_FRONTEND_DEMO=1` antes de
executar (no PowerShell: `$env:MLP_FRONTEND_DEMO="1"`).

Para rodar os testes (não abrem janela):

```bash
python -m pytest frontend/tests
```

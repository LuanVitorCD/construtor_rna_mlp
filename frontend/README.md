# Frontend: MLP Configurável

Interface gráfica desktop do Trabalho 02 (Sistema Configurável de Redes Neurais Artificiais). Com ela é
possível montar, treinar, avaliar, salvar e usar redes Multilayer Perceptron (MLP) sem alterar o
código-fonte. O foco é **regressão**; a **classificação** está disponível como modo extra.

O frontend não faz cálculos da rede: tudo passa por `backend/service.py` (`MLPService`). Enquanto o
backend não existir, a interface abre em **modo demonstração**, com resultados simulados e um aviso na tela.

## Funcionalidades

Ao abrir, a aplicação pergunta como começar. Os dois modos têm um botão para voltar a esse menu.

**Criar novo modelo**

- **Dados:** escolher a tarefa (regressão ou classificação), carregar um CSV, marcar as colunas de entrada
  e de saída, e definir a divisão treinamento/validação/teste e a normalização (Min-Max ou Z-score).
  As colunas são conferidas na hora (ex.: saída de texto numa regressão), com correção rápida, e na
  classificação as classes do rótulo são contadas antes de a rede existir.
- **Arquitetura:** entradas e saídas no modo **automático** (vindas do CSV) ou **personalizado**, camadas
  ocultas (neurônios e função de ativação), ativação da saída e semente. Ao criar a rede, os dados já
  são carregados, divididos e normalizados; se algo não bater, o erro aparece na mesma tela.
- **Treinamento:** definir épocas e taxa de aprendizagem (η) e acompanhar ao vivo a curva do MSE de
  treinamento e de validação. A tela marca a melhor época, avisa sobre sobreajuste e exporta o histórico em CSV.
- **Avaliação:** MSE por subconjunto e, na regressão, o coeficiente de determinação (R²) e o gráfico
  desejado × estimado. Na classificação, também acurácia e matriz de confusão.
- **Previsão:** calcular a saída da rede para novos valores, digitados ou de um **arquivo CSV** (todas as
  linhas de uma vez). A coluna com a resposta certa pode ficar no arquivo: marcada como "Saída (oculta)",
  ela não vai para a rede e, se você quiser, aparece ao lado do previsto com MSE e R² (ou taxa de acertos).
  Os resultados podem ser salvos em CSV.
- **Salvar modelo** em JSON e **usar modelo como base**: preenche o formulário com a configuração de um
  modelo salvo (tarefa, camadas, ativações, colunas e normalização), sem reaproveitar os pesos.
- **Reiniciar tudo** (com confirmação): apaga rede, CSV, configurações e resultados e volta para a aba Dados.

**Usar modelo existente**

- **Informações do modelo:** arquivo, tarefa, arquitetura (com diagrama), colunas e normalização.
- **Previsão:** a mesma da criação (valores digitados ou arquivo CSV), com o modelo salvo.

**Acessibilidade:** tema claro e escuro com as cores das logos do curso e contraste WCAG AA, atalhos de
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

"""Textos da interface (pt-BR).

Terminologia e orientações seguem o material da disciplina:
- slides "Aula 12 - Neurônio, Perceptron, Adaline" e "Aula 14 - RNA Multi-Layer";
- Ivan Nunes da Silva et al., *Redes Neurais Artificiais para Engenharia e Ciências Aplicadas*;
- Simon Haykin, *Redes Neurais: Princípios e Prática*;
- Goodfellow, Bengio e Courville, *Deep Learning*.
"""

APP_TITLE = "Construtor de Redes Neurais MLP"
APP_SUBTITLE = (
    "Sistema configurável para construir, treinar, armazenar e utilizar Redes Neurais "
    "Artificiais (RNAs) do tipo Multilayer Perceptron (MLP)"
)

TASK_LABELS = {
    "regression": "Regressão",
    "classification": "Classificação (extra)",
}

ACTIVATION_LABELS = {
    "linear": "Linear (identidade)",
    "sigmoid": "Sigmoid (logística)",
    "tanh": "Tangente hiperbólica (tanh)",
    "relu": "ReLU",
}

ACTIVATION_RANGES = {
    "linear": "saída em (−∞, +∞)",
    "sigmoid": "saída em (0, 1)",
    "tanh": "saída em (−1, 1)",
    "relu": "saída em [0, +∞)",
}

NORMALIZATION_LABELS = {
    None: "Nenhuma",
    "minmax": "Min-Max: escala cada entrada para [0, 1]",
    "zscore": "Z-score: média 0 e desvio-padrão 1",
}

NORMALIZATION_SHORT = {None: "nenhuma", "minmax": "Min-Max", "zscore": "Z-score"}

SUBSET_LABELS = {"train": "Treinamento", "val": "Validação", "test": "Teste"}


def activation_label(name: str) -> str:
    return ACTIVATION_LABELS.get(name, name)


HELP = {
    "task": (
        "Regressão estima valores contínuos (ex.: preço, temperatura). "
        "Classificação (extra) atribui uma classe: a saída é codificada como 0/1 (binária) "
        "ou one-hot (uma saída por classe) e treinada com o mesmo MSE."
    ),
    "inputs": "Quantidade de atributos de entrada. Deve ser igual ao nº de colunas de entrada do CSV.",
    "hidden": (
        "Uma camada oculta costuma bastar (Haykin). Para o nº de neurônios, a regra de "
        "Kolmogorov sugere 2n + 1, com n = nº de entradas. Se duas redes generalizam igual, "
        "prefira a menor (Ivan Nunes). Tanh é a escolha usual nas camadas ocultas."
    ),
    "output_regression": (
        "Na regressão, use Linear (identidade) na saída. Sigmoid limita a saída a (0, 1), "
        "Tanh a (−1, 1) e ReLU não produz negativos."
    ),
    "output_classification": (
        "Binária: 1 neurônio Sigmoid (classe 1 se saída ≥ 0,5). "
        "Multiclasse: 1 neurônio por classe (one-hot); vence a maior saída."
    ),
    "seed": "Fixa os números aleatórios para reproduzir o resultado. Deixe em branco para sortear.",
    "init": "Pesos iniciais: Xavier/Glorot para Linear, Sigmoid e Tanh; He para ReLU. Vieses começam em 0.",
    "csv": "Os dados devem vir limpos. Escolha quais colunas são entradas e quais são saídas esperadas.",
    "split": (
        "Treinamento ajusta os pesos. Validação mede a generalização a cada época, sem alterar "
        "os pesos. Teste fica reservado para a avaliação final. Ivan Nunes cita 60–90% para "
        "treinamento; Haykin separa 20% do treinamento para validação."
    ),
    "normalization": (
        "Os parâmetros (mínimo/máximo ou média/desvio) vêm só do subconjunto de treinamento "
        "e são aplicados aos três subconjuntos. As saídas esperadas não são normalizadas. "
        "Entradas em escalas muito diferentes atrapalham o gradiente."
    ),
    "epochs": "Época: uma apresentação completa do subconjunto de treinamento. No SGD, os pesos mudam após cada amostra.",
    "lr": (
        "Taxa de aprendizagem (η): tamanho do passo da descida do gradiente. Valores altos "
        "fazem o erro oscilar ou divergir; valores baixos deixam o treino lento. "
        "Ivan Nunes sugere 0,05 ≤ η ≤ 0,75. Sem normalização, valores menores (0,001–0,01) são mais seguros."
    ),
    "shuffle_seed": "Semente da ordem em que as amostras são apresentadas a cada época (opcional).",
    "retrain": (
        "Treinar de novo continua a partir dos pesos atuais. "
        "Para recomeçar com pesos aleatórios, recrie a rede na aba Arquitetura."
    ),
    "curve": (
        "Se o MSE de validação sobe enquanto o de treinamento continua caindo, a rede está "
        "memorizando o ruído (sobreajuste). O ponto de menor MSE de validação é o ponto de "
        "parada antecipada (Haykin, Fig. 4.20)."
    ),
    "mse": (
        "MSE = (1/N) · Σ (y − ŷ)²: a média dos erros ao quadrado entre a saída esperada (y) e a "
        "saída prevista pela rede (ŷ). Quanto menor, melhor."
    ),
    "test": (
        "Use o conjunto de teste só na avaliação final, depois de escolher arquitetura e "
        "hiperparâmetros. O erro de validação tende a subestimar o erro real (Goodfellow, cap. 5)."
    ),
    "extrapolation": (
        "Uma MLP costuma errar fora da faixa de valores vista no treinamento (Ivan Nunes, cap. 5). "
        "Prefira entradas dentro do domínio dos dados de treino."
    ),
    "predict": "A previsão só faz a propagação direta: os pesos e vieses não mudam.",
    "persistence": (
        "O arquivo JSON guarda entradas, camadas, ativações, pesos, vieses e a normalização "
        "(função e parâmetros) para reconstruir a rede com as mesmas saídas."
    ),
}

SHORTCUTS = [
    ("Ctrl + 1 … 5", "Ir para Arquitetura, Dados, Treinamento, Avaliação ou Previsão"),
    ("Ctrl + O", "Abrir modelo salvo (JSON)"),
    ("Ctrl + S", "Salvar modelo (JSON)"),
    ("Ctrl + T", "Alternar tema claro/escuro"),
    ("Tab / Shift + Tab", "Navegar entre os campos"),
    ("Enter", "Confirmar a ação principal do campo"),
]

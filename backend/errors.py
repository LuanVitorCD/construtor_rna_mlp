"""
Hierarquia de exceções de domínio do backend da rede neural MLP.

Permite que a interface gráfica capture falhas de configuração, dados, estado
ou persistência e exiba mensagens diagnósticas claras ao usuário.
"""


class BackendError(Exception):
    """Exceção base para erros de regra de negócio reportados à interface."""


class ConfigError(BackendError):
    """Erro em parâmetros de topologia, funções de ativação ou hiperparâmetros."""


class DatasetError(BackendError):
    """Erro na leitura, particionamento ou compatibilidade dimensional do dataset."""


class StateError(BackendError):
    """Operação invocada fora da sequência válida do fluxo de trabalho."""


class PersistenceError(BackendError):
    """Erro ao serializar, ler ou validar a integridade estrutural do modelo JSON."""
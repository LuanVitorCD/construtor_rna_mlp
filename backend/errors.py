class BackendError(Exception):
    """Exceção base para erros de regra de negócio reportados à interface."""


class ConfigError(BackendError):
    """Erro em parâmetros de arquitetura, ativações ou hiperparâmetros."""


class DatasetError(BackendError):
    """Erro na leitura, divisão ou compatibilidade de dimensões do dataset."""


class StateError(BackendError):
    """Operação chamada fora de ordem (ex.: treinar antes de carregar dados)."""


class PersistenceError(BackendError):
    """Erro ao salvar, ler ou validar o arquivo JSON do modelo."""
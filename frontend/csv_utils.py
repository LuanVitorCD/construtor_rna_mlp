"""Leitura leve do CSV só para a interface (sem Flet e sem NumPy).

O carregamento de verdade é do backend (``load_dataset``). Aqui a UI só olha as colunas
escolhidas **antes** de a rede existir, para:

- contar as classes da coluna de rótulo (nº de neurônios de saída no modo "Automático");
- avisar na hora, ainda na aba Dados, quando uma coluna não serve para a tarefa
  (ex.: saída de texto numa regressão), em vez de o erro só aparecer ao criar a rede.

As regras seguem as do backend: número = o que ``float(valor)`` aceita; classes = valores
sem espaços nas pontas, em ordem alfabética (``LabelEncoder``).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

MAX_CLASSES = 1000  # acima disso a coluna claramente não é um rótulo de classe


@dataclass
class ColumnProfile:
    distinct: list[str] = field(default_factory=list)  # ordenados; até MAX_CLASSES + 1 itens
    non_numeric: str | None = None  # primeiro valor que não é número (None = coluna numérica)

    @property
    def numeric(self) -> bool:
        return self.non_numeric is None

    @property
    def too_many_values(self) -> bool:
        return len(self.distinct) > MAX_CLASSES


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def profile_columns(path: str, columns: list[str], *, limit: int = MAX_CLASSES) -> dict[str, ColumnProfile] | None:
    """Perfil das ``columns`` do CSV, numa única leitura. None se não der para ler."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None
            keys = {c.strip(): c for c in reader.fieldnames}
            if any(col not in keys for col in columns):
                return None
            seen: dict[str, set[str]] = {col: set() for col in columns}
            profiles = {col: ColumnProfile() for col in columns}
            for row in reader:
                for col in columns:
                    raw = row.get(keys[col]) or ""
                    if len(seen[col]) <= limit:
                        seen[col].add(raw.strip())
                    prof = profiles[col]
                    if prof.non_numeric is None and not _is_number(raw):
                        prof.non_numeric = raw.strip() or "(vazio)"
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    for col in columns:
        profiles[col].distinct = sorted(seen[col])
    return profiles


def distinct_values(path: str, column: str, *, limit: int = MAX_CLASSES) -> list[str] | None:
    """Valores distintos de ``column`` (ordenados), ou None se não der para ler.

    Para de contar ao passar de ``limit`` valores (devolve ``limit + 1`` itens).
    """
    profiles = profile_columns(path, [column], limit=limit)
    return profiles[column].distinct if profiles else None

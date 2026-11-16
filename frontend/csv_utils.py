"""Leitura leve do CSV só para a interface (sem Flet e sem NumPy).

O carregamento de verdade é do backend (``load_dataset``). Aqui a UI só conta as classes
da coluna de rótulo **antes** de a rede existir, para sugerir o nº de neurônios de
saída no modo "Automático". Segue a mesma regra do ``LabelEncoder`` do backend:
valores sem espaços nas pontas, em ordem alfabética.
"""

from __future__ import annotations

import csv

MAX_CLASSES = 1000  # acima disso a coluna claramente não é um rótulo de classe


def distinct_values(path: str, column: str, *, limit: int = MAX_CLASSES) -> list[str] | None:
    """Valores distintos de ``column`` (ordenados), ou None se não der para ler.

    Para de contar ao passar de ``limit`` valores (devolve ``limit + 1`` itens).
    """
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or column not in [c.strip() for c in reader.fieldnames]:
                return None
            key = next(c for c in reader.fieldnames if c.strip() == column)
            seen: set[str] = set()
            for row in reader:
                value = (row.get(key) or "").strip()
                seen.add(value)
                if len(seen) > limit:
                    break
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    return sorted(seen)

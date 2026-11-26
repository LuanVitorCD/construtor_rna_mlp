"""Ponto de entrada.

Na raiz do projeto:
    python -m frontend            # recomendado
    python frontend/main.py       # também funciona
    flet run frontend/main.py     # com recarga automática ao editar
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # permite `import backend` e `import frontend` rodando como script
    sys.path.insert(0, str(ROOT))

import flet as ft  # noqa: E402

from frontend.app import main as app_main  # noqa: E402


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ft.run(app_main)


if __name__ == "__main__":
    run()

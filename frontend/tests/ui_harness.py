"""Monta as telas sem abrir janela, para testes (com o DemoService ou o MLPService real)."""

import asyncio

from frontend.context import AppContext
from frontend.demo_service import DemoBackendError, DemoService
from frontend.model_files import apply_loaded_model, read_model
from frontend.service_gateway import ServiceGateway
from frontend.state import AppState
from frontend.views.architecture import ArchitectureView
from frontend.views.data import DataView
from frontend.views.evaluation import EvaluationView
from frontend.views.model_info import ModelInfoView
from frontend.views.prediction import PredictionView
from frontend.views.training import TrainingView


class FakePage:
    """Substitui ``ft.Page``: as telas só usam update/run_task/diálogos."""

    def __init__(self):
        self.tasks = []

    def update(self, *controls):
        pass

    def run_task(self, handler, *args, **kwargs):
        task = asyncio.ensure_future(handler(*args, **kwargs))
        self.tasks.append(task)
        return task

    def show_dialog(self, dialog):
        pass

    def pop_dialog(self):
        return None


def _context(service, error_cls, mode):
    if service is None:
        service, error_cls = DemoService(), DemoBackendError
    state = AppState(mode=mode)
    page = FakePage()
    gateway = ServiceGateway(service, error_cls, demo=isinstance(service, DemoService))
    ctx = AppContext(page=page, gateway=gateway, state=state, file_picker=None)
    toasts = []
    ctx.toast = lambda level, text: toasts.append((level, text))
    ctx.navigate = lambda index: ctx.extras.setdefault("navigations", []).append(index)

    async def confirm(*_args):
        return True

    ctx.confirm = confirm
    return ctx, state, page, toasts, gateway


def build_app(service=None, error_cls=None):
    """Modo "Criar novo modelo".

    Devolve (state, page, [dados, arquitetura, treino, avaliação, previsão], toasts, gateway).
    """
    ctx, state, page, toasts, gateway = _context(service, error_cls, "create")
    views = [DataView(ctx), ArchitectureView(ctx), TrainingView(ctx), EvaluationView(ctx), PredictionView(ctx)]
    ctx.prepare_data = views[0].prepare
    state.subscribe(lambda: [v.refresh() for v in views])
    return state, page, views, toasts, gateway


async def build_use_app(path, service=None, error_cls=None):
    """Modo "Usar modelo existente": abre ``path`` e devolve (state, page, [info, previsão], toasts, gateway)."""
    ctx, state, page, toasts, gateway = _context(service, error_cls, "use")
    apply_loaded_model(state, await read_model(gateway, path))
    views = [ModelInfoView(ctx), PredictionView(ctx)]
    state.subscribe(lambda: [v.refresh() for v in views])
    state.notify()
    return state, page, views, toasts, gateway


def table_rows(data_view) -> list[list[str]]:
    """Textos das células da tabela de prévia (dados aplicados) da aba Dados."""
    for control in data_view.result_holder.controls:
        row = getattr(control, "controls", None)
        if row and type(row[0]).__name__ == "DataTable":
            return [[cell.content.value for cell in r.cells] for r in row[0].rows]
    return []

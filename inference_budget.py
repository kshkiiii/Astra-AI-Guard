
from contextlib import contextmanager
from contextvars import ContextVar
import time

_deadline = ContextVar("guard_inference_deadline",default=None)


class InferenceBudgetExceeded(RuntimeError):
    pass


def check_budget():
    deadline=_deadline.get()
    if deadline is not None and time.perf_counter() >= deadline:
        raise InferenceBudgetExceeded("CPU inference budget exceeded")


@contextmanager
def inference_budget(seconds):
    token=_deadline.set(time.perf_counter()+seconds)
    try:
        yield
    finally:
        _deadline.reset(token)

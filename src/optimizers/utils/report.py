import typing as tp

import numpy as np

from src.utils.testing import VERIFY_NOT_MODIFIED, verify_not_modified


@verify_not_modified(VERIFY_NOT_MODIFIED)
def f_history(f: tp.Callable[[np.ndarray], float],
              x_hist: list[np.ndarray]) -> list[float]:
    f_hist = [f(x) for x in x_hist]
    return f_hist

@verify_not_modified(VERIFY_NOT_MODIFIED)
def f_residual_history(f: tp.Callable[[np.ndarray], float],
                       x_hist: list[np.ndarray | float],
                       f_hist: list[float] = None) -> list[float]:
    if f_hist is None:
        f_hist = f_history(f, x_hist)
    f_residual_history = [abs(next - prev) for next, prev in zip(f_hist[1:], f_hist)]
    return f_residual_history

@verify_not_modified(VERIFY_NOT_MODIFIED)
def _f_grad_history(f_grad: tp.Callable[[np.ndarray], np.ndarray],
                    x_hist: list[np.ndarray]) -> list[np.ndarray]:
    f_grad_hist = [f_grad(x) for x in x_hist]
    return f_grad_hist

@verify_not_modified(VERIFY_NOT_MODIFIED)
def f_grad_norm_history(f_grad: tp.Callable[[np.ndarray], np.ndarray],
                        x_hist: list[np.ndarray],
                        f_grad_hist: list[np.ndarray] = None) -> list[np.ndarray]:
    if f_grad_hist is None:
        f_grad_hist = _f_grad_history(f_grad, x_hist)
    f_grad_norm_hist = [np.linalg.norm(grad) for grad in f_grad_hist]
    return f_grad_norm_hist

@verify_not_modified(VERIFY_NOT_MODIFIED)
def f_grad_residual_norm_history(f_grad: tp.Callable[[np.ndarray], np.ndarray],
                                 x_hist: list[np.ndarray],
                                 f_grad_hist: list[np.ndarray] = None) -> list[np.ndarray]:
    if f_grad_hist is None:
        f_grad_hist = _f_grad_history(f_grad, x_hist)
    f_grad_residual_norm_hist = [np.linalg.norm(next - prev) for next, prev in zip(f_grad_hist[1:], f_grad_hist)]
    return f_grad_residual_norm_hist


@verify_not_modified(VERIFY_NOT_MODIFIED)
def basic_report(f: tp.Callable[[np.ndarray], float],
                 f_grad: tp.Callable[[np.ndarray], np.ndarray],
                 x_hist: list[np.ndarray]) -> tuple[list[float], list[float], list[float], list[float]]:
    f_hist = f_history(f, x_hist)
    f_residual_hist = f_residual_history(f, x_hist, f_hist=f_hist)

    _f_grad_hist = _f_grad_history(f_grad, x_hist)
    f_grad_norm_hist = f_grad_norm_history(f_grad, x_hist, f_grad_hist=_f_grad_hist)
    f_grad_residual_norm_hist = f_grad_residual_norm_history(f_grad, x_hist, f_grad_hist=_f_grad_hist)

    return f_hist, f_residual_hist, f_grad_norm_hist, f_grad_residual_norm_hist

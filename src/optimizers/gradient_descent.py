import typing as tp
import warnings

import numpy as np

from src.optimizers.step_controllers.step_controllers import StepControllerBase
from src.rayleight_quotient import rayleight_quotient
from src.rayleight_quotient_derivative import rayleight_quotient_derivative
from src.rayleight_quotient_gradient import rayleight_quotient_gradient
from src.rayleight_quotient_hessian import rayleight_quotient_hessian
from src.utils.testing import VERIFY_NOT_MODIFIED, verify_not_modified
from src.utils.linalg import normalize_vector


class GradientDescentBase:
    def __init__(self, f: tp.Callable[[np.ndarray], float],
                       f_derivative: tp.Callable[[np.ndarray], np.ndarray],
                       f_grad: tp.Callable[[np.ndarray], float],
                       max_iter: int=10000) -> None:
        self._f = f
        self._f_derivative = f_derivative
        self._f_grad = f_grad
        self._max_iter = max_iter
        self._eps = 1e-4

        self.f_history: list[float] | None = None
        self.f_grad_history: list[float] | None = None

    @property
    def f_residual_history(self) -> list[float]:
        assert self.f_history is not None
        return [next_ - prev_ for next_, prev_ in zip(self.f_history[1:],
                                                      self.f_history)]

    @property
    def f_grad_norm_history(self) -> list[float]:
        assert self.f_grad_history is not None
        return [np.linalg.norm(grad) for grad in self.f_grad_history]

    @property
    def f_grad_residual_norm_history(self) -> list[float]:
        assert self.f_grad_history is not None
        return [np.linalg.norm(next_ - prev_) for next_, prev_ in zip(self.f_grad_history[1:],
                                                                      self.f_grad_history)]

    def _reset(self) -> None:
        self.f_history = []
        self.f_grad_history = []

    @classmethod
    def from_tensor(cls, A: np.ndarray, *args: tp.Any, **kwargs: tp.Any) -> 'GradientDescentBase':
        return cls(lambda x: rayleight_quotient(A, x),
                   lambda x: rayleight_quotient_derivative(A, x),
                   lambda x: rayleight_quotient_gradient(A, x),
                   *args, **kwargs)

    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def minimize(self, x_0: np.ndarray, step_controller: StepControllerBase) -> np.ndarray:
        self._reset()
        x = normalize_vector(x_0)
        iteration = 0

        while True:
            if iteration >= self._max_iter:
                warnings.warn("max iterations reached")
                break
            iteration += 1

            self.f_history.append(self._f(x))

            grad = self._f_grad(x)
            self.f_grad_history.append(grad)
            if np.linalg.norm(grad) < self._eps:
                break

            controller_params = {
                "grad": grad,
                "deriv": self._f_derivative(x),
                "p": -grad,
            }
            step = step_controller.step(self._f, x, **controller_params)
            x = normalize_vector(x - step * grad)
        return x

class GradientDescent(GradientDescentBase):
    pass

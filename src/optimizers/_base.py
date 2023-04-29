from abc import abstractmethod, ABC
import typing as tp

import numpy as np

from src.optimizers.step_controllers.step_controllers import (
    StepControllerBase, ConstantController
)
from src.rayleight_quotient import rayleight_quotient
from src.rayleight_quotient_derivative import rayleight_quotient_derivative
from src.rayleight_quotient_gradient import rayleight_quotient_gradient
from src.rayleight_quotient_hessian import rayleight_quotient_hessian
from src.utils.testing import VERIFY_NOT_MODIFIED, verify_not_modified


class GradientDescentBase(ABC):
    """
    Base class for all gradient descent-based minimization methods on unit sphere S^{n-1}.
    """
    def __init__(self, f: tp.Callable[[np.ndarray], float],
                       f_derivative: tp.Callable[[np.ndarray], np.ndarray],
                       f_grad: tp.Callable[[np.ndarray], np.ndarray],
                       max_iter: int = 10000,
                       eps: float = 1e-4) -> None:
        super().__init__()
        self._f = f
        self._f_derivative = f_derivative
        self._f_grad = f_grad
        self._max_iter = max_iter
        self._eps = eps

        self._x_history: list[float] = None
        self._step_history: list[float] = None

    def _reset(self) -> None:
        self._x_history = []
        self._step_history = []

    @classmethod
    def from_tensor(cls, A: np.ndarray, *args: tp.Any, **kwargs: tp.Any) -> 'GradientDescentBase':
        return cls(lambda x: rayleight_quotient(A, x),
                   lambda x: rayleight_quotient_derivative(A, x),
                   lambda x: rayleight_quotient_gradient(A, x),
                   *args, **kwargs)

    @abstractmethod
    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def _minimize_impl(self, x_0: np.ndarray, step_controller: StepControllerBase) -> np.ndarray:
        pass

    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def minimize(self, x_0: np.ndarray, step_controller: StepControllerBase) -> np.ndarray:
        self._reset()
        x = self._minimize_impl(x_0, step_controller)
        return x


class NewthonIterationsBase(GradientDescentBase):
    """
    Base class for all newthon-based minimization methods on unit sphere S^{n-1}.
    """
    def __init__(self, f: tp.Callable[[np.ndarray], float],
                       f_derivative: tp.Callable[[np.ndarray], np.ndarray],
                       f_grad: tp.Callable[[np.ndarray], float],
                       f_hessian: tp.Callable[[np.ndarray], np.ndarray],
                       max_iter: int = 10000,
                       eps: float = 1e-6) -> None:
        super().__init__(f, f_derivative, f_grad, max_iter=max_iter, eps=eps)
        self._f_hess = f_hessian

    @classmethod
    def from_tensor(cls, A: np.ndarray, *args: tp.Any, **kwargs: tp.Any) -> 'NewthonIterationsBase':
        return cls(lambda x: rayleight_quotient(A, x),
                   lambda x: rayleight_quotient_derivative(A, x),
                   lambda x: rayleight_quotient_gradient(A, x),
                   lambda x: rayleight_quotient_hessian(A, x),
                   *args, **kwargs)

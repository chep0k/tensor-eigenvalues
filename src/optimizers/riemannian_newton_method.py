import typing as tp
import warnings

import krypy
import numpy as np

from src.optimizers.step_controllers.step_controllers import (
    StepControllerBase, ConstantController
)
from src.rayleight_quotient import rayleight_quotient
from src.rayleight_quotient_derivative import rayleight_quotient_derivative
from src.rayleight_quotient_gradient import rayleight_quotient_gradient
from src.rayleight_quotient_hessian import rayleight_quotient_hessian
from src.utils.linalg import normalize_vector, projection


class RiemannianNewtonIterations:
    def __init__(self, f: tp.Callable[[np.ndarray], float],
                       f_derivative: tp.Callable[[np.ndarray], np.ndarray],
                       f_grad: tp.Callable[[np.ndarray], float],
                       f_hessian: tp.Callable[[np.ndarray], np.ndarray],
                       max_iter: int = 10000) -> None:
        self.f = f
        self._f_deriv = f_derivative
        self.f_grad = f_grad
        self._f_hess = f_hessian
        self._max_iter = max_iter

        self.eps: int = 1e-6
        self.f_history: list[float] | None = None
        self.f_grad_history: list[float] | None = None
        self.lin_solver_error_history: list[float] | None = None

    @property
    def f_residual_history(self) -> list[float]:
        assert self.f_history is not None
        return [next_ - prev_ for next_, prev_ in zip(self.f_history[1:],
                                                      self.f_history)]

    @property
    def f_grad_norm_history(self) -> list[float]:
        assert self.f_grad_history is not None
        return [np.linalg.norm(grad) for grad in self.f_grad_history]

    def _reset(self) -> None:
        self.f_history = []
        self.f_grad_history = []
        self.lin_solver_error_history = []

    @classmethod
    def from_tensor(cls, A: np.ndarray, *args: tp.Any, **kwargs: tp.Any) -> 'RiemannianNewtonIterations':
        return cls(lambda x: rayleight_quotient(A, x),
                   lambda x: rayleight_quotient_derivative(A, x),
                   lambda x: rayleight_quotient_gradient(A, x),
                   lambda x: rayleight_quotient_hessian(A, x),
                   *args, **kwargs)

    def launch(self, x_0: np.ndarray, step_controller: StepControllerBase | None = None):
        self._reset()
        x = normalize_vector(x_0)
        if step_controller is None:
            step_controller = ConstantController()

        iteration = 0
        while True:
            if iteration >= self._max_iter:
                warnings.warn("max iterations reached")
                break
            iteration += 1

            self.f_history.append(self.f(x))

            grad = self.f_grad(x)
            self.f_grad_history.append(grad)
            if np.linalg.norm(grad) < self.eps:
                break

            hess = self._f_hess(x)
            xi, _ = krypy.minres(hess, -grad)
            xi_tangent = xi - (x.T @ xi) * x  # xi_tangent = projection(x) @ xi
            self.lin_solver_error_history.append(np.linalg.norm(hess @ xi_tangent + grad))

            # x = normalize_vector(x + xi_tangent)
            step_controller_params = {
                "grad": grad,
                "deriv": self._f_deriv(x),
                "p": xi_tangent,
            }
            step = step_controller.step(self.f, x, **step_controller_params)
            x = normalize_vector(x + step * xi_tangent)
        return x

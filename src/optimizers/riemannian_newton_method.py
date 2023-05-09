import typing as tp
import warnings

import krypy
import numpy as np

from src.optimizers.step_controllers.step_controllers import (
    StepControllerBase, ConstantController
)
from src.utils.testing import VERIFY_NOT_MODIFIED, verify_not_modified
from src.utils.linalg import normalize_vector, projection

from ._base import NewthonIterationsBase


class RiemannianNewtonIterations(NewthonIterationsBase):
    """
    Riemannian-Newthon method implementation.

    Correction to the current point estimated to minimize quadratic expansion
    of Rayleigh quotient in the current point.
    #TODO: For more information, see []
    Compute gradient, hessian and directional derivative exactly in each point,
    as in the base class.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._invalid = kwargs.get("invalid", "error")

        self.lin_solver_error_history: list[float] = None

    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def shortname(self) -> str:
        return "RNI"

    def _reset(self) -> None:
        super()._reset()
        self.lin_solver_error_history = []

    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def _minimize_impl(self, x_0: np.ndarray, step_controller: StepControllerBase):
        x = normalize_vector(x_0)

        iteration = 0
        while True:
            if iteration >= self._max_iter:
                warnings.warn("max iterations reached")
                break
            iteration += 1
            self._x_history.append(x)

            grad = self._f_grad(x)
            if np.linalg.norm(grad) < self._eps:
                break

            hess = self._f_hess(x)
            xi, _ = krypy.minres(hess, -grad)
            xi_tangent = xi - (x.T @ xi) * x  # xi_tangent = projection(x) @ xi

            err = np.linalg.norm(hess @ xi_tangent + grad)
            self.lin_solver_error_history.append(err)

            # x = normalize_vector(x + xi_tangent)
            step_controller_params = dict(invalid=self._invalid, grad=grad, deriv=self._f_derivative(x))
            step = step_controller.step(self._f, x, xi_tangent, **step_controller_params)
            self._step_history.append(step)
            x = normalize_vector(x + step * xi_tangent)
        return x

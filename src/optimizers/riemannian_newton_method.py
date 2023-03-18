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
    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def _minimize_impl(self, x_0: np.ndarray, step_controller: StepControllerBase = None):
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
            self._x_history.append(x)

            grad = self._f_grad(x)
            if np.linalg.norm(grad) < self._eps:
                break

            hess = self._f_hess(x)
            xi, _ = krypy.minres(hess, -grad)
            xi_tangent = xi - (x.T @ xi) * x  # xi_tangent = projection(x) @ xi
            self.lin_solver_error_history.append(np.linalg.norm(hess @ xi_tangent + grad))

            # x = normalize_vector(x + xi_tangent)
            step_controller_params = dict(invalid='fix', grad=grad, deriv=self._f_derivative(x))
            step = step_controller.step(self._f, x, xi_tangent, **step_controller_params)
            x = normalize_vector(x + step * xi_tangent)
        self._report_back()
        return x

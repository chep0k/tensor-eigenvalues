import typing as tp
import warnings

import numpy as np

from ._base import GradientDescentBase
from src.optimizers.step_controllers.step_controllers import StepControllerBase
from src.utils.testing import VERIFY_NOT_MODIFIED, verify_not_modified
from src.utils.linalg import normalize_vector


class GradientDescent(GradientDescentBase):
    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def _minimize_impl(self, x_0: np.ndarray, step_controller: StepControllerBase) -> np.ndarray:
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

            controller_params = {
                "grad": grad,
                "deriv": self._f_derivative(x),
            }
            step = step_controller.step(self._f, x, -grad, **controller_params)
            self._step_history.append(step)
            x = normalize_vector(x - step * grad)
        return x

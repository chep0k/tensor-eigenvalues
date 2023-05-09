import typing as tp
import warnings

import krypy
import numpy as np

from src.optimizers.step_controllers.step_controllers import (
    StepControllerBase
)
from src.utils.testing import VERIFY_NOT_MODIFIED, verify_not_modified
from src.utils.linalg import normalize_vector

from .riemannian_newton_method import RiemannianNewtonIterations


class RiemannianTrustRegionMethod(RiemannianNewtonIterations):
    """
    Improve Riemannian-Newthon method with trust region approach.

    Work as the base class, but norm of correction on each iteration
    dinamically changes to ensure the quadratic expansion of
    the Rayleigh quotient in that radius approximates well
    the Rayleigh quotient itself.
    For more information, see [Optimization Algorithms on Matrix Manifolds,
                               P.-A. Absil et al. Algorithm 10]
    Compute gradient, hessian and directional derivative exactly in each point,
    as in the base class.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._invalid = kwargs.get("invalid", "error")

        self._delta_0 = 1.0
        self._lb = 0.25
        self._hb = 0.75
        self._p = 0.1

    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def shortname(self) -> str:
        return "RTR"

    def accuracy(self, x_0: np.ndarray, x_next: np.ndarray, xi: np.ndarray, step: float, grad: np.ndarray):
        rq_0 = self._f(x_0)
        rq_next = self._f(x_next)
        # model_next = rq_0 + grad.T @ xi + 0.5 * xi.T @ hess @ xi
        model_next = rq_0 + step * (1 - 0.5 * step) * np.inner(xi, grad)  # as hess @ xi \approx -grad

        p = (rq_0 - rq_next) / (rq_0 - model_next)
        return p


    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def _minimize_impl(self, x_0: np.ndarray, step_controller: StepControllerBase):
        x = normalize_vector(x_0)

        iteration = 0
        delta = self._delta_0
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

            # augment initial step (alpha_0) of step controller
            alpha_0 = min(1.0, delta / np.linalg.norm(xi_tangent))
            step_controller_params = dict(alpha_0=alpha_0, invalid=self._invalid, grad=grad, deriv=self._f_derivative(x))
            step = step_controller.step(self._f, x, xi_tangent, **step_controller_params)
            self._step_history.append(step)

            x_next = normalize_vector(x + step * xi_tangent)

            p = self.accuracy(x, x_next, xi_tangent, step, grad)
            if 0 < p < self._lb:
                delta /= 4
            elif p > self._hb and abs(step) == alpha_0:
                delta = min(2 * delta, 2 * self._delta_0)

            if p > self._p:
                x = x_next

        return x

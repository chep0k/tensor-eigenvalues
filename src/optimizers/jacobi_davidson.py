import typing as tp
import warnings

import krypy
import numpy as np

from src.optimizers.step_controllers.step_controllers import StepControllerBase
from src.rayleigh_quotient import rayleigh_quotient
from src.rayleigh_quotient_gradient import rayleigh_quotient_gradient
from src.rayleigh_quotient_hessian import rayleigh_quotient_hessian
from src.utils.linalg import (
    generate_normalized_vector, normalize_vector, projection, tenvec, tenxmatrix
)
from src.utils.testing import VERIFY_NOT_MODIFIED, verify_not_modified

from ._base import GradientDescentBase


class JDIteration:
    """
    Implement Jacobi-Davidson method for optimizing Rayleigh quotient.

    Jacobi correction equations is as follows
    Proj(x) @ ( (k-1) A(I, I, x, ..., x) - R(x) I) Proj(x) xi = - Proj(x) A(I, x, ..., x)
    """
    def __init__(self, A: np.ndarray,
                 eigsolver: type[GradientDescentBase], eigsolver_params: dict = {},
                 max_iter: int = 100, eps=1e-4) -> None:
        self.A = A
        self.n = self.A.shape[0]
        self.k = self.A.ndim
        self.eigsolver = eigsolver
        self._eigsolver_params = eigsolver_params
        self._max_iter = max_iter
        self._eps = eps

        self.rq = lambda x: rayleigh_quotient(A, x)
        self.rq_grad = lambda x: rayleigh_quotient_gradient(A, x)
        self.rq_hess = lambda x: rayleigh_quotient_hessian(A, x)

        self._x_top_history = None
        self._x_history = None
        self._step_history = None

    # @classmethod
    # def from_tensor(cls, A: np.ndarray, *args: tp.Any, **kwargs: tp.Any) -> 'JDIteration':
    #     return cls(A, *args, **kwargs)

    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def shortname(self) -> str:
        return "JDI"

    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def _lhs_rhs_jacobi_equation(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # tL = tenvec(self.A, x, self.k - 2)
        # tR = tL @ x  # tR = tenvec(self.A, x, self.k - 1)
        # rq = tR.T @ x  # rq = rayleight_quotient(self.A, x)

        # lhs = (self.k - 1) * (tL - x @ (x.T @ tL)) - \
        #     rq * np.eye(self.n) + rq * np.outer(x, x)  # TODO: lhs *= proj(x)
        # rhs = - (tR - (x.T @ tR) * x)

        lhs = projection(x) @ ((self.k - 1) * tenvec(self.A, x, self.k - 2) - self.rq(x) * np.eye(self.n)) @ projection(x)
        rhs = - projection(x) @ tenvec(self.A, x, self.k - 1)

        return lhs, rhs

    @staticmethod
    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def _add_col_and_orthonormalize_(V: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        compute orthonormalized [V, s]

        :param V: orthonormalized matrix of shape (n, m)
        :param s: vector of dimension n to cat

        :return: resulted matrix
        """
        # V.T @ (s + Vt) == 0
        # V.T @ s + t == 0
        t = -V.T @ s
        s = s + V @ t
        s = normalize_vector(s)

        V = np.hstack([V, s.reshape(-1, 1)])
        # print(V.shape)
        # print()

        return V


    def _reset(self):
        self._x_top_history = []
        self._x_history = []
        self._step_history = []


    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def minimize(self, V: np.ndarray, step_controller: StepControllerBase) -> np.ndarray:
        self._reset()

        if V.ndim == 1:
            V = V.reshape((-1, 1))
        iteration = 0
        while True:
            if iteration >= self._max_iter:
                warnings.warn("max iterations reached")
                break
            if V.shape[1] > V.shape[0]:
                warnings.warn("|V| is no more orthonormal. It has more columns than rows")
                break
            iteration += 1

            H = tenxmatrix(self.A, V, self.k)

            v_0 = generate_normalized_vector(V.shape[1])
            # v_0 = V.T @ self._x_top_history[-1] if self._x_top_history else generate_normalized_vector(V.shape[1])
            eigsolver = self.eigsolver.from_tensor(H, **self._eigsolver_params)
            # it is important that |v| is the leftmost eigenvalue here
            v = eigsolver.minimize(v_0, step_controller)

            x = V @ v  # x = normalize_vector(V @ v)
            self._x_top_history.append(x)
            self._x_history += [V @ u for u in eigsolver._x_history]
            self._step_history += eigsolver._step_history

            if np.linalg.norm(self.rq_grad(x)) < self._eps:
                break

            lhs, rhs = self._lhs_rhs_jacobi_equation(x)
            s, _ = krypy.minres(lhs, rhs)
            # V = self._add_col_and_orthonormalize_(V, s)
            s_tangent = s - (x.T @ s) * x  # s_tangent = projection(x) @ s
            V = self._add_col_and_orthonormalize_(V, s_tangent)

        return x

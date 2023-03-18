from abc import ABC, abstractmethod
import typing as tp

import numpy as np

from src.utils.testing import VERIFY_NOT_MODIFIED, verify_not_modified
from src.utils.linalg import normalize_vector


class StepControllerBase(ABC):
    """
    Base class for calculating a step, which is a multiplier to calculated retraction.
    """
    @abstractmethod
    def name(self) -> str:
        """
        Provide a name of each method

        :return: a name
        """
        raise NotImplemented()

    def _update_step(self, d: float, step: float, invalid: str) -> float:
        """
        Function to handle invalid direction and invert step, if needed

        :param d: grad @ direction
        :param step: step to update
        :param invalid: should be either 'error' or 'fix' or 'ignore'

        :return: updated step
        """
        if d > 0:
            if invalid == "error":
                raise ValueError("provided direction |p| is invalid")
            elif invalid == "fix":
                step = -step
            elif invalid == "ignore":
                pass
            else:
                raise ValueError("unexpected value for |invalid| argument")
        return step

    @abstractmethod
    def step(self, f: tp.Callable[[np.ndarray], float], x: np.ndarray, p: np.ndarray,
             invalid: str = "error", **kwargs: tp.Any) -> float:
        """
        :param f: function to be minimized
        :param x: current point
        :param p: computed direction of retraction
        :param invalid: indicates how to handle invalid direction
        :param **kwargs: see documentation of a certain implementation

        :return: calculated step
        """
        raise NotImplemented("abstract method |step| have to be overrdien in class derived from |StepControllerBase|")


class ConstantController(StepControllerBase):
    """
    Implements constant step method.
    """
    def name(self):
        return f"constant={self._step}"

    def __init__(self, step: float=0.01) -> None:
        """
        :param step: value of step to be returned on every single call
        """
        assert step > 0, "parameter |step| should be strictly positive"
        self._step = step

    verify_not_modified(VERIFY_NOT_MODIFIED)
    def step(self, f, x, p, invalid="error", **kwargs):
        """
        :param **kwargs: should contain value with key |grad|,
                         which is the gradient of function |f| at |x|
        """
        assert "grad" in kwargs, "provide |grad| argument please"
        grad: np.ndarray = kwargs["grad"]

        t = grad @ p
        step = self._update_step(t, self._step, invalid=invalid)
        return step


class BacktrackingLineSearchController(StepControllerBase):
    """
    Implements backtracking line search based on Armijo rule.
    #TODO: insert formula here
    """
    def name(self):
        return "backtracking line search"

    def __init__(self, alpha_0: float = 1.0, tau: float=0.5, c: float=0.5) -> None:
        """
        :param alpha_0: initial alpha (step) value
        :param tau: coefficient to multiply |alpha| by on each iteration
        :param c: coefficient for rhs of inequality
        """
        self._alpha_0 = alpha_0
        self._tau = tau
        self._c = c

    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def _step_safe(self, f: tp.Callable[[np.ndarray], float],
                   x: np.ndarray, p: np.ndarray, grad_by_p: float, alpha: float) -> float:
        """
        |step| implementation with correct parameters

        :param grad_by_p: grad @ p, should be > 0 (no check)
        :alpha: initial step value
        other params derives parameters of the same name from |step| function
        """
        f_x_start = f(x)
        t = self._c * grad_by_p
        while True:
            x_next = normalize_vector(x + alpha * p)
            if f(x_next) <= f_x_start + alpha * t:
                break
            alpha = self._tau * alpha
        return alpha

    # @verify_not_modified(VERIFY_NOT_MODIFIED)
    def step(self, f, x, p, invalid="error", **kwargs: tp.Any):
        """
        :param **kwargs: should contain value with key |grad|,
                         which is the gradient of function |f| at |x|
        """
        assert "grad" in kwargs, "provide |grad| argument please"
        grad = kwargs["grad"]

        t = grad @ p
        alpha = self._update_step(t, self._alpha_0, invalid=invalid)
        return self._step_safe(f, x, p, t, alpha)

from abc import ABC, abstractmethod
import typing as tp

import numpy as np

from src.utils.testing import VERIFY_NOT_MODIFIED, verify_not_modified
from src.utils.linalg import normalize_vector


class StepControllerBase(ABC):
    @staticmethod
    @abstractmethod
    def name():
        raise NotImplemented()

    @abstractmethod
    def step(self, f: tp.Callable[[np.ndarray], float], x: np.ndarray, **kwargs: tp.Any) -> float:
        raise NotImplemented("abstract method |step| have to be overrdien in class derived from |StepControllerBase|")


class ConstantController(StepControllerBase):
    @staticmethod
    def name():
        return "constant"

    def __init__(self, step: float=0.01) -> None:
        self._step = step

    verify_not_modified(VERIFY_NOT_MODIFIED)
    def step(self, *args: tp.Any, **kwargs: tp.Any) -> float:
        return self._step


class WeirdGradientController(StepControllerBase):
    @staticmethod
    def name():
        return "weird gradient controller"

    def __init__(self, alpha_0: float = 1.0, tau: float=0.5, c: float=0.5) -> None:
        self._alpha_0 = alpha_0
        self._tau = tau
        self._c = c

    @verify_not_modified(VERIFY_NOT_MODIFIED)
    def step(self, f: tp.Callable[[np.ndarray], float], x: np.ndarray, **kwargs: tp.Any) -> float:
        assert "grad" in kwargs
        grad = kwargs["grad"]
        p = -grad
        m = p @ grad
        t = -(self._c * m)
        alpha = self._alpha_0
        f_x = f(x)
        while True:
            if f_x - f(x + alpha * p) >= alpha * t:
                break
            alpha *= self._tau
        return alpha


class BacktrackingLineSearchController(StepControllerBase):
    @staticmethod
    def name():
        return "backtracking line search"

    def __init__(self, alpha_0: float = 1.0, tau: float=0.5, c: float=0.5) -> None:
        self._alpha_0 = alpha_0
        self._tau = tau
        self._c = c

    # @verify_not_modified(VERIFY_NOT_MODIFIED)
    def step(self, f: tp.Callable[[np.ndarray], float], x: np.ndarray, **kwargs: tp.Any) -> float:
        assert "grad" in kwargs and "p" in kwargs
        grad, p = kwargs["grad"], kwargs["p"]

        f_x_start = f(x)
        t = self._c * (grad @ p)
        print(t)
        alpha = self._alpha_0
        while True:
            x_next = normalize_vector(x + alpha * p)
            # if f(x_next) - f_x_start <= alpha * t:
            if f_x_start - f(x_next) >= - alpha * t:
                break
            alpha *= self._tau
        # print(f"{alpha=}")
        # print(f_x_start)
        # print(f(x_next))
        # print()
        return alpha

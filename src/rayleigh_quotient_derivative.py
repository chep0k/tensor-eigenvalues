import numpy as np
import typing as tp

from .utils.testing import VERIFY_NOT_MODIFIED, verify_not_modified
from .utils.linalg import tenvec


@verify_not_modified(VERIFY_NOT_MODIFIED)
def rayleigh_quotient_derivative(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    k = A.ndim
    return k * tenvec(A, x, times=k-1)

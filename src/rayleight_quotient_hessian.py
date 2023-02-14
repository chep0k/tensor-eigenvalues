import numpy as np

from .rayleight_quotient import rayleight_quotient
from .utils.linalg import projection, tenvec
from .utils.testing import VERIFY_NOT_MODIFIED, verify_not_modified


@verify_not_modified(VERIFY_NOT_MODIFIED)
def rayleight_quotient_hessian(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    # x expected to be normalized
    k = A.ndim
    t = (k - 1) * tenvec(A, x, times=k-2)
    t[np.diag_indices_from(t)] -= rayleight_quotient(A, x)
    proj = projection(x)
    return k * (proj @ t @ proj)

@verify_not_modified(VERIFY_NOT_MODIFIED)
def rayleight_quotient_hessian_naive(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    pass

from itertools import product
import numpy as np
import typing as tp

from .utils.testing import VERIFY_NOT_MODIFIED, verify_not_modified
from .utils.linalg import tenvec


@verify_not_modified(VERIFY_NOT_MODIFIED)
def rayleight_quotient(A: np.ndarray, X: np.ndarray) -> float:
    """
    Compute Rayleigh quotient with numpy.

    :param A: supersymmetric tensor of shape (n x n x ... x n) of order k
    :param X: normalized vector of shape (n,)

    :return: Rayleigh quotient R_A(X)
    """
    k = A.ndim
    return tenvec(A, X, times=k)


@verify_not_modified(VERIFY_NOT_MODIFIED)
def rayleight_quotient_naive(A: np.ndarray, X: np.ndarray) -> float:
    """
    Compute Rayleigh quotient with for loops.

    :param A: supersymmetric tensor of shape (n x n x ... x n) of order k
    :param X: normalized vector of shape (n,)

    :return: Rayleigh quotient R_A(X)
    """
    n = X.shape[0]
    k = A.ndim
    rq = 0.0
    for perm in product(range(n), repeat=k):
        X_prod = np.prod(X[list(perm)])
        rq += X_prod * A[tuple(perm)]
    return rq

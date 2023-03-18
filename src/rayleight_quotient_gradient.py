from itertools import product
import numpy as np
import typing as tp

from .rayleight_quotient import rayleight_quotient, rayleight_quotient_naive
from .utils.testing import VERIFY_NOT_MODIFIED, verify_not_modified
from .utils.linalg import tenvec


@verify_not_modified(VERIFY_NOT_MODIFIED)
def rayleight_quotient_gradient(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute Rayleigh quotient gradient with numpy.

    :param A: supersymmetric tensor of shape (n x n x ... x n) of order k
    :param x: normalized vector of shape (n,)

    :return: Rayleigh quotient gradient grad(R_A(x))
    """
    k = A.ndim
    t = tenvec(A, x, times=k-1)
    rq = x @ t  # rq = rayleight_quotient(A, x)
    return k * (t - rq * x)


@verify_not_modified(VERIFY_NOT_MODIFIED)
def rayleight_quotient_gradient_naive(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Compute Rayleigh quotient gradient with for loops.

    :param A: supersymmetric tensor of shape (n x n x ... x n) of order k
    :param X: normalized vector of shape (n,)

    :return: Rayleigh quotient gradient  grad(R_A(X))
    """
    n = X.shape[0]
    k = A.ndim
    rq = np.zeros(n)
    for perm in product(range(n), repeat=k):
        rq_ind, *X_ind = perm
        X_prod = np.prod(X[list(X_ind)])
        rq[rq_ind] += X_prod * A[tuple(perm)]
    rq_naive = rayleight_quotient_naive(A, X)
    return k * (rq - rq_naive * X)

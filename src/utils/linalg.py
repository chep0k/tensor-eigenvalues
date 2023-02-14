from itertools import permutations
import numpy as np
import typing as tp

from .testing import VERIFY_NOT_MODIFIED, verify_not_modified


@verify_not_modified(VERIFY_NOT_MODIFIED)
def generate_supersymmetric_tensor(n: int, k: int) -> np.ndarray:
    tensor_shape = np.full(shape=k, fill_value=n)
    base_tensor = np.random.rand(*tensor_shape)
    a = np.zeros(tensor_shape)
    for perm in permutations(range(k), r=k):
        a += np.transpose(base_tensor, axes=perm)
    return a


@verify_not_modified(VERIFY_NOT_MODIFIED)
def normalize_vector(x: np.ndarray) -> np.ndarray:
    x_norm = np.linalg.norm(x)
    assert x_norm > 0, "cannot normalize vector of zeros"
    return x / x_norm


@verify_not_modified(VERIFY_NOT_MODIFIED)
def generate_normalized_vector(n: int) -> np.ndarray:
    x = np.random.rand(n)
    return normalize_vector(x)


@verify_not_modified(VERIFY_NOT_MODIFIED)
def tenvec(A: np.ndarray, x: np.ndarray, times: int) -> np.ndarray | float:
    A_by_x = A
    for _ in range(times):
        A_by_x = np.einsum("i,...i->...", x, A_by_x)
    return A_by_x


class Projection:
    def __init__(self, x: np.ndarray) -> None:
        self.x = x

    def __matmul__(self, other):
        pass

    def __rmatmul__(self, other):
        pass

@verify_not_modified(VERIFY_NOT_MODIFIED)
def projection(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    # return np.eye(n) - x @ x.T
    return np.eye(n) - np.outer(x, x)

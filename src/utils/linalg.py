from itertools import permutations
import numpy as np
import typing as tp

from .testing import VERIFY_NOT_MODIFIED, verify_not_modified


@verify_not_modified(VERIFY_NOT_MODIFIED)
def generate_supersymmetric_tensor(n: int, k: int) -> np.ndarray:
    """
    Generate supersymmetric tensor of shape (n, n, ..., n) of order k

    :param n: dimension of tensor
    :param k: order of tensor

    :return: resulted tensor
    """
    tensor_shape = np.full(shape=k, fill_value=n)
    base_tensor = np.random.rand(*tensor_shape)
    A = np.zeros(tensor_shape)
    for perm in permutations(range(k), r=k):
        A += np.transpose(base_tensor, axes=perm)
    return A

@verify_not_modified(VERIFY_NOT_MODIFIED)
def generate_supersymmetric_tensor_plus(n: int, k: int) -> np.ndarray:
    """
    Generate supersymmetric positive definite tensor of shape (n, n, ..., n) of order k

    :param n: dimension of tensor
    :param k: order of tensor

    :return: resulted tensor
    """
    tensor_shape = np.full(shape=k, fill_value=n)
    A = np.zeros(tensor_shape)
    diag_values = np.random.exponential(size=n)
    for i, value in enumerate(diag_values):
        index = tuple(np.full(shape=k, fill_value=i))
        A[index] = value
    return A

@verify_not_modified(VERIFY_NOT_MODIFIED)
def normalize_vector(x: np.ndarray) -> np.ndarray:
    """
    Normalize vector x, i.e. compute x / ||x||_2

    :param x: vector to normalize

    :return: resulted vector
    """
    x = x.astype(np.float128)

    x_norm = np.linalg.norm(x)
    assert x_norm > 0, "cannot normalize vector of zeros"

    x = x / x_norm
    return x.astype(float)

@verify_not_modified(VERIFY_NOT_MODIFIED)
def generate_normalized_vector(n: int) -> np.ndarray:
    """
    Generate normalize vector x, i.e. vector x, which satisfies ||x||_2 == 1

    :param n: desired dimension of vector

    :return: resulted vector
    """
    x = np.random.rand(n)
    x = normalize_vector(x)
    return x

@verify_not_modified(VERIFY_NOT_MODIFIED)
def tenvec(A: np.ndarray, x: np.ndarray, times: int) -> np.ndarray | float:
    """
    Compute A(I_n, ..., I_n, x, ..., x) with |times| entries of vector x in product

    :param A: (supersymmetric) tensor of shape (n, n, ..., n) of order k
    :param x: (normalized) vector of dimension n
    :param times: number of entries of vector x in product

    :return: resulted tensor (or float number, when times == k)
    """
    t = A
    for _ in range(times):
        t = np.einsum("i,...i->...", x, t)
    return t

@verify_not_modified(VERIFY_NOT_MODIFIED)
def tenxmatrix(A: np.ndarray, M: np.ndarray, times: int) -> np.ndarray | float:
    """
    Compute A(I_n, ..., I_n, M, ..., M) with |times| entries of matrix M in product

    :param A: (supersymmetric) tensor of shape (n, n, ..., n) of order k
    :param M: matrix of shape (n, m)
    :param times: number of entries of matrix M in product

    :return: resulted tensor
    """
    t = A
    for _ in range(times):
        t = np.einsum("i...,ij->...j", t, M)
    return t

# class Projection:
#     def __init__(self, x: np.ndarray) -> None:
#         self.x = x

#     def __matmul__(self, other):
#         pass

#     def __rmatmul__(self, other):
#         pass

@verify_not_modified(VERIFY_NOT_MODIFIED)
def projection(x: np.ndarray) -> np.ndarray:
    """
    Compute projection of vector x to unit sphere S^{n-1}

    :param x: vector of dimension n

    :return: resulted projection
    """
    t = -np.outer(x, x)
    t[np.diag_indices_from(t)] += 1
    return t
    # n = x.shape[0]
    # return np.eye(n) - np.outer(x, x)

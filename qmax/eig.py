from __future__ import annotations
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray, Array, Scalar

from .hilbert_space import AbstractHilbertSpace, AbstractState
from .lanczos import lanczos

if TYPE_CHECKING:
    from .operator import Operator


# TODO:
#   1. Davidson's method 
#   2. Chebyshev filter based preconditioning
#
#


def op_eigh(
    operator: Operator,
    hilbert_space: AbstractHilbertSpace) -> tuple[Array, AbstractState, Array]:

    mat = operator.to_matrix(hilbert_space)
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    y_eigvecs = hilbert_space.from_coeffs(eigvecs.T)
    residuals = jnp.linalg.norm(mat @ eigvecs - eigvecs * eigvals, axis=1)
    return eigvals, y_eigvecs, residuals


def op_eigh_lanczos(
    op: Operator,
    hilbert_space: AbstractHilbertSpace,
    num_iterations: int,
    *,
    orthogonalize: bool=True,
    key: PRNGKeyArray=jax.random.key(0)) -> tuple[Array, AbstractState, Array]:

    alpha, beta, Q, _ = lanczos(
        op, hilbert_space, num_iterations, orthogonalize=orthogonalize, key=key)
    eigvals, Y = jax.scipy.linalg.eigh_tridiagonal(alpha, beta[:-1])
    eigvecs = Q.contract(Y)
    residuals = jnp.abs(beta[-1] * Y[-1, :])

    return eigvals, eigvecs, residuals


def op_spectral_bounds_lanczos(
    op: Operator, 
    hilbert_space: AbstractHilbertSpace, 
    num_iterations: int=25,
    *, 
    orthogonalize: bool=False,
    key: PRNGKeyArray=jax.random.key(0)) -> tuple[Scalar, Scalar]:

    alpha, beta, _, _ = lanczos(
        op, hilbert_space, num_iterations, orthogonalize=orthogonalize, key=key)
    eigvals = jax.scipy.linalg.eigh_tridiagonal(alpha, beta[:-1], eigvals_only=True)
    return jnp.min(eigvals), jnp.max(eigvals)

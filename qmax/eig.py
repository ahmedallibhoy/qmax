from typing import Optional

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray, Array

from .hilbert_space import AbstractHilbertSpace, AbstractState
from .operator import Operator
from .lanczos import eigh_lanczos_matvec, restart_lanczos


def op_eigh(
    operator: Operator,
    hilbert_space: AbstractHilbertSpace) -> tuple[Array, AbstractState, Array]:

    mat = operator.to_matrix(hilbert_space)
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    y_eigvecs = hilbert_space.from_coeffs(eigvecs.T)
    residuals = jnp.linalg.norm(mat @ eigvecs - eigvecs * eigvals, axis=1)
    return eigvals, y_eigvecs, residuals


def op_eigh_lanczos(
    operator: Operator,
    hilbert_space: AbstractHilbertSpace,
    num_eig: int,
    num_iterations: int=200,
    *,
    select: str="smallest",
    orthogonalize: bool=True,
    grad_safe: bool=False,
    tol: Optional[float]=None,
    key: PRNGKeyArray=jax.random.key(0)) -> tuple[Array, AbstractHilbertSpace, Array]:

    def matvec(q):
        y = hilbert_space.from_coeffs(q)
        result = operator(y)
        return result.coeffs

    eigvals, eigvecs, residuals = eigh_lanczos_matvec(
        matvec, hilbert_space.dim, num_eig, num_iterations,
        eigvals_only=False, return_residuals=True, grad_safe=grad_safe,
        select=select, orthogonalize=orthogonalize, tol=tol,
        is_complex=True, key=key)

    Ys = hilbert_space.from_coeffs(eigvecs.T)
    return eigvals, Ys, residuals


def op_eigh_restart(
    operator: Operator,
    hilbert_space: AbstractHilbertSpace,
    num_eig: int,
    num_iterations: int=200,
    *,
    buffer: int=25,
    num_restarts: int=4,
    select: str="smallest",
    key: PRNGKeyArray=jax.random.key(0)) -> tuple[Array, AbstractHilbertSpace, Array]:

    def matvec(q):
        y = hilbert_space.from_coeffs(q)
        result = operator(y)
        return result.coeffs

    eigvals, eigvecs, residuals = restart_lanczos(
        matvec, hilbert_space.dim, num_eig, num_iterations,
        buffer=buffer, num_restarts=num_restarts, select=select,
        is_complex=True, key=key)

    Ys = hilbert_space.from_coeffs(eigvecs.T)
    return eigvals, Ys, residuals

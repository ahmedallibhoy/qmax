from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import ScalarLike

from ..hilbert_space import AbstractHilbertSpace, AbstractState
from ..lanczos import lanczos
from ..utils import over_batch
from .base import AbstractExponentiator, Order
from ..eig import op_spectral_bounds_lanczos

if TYPE_CHECKING:
    from ..operator import Operator


N_MAX = 100


class KrylovExponentiator(AbstractExponentiator):
    """
    Computes Krylov subspace approximation of exp(h * A) @ y, where A is an
    operator. Let

        A @ Q_m = Q_m @ T + β_m * q_{m + 1} @ e_m.T

    be the approximation computed after m iterations of the Lanczos algorithm,
    where Q_m is a basis for the Krylov subspace,

        K_m = span {y, A @ y, A^2 @ y, ..., A^{m - 1} @ y}

    It follows that

        exp(h * A) @ y ~= β_1 * Q @ exp(h * T) @ e_1

    References:
        1. Saad, Yousef. "Analysis of some Krylov subspace approximations to the
        matrix exponential operator."
        SIAM Journal on Numerical Analysis 29.1 (1992): 209-228.

        2. Hochbruck, Marlis, and Christian Lubich. "On Krylov subspace approximations
        to the matrix exponential operator."
        SIAM Journal on Numerical Analysis 34.5 (1997): 1911-1925.
    """

    num_iterations: int = 10
    orthogonalize: bool = False

    def adapt(self, op: Operator, dt_max: ScalarLike) -> AbstractExponentiator:
        try:
            lmin, lmax = op.spectral_bounds
        except NotImplementedError:
            lmin, lmax = op_spectral_bounds_lanczos(op)

        w = 0.5 * jnp.abs(dt_max) * (lmax - lmin)

        if jax.config.x64_enabled:
            tol = 2.0 ** -53
        else:
            tol = 2.0 ** -24

        n, term = 1, 2.0
        while term > tol and n < N_MAX:
            n += 1
            term *= (w / 2) / n

        return KrylovExponentiator(n, self.orthogonalize)

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:

        def fn(y_i):
            alpha, beta, Q, _ = lanczos(op, self.num_iterations,
                orthogonalize=self.orthogonalize, w0=y_i)

            beta0 = y_i.norm()
            eigvals, eigvecs = jax.scipy.linalg.eigh_tridiagonal(alpha, beta[:-1])
            expm = eigvecs @ (jnp.exp(h * eigvals) * eigvecs[0, :])
            return beta0 * Q.contract(expm)

        return over_batch(fn, y)

    @property
    def order(self) -> Order:
        return None

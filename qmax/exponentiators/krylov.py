from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import ScalarLike

from ..hilbert_space import AbstractHilbertSpace, AbstractState
from ..lanczos import lanczos
from ..utils import over_batch
from .base import AbstractExponentiator, Order
from ..eig import op_eigh_lanczos

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

    def adapt(
        self,
        op: Operator,
        hilbert_space: AbstractHilbertSpace,
        dt_max: ScalarLike) -> AbstractExponentiator:

        try:
            lmin, lmax = op.spectral_bounds(hilbert_space)
        except NotImplementedError:
            eigvals, _, _ = op_eigh_lanczos(op, hilbert_space, 25, 25)
            lmin, lmax = jnp.min(eigvals), jnp.max(eigvals)    

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
        hilbert_space = y.hilbert_space

        def matvec(coeffs):
            y = hilbert_space.from_coeffs(coeffs)
            Ay = op.action(y)
            return Ay.coeffs

        # The Krylov basis is built per vector -- beta0 is a per-state norm and Q
        # is sized for a single vector -- so batch axes have to be mapped over
        # rather than broadcast through.
        def fn(y_i):
            beta0 = jnp.linalg.norm(y_i.coeffs)
            w0 = y_i.coeffs
            init = (0, beta0, jnp.zeros((hilbert_space.dim, self.num_iterations + 1), dtype=complex), w0)

            alpha, beta, Q, _ = lanczos(
                matvec, hilbert_space.dim, self.num_iterations,
                orthogonalize=self.orthogonalize, return_ritz=True, return_residual=True, init=init)

            eigvals, eigvecs = jax.scipy.linalg.eigh_tridiagonal(alpha, beta[:-1])
            expm = eigvecs @ (jnp.exp(h * eigvals) * eigvecs[0, :])
            return hilbert_space.from_coeffs(beta0 * Q[:, 1:] @ expm)

        return over_batch(fn, y)

    @property
    def order(self) -> Order:
        return None

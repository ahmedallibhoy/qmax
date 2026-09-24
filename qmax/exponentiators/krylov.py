from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
from jaxtyping import ScalarLike

from .._introspect import CountDict, Path
from .._types import ComplexScalarLike
from ..eig import op_spectral_bounds_lanczos
from ..hilbert_space import AbstractState
from ..lanczos import lanczos
from ..utils import over_batch
from .base import AbstractExponentiator, Order

if TYPE_CHECKING:
    from ..operator import Operator


__all__ = ["KrylovExponentiator"]


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

    Does not support reverse mode differentiation

    ??? cite "References"

        1. Y. Saad, "Analysis of some Krylov subspace approximations to the matrix
            exponential operator," SIAM J. Numer. Anal., vol. 29, no. 1, pp. 209-228,
            1992.

        2. M. Hochbruck and C. Lubich, "On Krylov subspace approximations to the
            matrix exponential operator," SIAM J. Numer. Anal., vol. 34, no. 5,
            pp. 1911-1925, 1997.
    """

    num_iterations: int = 10
    orthogonalize: bool = False

    def adapt(self, op: Operator, dt_max: ScalarLike) -> AbstractExponentiator:
        try:
            lmin, lmax = op.spectral_bounds
        except NotImplementedError:
            lmin, lmax = op_spectral_bounds_lanczos(op)

        w = 0.5 * jnp.abs(dt_max) * (lmax - lmin)

        # unit roundoff
        tol = float(jnp.finfo(jnp.result_type(float)).eps) / 2

        n, term = 1, 2.0
        while term > tol and n < N_MAX:
            n += 1
            term *= (w / 2) / n

        return KrylovExponentiator(n, self.orthogonalize)

    def exp(self, op: Operator, h: ComplexScalarLike, y: AbstractState) -> AbstractState:

        def fn(y_i):
            alpha, beta, Q, _ = lanczos(
                op, self.num_iterations, orthogonalize=self.orthogonalize, w0=y_i
            )

            beta0 = y_i.norm()
            eigvals, eigvecs = jax.scipy.linalg.eigh_tridiagonal(alpha, beta[:-1])
            expm = eigvecs @ (jnp.exp(h * eigvals) * eigvecs[0, :])
            return beta0 * Q.contract(expm)

        return over_batch(fn, y)

    @property
    def order(self) -> Order:
        return None

    def count(
        self,
        op: Operator,
        h: ComplexScalarLike,
        parent_path: Optional[Path] = None,
        child_idx: Optional[int] = None,
    ) -> CountDict:

        return self.num_iterations * op.interface_count(parent_path, child_idx).action

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Scalar, ScalarLike

from .._introspect import CountDict, Path
from .._types import ComplexScalarLike
from ..chebyshev import chebyshev
from ..eig import op_spectral_bounds_lanczos
from ..hilbert_space import AbstractState
from .base import AbstractExponentiator, Order

if TYPE_CHECKING:
    from ..operator import Operator


__all__ = ["ChebyshevExponentiator"]


N_MAX = 100


# only works if jax_enable_x64 is True
# TODO:
#   1. raise warning on overflow
#   2. overflow safe implementation
def _modified_bessel(order: int, z: Scalar, extend: int = 25) -> Array:
    def miller(carry, idx):
        s_next, s = carry
        s_prev = s_next + (2 * idx) / z * s
        return (s, s_prev), s_prev

    idx_list = jnp.arange(1, order + extend)
    _, Is = jax.lax.scan(miller, (0j, 1j), idx_list[::-1])
    Is = Is[::-1]

    ks = jnp.arange(Is.shape[0])
    S = jnp.sum(jnp.where(ks == 0, 1.0, 2.0) * Is)
    Is = Is[:order] * jnp.exp(z) / S
    return Is


class ChebyshevExponentiator(AbstractExponentiator):
    r"""
    Chebyshev polynomial method to approximate exp(h * A) @ y. Let

        e^{s h} = \sum_{k}\mu_k(h) T_k(s)

    be an expansion of the scalar exponential in the basis of Chebyshev functions. Then
    the matrix exponential may be approximated by

        exp(h * A) = \sum_{k}\mu_k(h) * T_k(A)

    where T_k(A) satisfies the 3-term recurrence for Chebyshev polynomials

        T_{k + 1}(A) = 2A @ T_k(A) - T_{k - 1}(A).

    The series coefficients are \mu_k = 2I_k(h) and \mu_0 = I_0(h) where I_k is the
    kth modified Bessel function of the first kind. The latter are evaluated using
    the Miller recurrence algorithm.

    !!! warning

        The Miller recurrence numerically overflows in 32-bit precision. It is recommend
        to use 64-bit precision or a different exponentiation method at low precision.

    ??? cite "References"

        1. B. N. Sheehan, Y. Saad, and R. B. Sidje, "Computing exp(-τA) b with
            Laguerre polynomials," Electron. Trans. Numer. Anal., vol. 37,
            pp. 147-165, 2010.
    """

    num_iterations: int = 10

    def __check_init__(self):
        if not jax.config.read("jax_enable_x64"):
            warnings.warn(
                "ChebyshevExponentiator risks numerical overflow in single precision."
                "It is recommended to set jax_enable_x64=True or use a different exponentiator "
            )

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

        return ChebyshevExponentiator(n)

    def exp(self, op: Operator, h: ComplexScalarLike, y: AbstractState) -> AbstractState:
        lambda_min, lambda_max = op.spectral_bounds

        a, b = 0.5 * (lambda_max - lambda_min), 0.5 * (lambda_max + lambda_min)
        c = jnp.exp(b * h)
        op_scaled = (op - b) / a

        Is = _modified_bessel(self.num_iterations, h * a)
        coeffs = (2 - (jnp.arange(self.num_iterations) == 0)) * Is[: self.num_iterations]

        exp_y = c * chebyshev(op_scaled, y, coeffs)
        return exp_y

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

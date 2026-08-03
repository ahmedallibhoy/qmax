from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, ScalarLike

from ..hilbert_space import AbstractHilbertSpace, AbstractState
from .base import AbstractExponentiator, Order
from ..eig import op_eigh_lanczos

if TYPE_CHECKING:
    from ..operator import Operator


N_MAX = 100


def _polynomial_recurrence(op, y, init, coeffs, r_params):
    """
    Given p_0(A) @ y and p_1(A) @ y, computes sum_{k} c_k * p_k(A) @ y
    where {p_k} is a family of orthogonal polynomials satisfying the
    recurrence relation:

        beta_k * p_{k + 1}(x) = (x - alpha_k) * p_k(x) - gamma_k * p_{k - 1}(x)
    """

    def poly(carry, args):
        idx, coeff = args
        (v_prev, v, fv) = carry

        alpha, beta, gamma = r_params(idx)
        v_next = (op.action(v) - alpha * v  - gamma * v_prev) / beta
        fv_next = fv + coeff * v_next
        return (v, v_next, fv_next), None

    v0, v1 = init
    fv = coeffs[0] * v0 + coeffs[1] * v1

    init = (v0, v1, fv)
    idx_list = jnp.arange(2, coeffs.shape[0])
    (_, _, fv), _ = jax.lax.scan(poly, init, (idx_list, coeffs[2:]))

    return fv


# only works if jax_enable_x64 is True
# TODO: 
#   1. raise warning on overflow 
#   2. overflow safe implementation
def _modified_bessel(order: ArrayLike, z: ScalarLike, extend: int=25) -> ArrayLike:
    def miller(carry, idx):
        s_next, s = carry
        s_prev = s_next + (2 * idx) / z * s
        return (s, s_prev), s_prev

    idx_list = jnp.arange(1, order + extend)
    _, Is = jax.lax.scan(miller, (0j, 1j), idx_list[::-1])
    Is = Is[::-1]

    ks = jnp.arange(Is.shape[0])
    S  = jnp.sum(jnp.where(ks == 0, 1.0, 2.0) * Is)
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

    WARNING: The Miller recurrence numerically overflows in 32-bit precision. It is recommend
    to use 64-bit precision or a different exponentiation method at low precision.

    1. Sheehan, Bernard N., Yousef Saad, and Roger B. Sidje. "Computing exp (-τA) b with Laguerre polynomials."
    Electronic Transactions on Numerical Analysis 37 (2010): 147-165.
    """
    num_iterations: int = 10

    def __check_init__(self):
        if not jax.config.x64_enabled:
            warnings.warn(
                "ChebyshevExponentiator risks numerical overflow in single precision."
                "It is recommended to set jax_enable_x64=True or use a different exponentiator "
            )

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

        return ChebyshevExponentiator(n)

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        hilbert_space = y.hilbert_space
        lambda_min, lambda_max = op.spectral_bounds(hilbert_space)

        a, b = 0.5 * (lambda_max - lambda_min), 0.5 * (lambda_max + lambda_min)
        c = jnp.exp(b * h)
        op_scaled = (op - b) / a

        Is = _modified_bessel(self.num_iterations, h * a)
        coeffs = (2 - (jnp.arange(self.num_iterations) == 0)) * Is[:self.num_iterations]

        def r_params(idx):
            return (0.0, 0.5, 0.5)

        v0, v1 = y, op_scaled.action(y)
        p_y = _polynomial_recurrence(op_scaled, y, (v0, v1), coeffs, r_params)
        exp_y = c * p_y

        return exp_y

    @property
    def order(self) -> Order:
        return None


class LaguerreExponentiator(AbstractExponentiator):
    """
    Laguerre polynomial method to approximate exp(h * A) @ y. Not recommended
    due to numerical overflow issues.

    1. Sheehan, Bernard N., Yousef Saad, and Roger B. Sidje. "Computing exp (-τA) b with Laguerre polynomials."
    Electronic Transactions on Numerical Analysis 37 (2010): 147-165.
    """
    num_iterations: int

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        from ..operator import Identity   # avoid circular import
        hilbert_space = y.hilbert_space
        _, lambda_max = op.spectral_bounds(hilbert_space)

        op_scaled = lambda_max * Identity() - op
        idx_list = jnp.arange(self.num_iterations)
        coeffs = (h ** idx_list) / (1 + h) ** (idx_list + 1)

        def r_params(idx):
            return (2 * idx - 1, -idx, -(idx - 1))

        v0, v1 = y, y - op_scaled.action(y)
        p_y = _polynomial_recurrence(op_scaled, y, (v0, v1), coeffs, r_params)
        exp_y = jnp.exp(h * lambda_max) * p_y

        return exp_y

    @property
    def order(self) -> Order:
        return None

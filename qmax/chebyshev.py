from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from jaxtyping import Array

from .hilbert_space import AbstractState

if TYPE_CHECKING:
    from .operator import Operator


def chebyshev(A: Operator, y: AbstractState, coeffs: Array) -> AbstractState:
    """
    Computes Σ_k c_k T_k(A)y, where T_k is the kth Chebyshev polynomial
    of the first kind.
    """

    def loop(carry, coeff):
        y_prev, y, total = carry
        y_next = 2 * A(y) - y_prev
        total_next = total + coeff * y_next
        return (y, y_next, total_next), None

    y0, y1 = y, A(y)
    total = coeffs[0] * y0 + coeffs[1] * y1
    (_, _, y_total), _ = jax.lax.scan(loop, (y0, y1, total), coeffs[2:])
    return y_total

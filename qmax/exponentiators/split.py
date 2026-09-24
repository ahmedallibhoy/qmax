from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, Scalar, ScalarLike

from .._types import ComplexScalarLike
from ..hilbert_space import AbstractState
from .base import DelegatingExponentiator, Order

if TYPE_CHECKING:
    from ..operator import AddOperator


__all__ = ["AbstractSplitMethod", "Strang", "PRK_r2_s2", "PRK_r4_s6", "PRK_r6_s10"]


class AbstractSplitMethod[S: AbstractState[Any]](DelegatingExponentiator["AddOperator[S]", S]):
    """
    Exponential splitting of op = A + B into an alternating sequence of exponentials:
        exp(h(A + B)) ~ exp(a_0 hA) exp(b_0 hB)  ... exp(a_{n-1} hA) exp(b_{n-1} hB) exp(a_n hA)

    """
    nest_left: bool = eqx.field(static=True, kw_only=True, default=True)

    a: eqx.AbstractVar[np.ndarray]
    b: eqx.AbstractVar[np.ndarray]

    def __check_init__(self):
        if not (np.allclose(np.sum(self.a), 1) and np.allclose(np.sum(self.b), 1)):
            raise ValueError("Coefficient arrays a and b must sum to 1")
        if not self.a.shape[0] == self.b.shape[0] + 1:
            raise ValueError(
                f"Need len(self.a) == len(self.b) + 1 "
                f"but len(self.a)={self.a.shape[0]} and len(self.b)={self.b.shape[0]}")
        if not (np.allclose(self.a, self.a[::-1]) and np.allclose(self.b, self.b[::-1])):
            raise ValueError("self.a and self.b must be palindromic sequences")

    def schedule(self, op: AddOperator[S]) -> list[tuple[int, ComplexScalarLike, int]]:
        if self.nest_left:
            a_index, b_index = 1, 0
        else:
            a_index, b_index = 0, 1

        a = self.a
        b = self.b
        sched = [(a_index, a[0], 1)]
        for (ai, bi) in zip(a[1:], b):
            sched += [(b_index, bi, 1), (a_index, ai, 1)]
        return sched

    def exp(self, op: AddOperator[S], h: ComplexScalarLike, y: S) -> S:
        if self.nest_left:
            # We flip so that the Strang method on a nested sum ((A + B) + C) expands as
            #   exp(h/2 C)exp(h/2 B)exp(hA)exp(h/2 B)exp(h/2 C) 
            # rather than 
            #   exp(h/4 A)exp(h/2 B)exp(h/4 A)exp(hC)exp(h/4 A)exp(h/2 B)exp(h/4 A)
            B, A = op.children
        else:
            # Assumes sums are nested on the right (A + (B + (C + ...) ...))
            # This is NOT the case in general: by default A + B + C + ... nests on the left
            # We implement this for completeness, but generally one should not use nest_left=False
            A, B = op.children

        def do_step(y, coeffs):
            ai, bi = coeffs
            return A._exp(ai * h, B._exp(bi * h, y)), None

        a = self.a
        b = self.b
        y_exp, _ = jax.lax.scan(do_step, A._exp(a[0] * h, y), (a[1:], b))
        return y_exp

    @property
    def operator_type(self) -> type[AddOperator[S]]:
        from ..operator import AddOperator
        return AddOperator


class Strang(AbstractSplitMethod):
    r"""
    Strang splitting: $\exp(h(A + B)) \approx \exp(\frac{h}{2}A)\exp(hB)\exp(\frac{h}{2}A)$. 
    Equivalent to a 2nd order partitioned Runge-Kutta exponential splitting method. 
    """
    a: ClassVar[np.ndarray] = np.array([0.5, 0.5])
    b: ClassVar[np.ndarray] = np.array([1.0])

    @property
    def order(self) -> Order:
        return 2


class PRK_r2_s2(AbstractSplitMethod):
    """
    2nd order partitioned Runge-Kutta exponential splitting, c.f. Section 3.7.1 of [1].

    ??? cite "References"

        1. S. Blanes and F. Casas, A Concise Introduction to Geometric Numerical
            Integration, 2nd ed. Boca Raton, FL: CRC Press, 2025.
    """
    a: ClassVar[np.ndarray] = np.array([0.19318332750378, 0.61363334499244, 0.19318332750378])
    b: ClassVar[np.ndarray] = np.array([0.5, 0.5])

    @property
    def order(self) -> Order:
        return 2


class PRK_r4_s6(AbstractSplitMethod):
    """
    4th order partitioned Runge-Kutta exponential splitting, c.f. Table 2 of [1].

    ??? cite "References"

        1. S. Blanes and P. C. Moan, "Practical symplectic partitioned Runge-Kutta
            and Runge-Kutta-Nyström methods," J. Comput. Appl. Math., vol. 142,
            no. 2, pp. 313-330, 2002.
    """
    a: ClassVar[np.ndarray] = np.array(
        [0.0792036964311956, 0.353172906049774, -0.0420650803577195, 0.2193769557534997,
         -0.0420650803577195, 0.353172906049774, 0.0792036964311956])
    b: ClassVar[np.ndarray] = np.array(
        [0.209515106613362, -0.1438517731798181, 0.4343366665664561,
         0.4343366665664561, -0.1438517731798181, 0.209515106613362])

    @property
    def order(self) -> Order:
        return 4


class PRK_r6_s10(AbstractSplitMethod):
    """
    6th order partitioned Runge-Kutta exponential splitting, c.f. Table 2 of [1].

    ??? cite "References"

        1. S. Blanes and P. C. Moan, "Practical symplectic partitioned Runge-Kutta
            and Runge-Kutta-Nyström methods," J. Comput. Appl. Math., vol. 142,
            no. 2, pp. 313-330, 2002.
    """
    a: ClassVar[np.ndarray] = np.array(
        [0.0502627644003922, 0.413514300428344, 0.0450798897943977, -0.188054853819569,
         0.541960678450780, -0.7255255585086897, 0.541960678450780, -0.188054853819569,
         0.0450798897943977, 0.413514300428344, 0.0502627644003922])
    b: ClassVar[np.ndarray] = np.array(
        [0.148816447901042, -0.132385865767784, 0.067307604692185, 0.432666402578175,
         -0.0164045894036180, -0.0164045894036180, 0.432666402578175, 0.067307604692185,
         -0.132385865767784, 0.148816447901042])

    @property
    def order(self) -> Order:
        return 6


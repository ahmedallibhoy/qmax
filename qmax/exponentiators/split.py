from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, Scalar, ScalarLike

from ..hilbert_space import AbstractState
from .base import Order, DelegatingExponentiator

if TYPE_CHECKING:
    from ..operator import AddOperator


__all__ = ["AbstractSplitMethod", "Strang", "PRK_r2_s2", "PRK_r4_s6", "PRK_r6_s10"]


class AbstractSplitMethod(DelegatingExponentiator):
    """
    Exponential splitting of op = A + B into an alternating sequence of exponentials:
        exp(h(A + B)) ~ exp(a_0 hA) exp(b_0 hB)  ... exp(a_{n-1} hA) exp(b_{n-1} hB) exp(a_n hA)

    """
    nest_left: bool = eqx.field(static=True, kw_only=True, default=True)

    @property
    @abstractmethod
    def a(self) -> Array:
        pass

    @property
    @abstractmethod
    def b(self) -> Array:
        pass

    def __check_init__(self):
        if not (jnp.allclose(jnp.sum(self.a), 1) and jnp.allclose(jnp.sum(self.b), 1)):
            raise ValueError("Coefficient arrays a and b must sum to 1")
        if not self.a.shape[0] == self.b.shape[0] + 1:
            raise ValueError(
                f"Need len(self.a) == len(self.b) + 1 "
                f"but len(self.a)={self.a.shape[0]} and len(self.b)={self.b.shape[0]}")
        if not (jnp.allclose(self.a, self.a[::-1]) and jnp.allclose(self.b, self.b[::-1])):
            raise ValueError(f"self.a and self.b must be palindromic sequences")

    def schedule(self, op: AddOperator) -> list[tuple[int, Scalar, int]]:
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

    def exp(self, add_op: AddOperator, h: ScalarLike, y: AbstractState) -> AbstractState:        
        if self.nest_left:
            # We flip so that the Strang method on a nested sum ((A + B) + C) expands as
            #   exp(h/2 C)exp(h/2 B)exp(hA)exp(h/2 B)exp(h/2 C) 
            # rather than 
            #   exp(h/4 A)exp(h/2 B)exp(h/4 A)exp(hC)exp(h/4 A)exp(h/2 B)exp(h/4 A)
            B, A = add_op.children
        else:
            # Assumes sums are nested on the right (A + (B + (C + ...) ...))
            # This is NOT the case in general: by default A + B + C + ... nests on the left
            A, B = add_op.children

        def do_step(y, coeffs):
            ai, bi = coeffs
            return A._exp(ai * h, B._exp(bi * h, y)), None

        a = self.a
        b = self.b
        y_exp, _ = jax.lax.scan(do_step, A._exp(a[0] * h, y), (a[1:], b))
        return y_exp

    @property
    def operator_type(self) -> type:
        from ..operator import AddOperator
        return AddOperator


class Strang(AbstractSplitMethod):
    a: ClassVar[Array] = jnp.array([0.5, 0.5])
    b: ClassVar[Array] = jnp.array([1.0])

    @property
    def order(self) -> Order:
        return 2


class PRK_r2_s2(AbstractSplitMethod):
    """
    2nd order partitioned Runge-Kutta exponential splitting, c.f. Section 3.7.1 from:

        A Concise Introduction to Geometric Numerical Integration (2nd ed.).
        Blanes, Sergio, and Fernando Casas. Chapman and Hall/CRC. 2025.
    """
    a: ClassVar[Array] = jnp.array([0.19318332750378, 0.61363334499244, 0.19318332750378])
    b: ClassVar[Array] = jnp.array([0.5, 0.5])

    @property
    def order(self) -> Order:
        return 2


class PRK_r4_s6(AbstractSplitMethod):
    """
    4th order partitioned Runge-Kutta exponential splitting, c.f. Table 2 from:

        Practical symplectic partitioned Runge–Kutta and Runge–Kutta–Nyström methods.
        Blanes, Sergio, and Per Christian Moan.
        Journal of Computational and Applied Mathematics 142.2 (2002): 313-330.
    """
    a: ClassVar[Array] = jnp.array(
        [0.0792036964311956, 0.353172906049774, -0.0420650803577195, 0.2193769557534997,
         -0.0420650803577195, 0.353172906049774, 0.0792036964311956])
    b: ClassVar[Array] = jnp.array(
        [0.209515106613362, -0.1438517731798181, 0.4343366665664561,
         0.4343366665664561, -0.1438517731798181, 0.209515106613362])

    @property
    def order(self) -> Order:
        return 4


class PRK_r6_s10(AbstractSplitMethod):
    """
    6th order partitioned Runge-Kutta exponential splitting, c.f. Table 2 from:

        Practical symplectic partitioned Runge–Kutta and Runge–Kutta–Nyström methods.
        Blanes, Sergio, and Per Christian Moan.
        Journal of Computational and Applied Mathematics 142.2 (2002): 313-330.
    """
    a: ClassVar[Array] = jnp.array(
        [0.0502627644003922, 0.413514300428344, 0.0450798897943977, -0.188054853819569,
         0.541960678450780, -0.7255255585086897, 0.541960678450780, -0.188054853819569,
         0.0450798897943977, 0.413514300428344, 0.0502627644003922])
    b: ClassVar[Array] = jnp.array(
        [0.148816447901042, -0.132385865767784, 0.067307604692185, 0.432666402578175,
         -0.0164045894036180, -0.0164045894036180, 0.432666402578175, 0.067307604692185,
         -0.132385865767784, 0.148816447901042])

    @property
    def order(self) -> Order:
        return 6


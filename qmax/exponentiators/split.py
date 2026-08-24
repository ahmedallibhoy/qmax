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


class AbstractSplitMethod(DelegatingExponentiator):
    """
    Exponential splitting of op = A + B into an alternating sequence of exponentials:

        exp(h(A + B)) ~ exp(a_0 hA) exp(b_0 hB)  ... exp(a_{n -1} hA) exp(b_{n-1} hB) exp(a_n hA)

    where a, b = self._coeffs
    """

    @property
    @abstractmethod
    def _coeffs(self) -> tuple[Array, Array]:
        pass

    def schedule(self, op: AddOperator) -> list[tuple[int, Scalar, int]]:
        a, b = self._coeffs
        sched = [(0, a[0], 1)]
        for (ai, bi) in zip(a[1:], b):
            sched += [(1, bi, 1), (0, ai, 1)]
        return sched

    def exp(self, add_op: AddOperator, h: ScalarLike, y: AbstractState) -> AbstractState:
        A, B = add_op.children
        a, b = self._coeffs

        def do_step(y, coeffs):
            ai, bi = coeffs
            return A.exp(ai * h, B.exp(bi * h, y)), None

        y_exp, _ = jax.lax.scan(do_step, A.exp(a[0] * h, y), (a[1:], b))
        return y_exp

    @property
    def operator_type(self) -> type:
        from ..operator import AddOperator
        return AddOperator


class Strang(AbstractSplitMethod):

    @property
    def _coeffs(self) -> tuple[Array, Array]:
        return jnp.array([0.5, 0.5]), jnp.array([1.0])

    @property
    def order(self) -> Order:
        return 2


class Yoshida(AbstractSplitMethod):
    """
    Yoshida triple-jump exponential splitting, see e.g. [1], and [2, Example 4.2]

        1. Yoshida, Haruo. "Construction of higher order symplectic integrators."
           Physics letters A 150.5-7 (1990): 262-268.

        2. Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations
           Hairer, Ernst and Lubich, Christian and Wanner, Gerhard. Springer-Verlag, 2006
    """

    level: int = eqx.field(static=True)

    @property
    def _coeffs(self) -> tuple[Array, Array]:

        def iterate(k, h, seq):
            if k == 0:
                return seq + [(0, 0.5 * h), (1, h), (0, 0.5 * h)]

            w1 = 1 / (2 - 2 ** (1 / (2 * k + 1)))
            w2 = 1 - 2 * w1
            return iterate(k - 1, w1 * h, iterate(k - 1, w2 * h, iterate(k - 1, w1 * h, seq)))

        fused = []
        for idx, c in iterate(self.level, 1.0, []):
            if fused and fused[-1][0] == idx:
                fused[-1] = (idx, fused[-1][1] + c)
            else:
                fused.append((idx, c))

        return (jnp.array([c for idx, c in fused if idx == 0]),
                jnp.array([c for idx, c in fused if idx == 1]))

    @property
    def order(self) -> Order:
        return 2 * (self.level + 1)


class AbstractPRKSplitMethod(AbstractSplitMethod):
    """
    Abstract class for partitioned Runge-Kutta exponential splitting, c.f. (7) from

        Practical symplectic partitioned Runge–Kutta and Runge–Kutta–Nyström methods.
        Blanes, Sergio, and Per Christian Moan.
        Journal of Computational and Applied Mathematics 142.2 (2002): 313-330.
    """

    a: eqx.AbstractClassVar[Array]
    b: eqx.AbstractClassVar[Array]

    @property
    def _coeffs(self) -> tuple[Array, Array]:
        a = jnp.concatenate([self.a, jnp.array([1 - 2 * self.a.sum()]), self.a[::-1]])
        b_half = jnp.concatenate([self.b, jnp.array([0.5 - self.b.sum()])])
        return a, jnp.concatenate([b_half, b_half[::-1]])


class BlanesMoan6(AbstractPRKSplitMethod):
    """
    6th order partitioned Runge-Kutta exponential splitting, c.f. Table 2 from:

        Practical symplectic partitioned Runge–Kutta and Runge–Kutta–Nyström methods.
        Blanes, Sergio, and Per Christian Moan.
        Journal of Computational and Applied Mathematics 142.2 (2002): 313-330.
    """

    a: ClassVar[Array] = jnp.array(
        [0.0502627644003922, 0.413514300428344, 0.0450798897943977, -0.188054853819569, 0.541960678450780]
    )

    b: ClassVar[Array] = jnp.array(
        [0.148816447901042, -0.132385865767784, 0.067307604692185, 0.432666402578175]
    )

    @property
    def order(self) -> Order:
        return 6

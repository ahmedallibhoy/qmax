from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import ScalarLike, Array

from ..hilbert_space import AbstractState
from .base import Order, min_order, AbstractExponentiator, NotExponentiableError

if TYPE_CHECKING:
    from ..operator import Operator, AddOperator

class AbstractSplitMethod(AbstractExponentiator):
    
    def effective_order(self, add_op: AddOperator) -> Order:
        return min_order(
            self.order, add_op.op1.exp_order, add_op.op2.exp_order)

    def check_exponentiable(self, op: Operator):
        from ..operator import AddOperator

        if not isinstance(op, AddOperator):
            raise NotExponentiableError(
                f"{type(self).__name__} can only exponentiate operators of type AddOperator "
                f"but received operator of type {type(op).__name__}"
            )

        op.op1.check_exponentiable_tree()
        op.op2.check_exponentiable_tree()

    @property
    @abstractmethod
    def h_scales(self):
        pass


class Strang(AbstractSplitMethod):

    def exp(self, add_op: AddOperator, h: ScalarLike, y: AbstractState) -> AbstractState:
        return add_op.op1.exp(h / 2, add_op.op2.exp(h, add_op.op1.exp(h / 2, y)))

    @property
    def order(self) -> Order:
        return 2

    @property
    def h_scales(self):
        return 0.5, 1.0


class Yoshida(AbstractSplitMethod):
    """
    Yoshida triple-jump exponential splitting, see e.g. [1], and [2, Example 4.2]

        1. Yoshida, Haruo. "Construction of higher order symplectic integrators."
           Physics letters A 150.5-7 (1990): 262-268.

        2. Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations
           Hairer, Ernst and Lubich, Christian and Wanner, Gerhard. Springer-Verlag, 2006
    """

    level: int = eqx.field(static=True)

    def exp(self, add_op: AddOperator, h: ScalarLike, y: AbstractState) -> AbstractState:

        def iterate(k, h, y):
            if k == 0:
                return add_op.op1.exp(h / 2, add_op.op2.exp(h, add_op.op1.exp(h / 2, y)))

            w1 = 1 / (2 - 2 ** (1 / (2 * k + 1)))
            w2 = 1 - 2 * w1
            return iterate(k - 1, w1 * h, iterate(k - 1, w2 * h, iterate(k - 1, w1 * h, y)))

        return iterate(self.level, h, y)

    @property
    def order(self) -> Order:
        return 2 * (self.level + 1)

    @property
    def h_scales(self):
        m = 1.0
        for k in range(1, self.level + 1):
            w1 = 1 / (2 - 2 ** (1 / (2 * k + 1)))
            m *= max(abs(w1), abs(1 - 2 * w1))
        return 0.5 * m, m


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
    def _coeffs(self):
        a = jnp.concatenate([self.a, jnp.array([1 - 2 * self.a.sum()]), self.a[::-1]])
        b_half = jnp.concatenate([self.b, jnp.array([0.5 - self.b.sum()])])
        return a, jnp.concatenate([b_half, b_half[::-1]])

    def exp(self, add_op: AddOperator, h: ScalarLike, y: AbstractState) -> AbstractState:
        a, b = self._coeffs

        def do_step(y, coeffs):
            ai, bi = coeffs
            y_next = add_op.op1.exp(ai * h, add_op.op2.exp(bi * h, y))
            return y_next, None

        y_init = add_op.op1.exp(a[0] * h, y)
        y_exp, _ = jax.lax.scan(do_step, y_init, (a[1:], b))
        return y_exp

    @property
    def h_scales(self):
        a, b = self._coeffs
        return float(jnp.abs(a).max()), float(jnp.abs(b).max())


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

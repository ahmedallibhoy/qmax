from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import ScalarLike, Array

from .._internal import _update_fields
from .._introspect import CountDict, Path, Field
from ..hilbert_space import AbstractHilbertSpace, AbstractState
from .base import Order, min_order, AbstractExponentiator

if TYPE_CHECKING:
    from ..operator import Operator, AddOperator


class AbstractSplitMethod(AbstractExponentiator):

    def adapt_children(self, op: AddOperator, dt_max: ScalarLike) -> Operator:
        h1, h2 = self.h_scales
        return _update_fields(
            op,
            op1=op.op1.adapt(h1 * dt_max),
            op2=op.op2.adapt(h2 * dt_max),
        )

    def effective_order(self, add_op: AddOperator) -> Order:
        return min_order(
            self.order, add_op.op1.tree_order, add_op.op2.tree_order)

    @property
    def operator_type(self) -> type:
        from ..operator import AddOperator
        return AddOperator

    def check_exponentiable(
        self, 
        op: Operator, 
        parent_path: Path=Path(), 
        field: Field=Field()):

        path = op.path(parent_path, field)
        op.op1.check_exponentiable_tree(path, Field("op1"))
        op.op2.check_exponentiable_tree(path, Field("op2"))

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

    def count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Path=Path(), 
        field: Field=Field()) -> CountDict:

        path = op.path(parent_path, field)
        h1, h2 = self.h_scales
        c1 = 2 * op.op1.exp_count(h1 * h, path, Field("op1"))
        c2 = op.op2.exp_count(h2 * h, path, Field("op2"))
        return c1 + c2

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

    def count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Path=Path(), 
        field: Field=Field()) -> CountDict:

        c = CountDict()
        path = op.path(parent_path, field)

        def iterate(k, h, c):
            if k == 0:
                c1 = 2 * op.op1.exp_count(h / 2, path, Field("op1"))
                c2 = op.op2.exp_count(h, path, Field("op2"))
                return c + c1 + c2

            w1 = 1 / (2 - 2 ** (1 / (2 * k + 1)))
            w2 = 1 - 2 * w1
            return iterate(k - 1, w1 * h, iterate(k - 1, w2 * h, iterate(k - 1, w1 * h, c)))

        return iterate(self.level, h, c)

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

    def count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Path=Path(), 
        field: Field=Field()) -> CountDict:

        path = op.path(parent_path, field)
        a, b = self._coeffs
        c = op.op1.exp_count(a[0] * h, path, Field("op1"))
        for (ai, bi) in zip(a[1:], b):
            c += op.op1.exp_count(ai * h, path, Field("op1")) + op.op2.exp_count(bi * h, path, Field("op2"))
        return c

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

from __future__ import annotations

from abc import abstractmethod
from typing import Callable, ClassVar, Optional, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike, Scalar, ScalarLike

from .operator import Operator

if TYPE_CHECKING:
    from .timevarying_operator import AbstractTimeVaryingOperator


class AbstractControl(eqx.Module):

    def __call__(self, t: ScalarLike) -> Scalar:
        return self.evaluate(t)

    def __mul__(self, op: Operator) -> AbstractTimeVaryingOperator:
        if not isinstance(op, Operator):
            return NotImplemented

        from .timevarying_operator import ConstantTimeVaryingOperator
        return self * ConstantTimeVaryingOperator(op)

    def __rmul__(self, op: Operator) -> AbstractTimeVaryingOperator:
        if not isinstance(op, Operator):
            return NotImplemented

        from .timevarying_operator import ConstantTimeVaryingOperator
        return self * ConstantTimeVaryingOperator(op)

    @abstractmethod
    def evaluate(self, t: ScalarLike) -> Scalar:
        pass


class ControlFunction(AbstractControl):
    cntrl: Callable[[ScalarLike], Scalar]

    def evaluate(self, t: ScalarLike) -> Scalar:
        return self.cntrl(t)


class ConstantControl(AbstractControl):
    u: Scalar = eqx.field(static=True, converter=float)

    def evaluate(self, t: ScalarLike) -> Scalar:
        return self.u


type CanMultiply = AbstractInterpolatedControl | ScalarLike | Operator


class AbstractInterpolatedControl(AbstractControl):
    u_range: ArrayLike
    t0: Scalar = eqx.field(static=True, converter=float)
    t1: Scalar = eqx.field(static=True, converter=float)

    @property
    def num_steps(self) -> int:
        return self.u_range.shape[0]

    @property
    def dt(self) -> Scalar:
        return (self.t1 - self.t0) / (self.num_steps - 1)

    def idx(self, t: ScalarLike) -> int:
        return jnp.clip(jnp.trunc((t - self.t0) / self.dt).astype(int), 0, self.num_steps - 2)

    def binary_op(self, other: AbstractInterpolatedControl, func: Callable) -> AbstractInterpolatedControl:
        if not isinstance(other, AbstractInterpolatedControl):
            return NotImplemented

        if not type(self) == type(other):
            raise ValueError(
                f"Only controls of the same type may be combined but "
                f"type(u1)={type(self).__name__} and type(u2)={type(other).__name__}")

        if not (jnp.allclose(self.t0, other.t0) and jnp.allclose(self.t1, other.t1)):
            raise ValueError(
                f"Only controls defined on the same interval may be combined but "
                f"u1 is defined on ({self.t0}, {self.t1}) and u2 is defined on ({other.t0}, {other.t1})")

        return type(self)(func(self.u_range, other.u_range), self.t0, self.t1)
        
    def __add__(self, other: AbstractInterpolatedControl) -> AbstractInterpolatedControl:
        return self.binary_op(other, lambda a, b: a + b)

    def __sub__(self, other: AbstractInterpolatedControl) -> AbstractInterpolatedControl:
        return self.binary_op(other, lambda a, b: a - b)

    def __mul__(self, other: CanMultiply) -> AbstractInterpolatedControl | AbstractTimeVaryingOperator:
        if isinstance(other, AbstractInterpolatedControl):
            return self.binary_op(other, lambda a, b: a * b)

        if jnp.isscalar(other):
            return type(self)(other * self.u_range, self.t0, self.t1)

        return super().__mul__(other)

    def __rmul__(self, other: CanMultiply) -> AbstractInterpolatedControl | AbstractTimeVaryingOperator:
        if jnp.isscalar(other):
            return type(self)(other * self.u_range, self.t0, self.t1)

        return super().__rmul__(other)

    def __truediv__(self, other: ScalarLike) -> AbstractInterpolatedControl:
        if jnp.isscalar(other):
            return type(self)(self.u_range / other, self.t0, self.t1)

        return NotImplemented


class PiecewiseConstantControl(AbstractInterpolatedControl):

    def evaluate(self, t: ScalarLike) -> Scalar:
        idx = self.idx(t)
        return self.u_range[idx]


class PiecewiseLinearControl(AbstractInterpolatedControl):

    def evaluate(self, t: ScalarLike) -> Scalar:
        idx = self.idx(t)
        t_prev = self.t0 + self.dt * idx 
        t_next = self.t0 + self.dt * (idx + 1)
        u_prev = self.u_range[idx]
        u_next = self.u_range[idx + 1]
        return u_prev + (t - t_prev) * (u_next - u_prev) / (t_next - t_prev)

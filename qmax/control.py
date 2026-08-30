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
    has_integral: ClassVar[bool] = False 

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

    def bound(self, t_range: ArrayLike) -> Scalar:
        return jnp.max(jnp.abs(jax.vmap(self.evaluate)(t_range)))

    def integral(self, t: ScalarLike) -> Scalar:
        raise NotImplementedError


class ControlFunction(AbstractControl):
    cntrl: Callable[[ScalarLike], Scalar]
    cntrl_int: Optional[Callable[[ScalarLike], Scalar]] = eqx.field(default=None)

    @property
    def has_integral(self):
        return self.cntrl_int is not None

    def evaluate(self, t: ScalarLike) -> Scalar:
        return self.cntrl(t)

    def bound(self, t_range: ArrayLike) -> Scalar:
        u_range = jax.vmap(self.evaluate)(t_range)
        return jnp.max(jnp.abs(u_range))

    def integral(self, t: ScalarLike) -> Scalar:
        return self.cntrl_int(t)


class InterpolatedControl(AbstractControl):
    u_range: ArrayLike
    t0: ScalarLike = eqx.field(static=True)
    t1: ScalarLike = eqx.field(static=True)
    has_integral: ClassVar[bool] = True 

    @property
    def num_steps(self) -> int:
        return self.u_range.shape[0]

    @property
    def dt(self) -> Scalar:
        return (self.t1 - self.t0) / (self.num_steps - 1)

    def idx(self, t: ScalarLike) -> int:
        return jnp.clip(jnp.trunc((t - self.t0) / self.dt).astype(int), 0, self.num_steps - 2)

    def evaluate(self, t: ScalarLike) -> Scalar:
        idx = self.idx(t)
        t_prev = self.t0 + self.dt * idx 
        t_next = self.t0 + self.dt * (idx + 1)
        u_prev = self.u_range[idx]
        u_next = self.u_range[idx + 1]
        return u_prev + (t - t_prev) * (u_next - u_prev) / (t_next - t_prev)

    def bound(self, t_range: ArrayLike) -> Scalar:
        if t_range[0] < self.t0 or t_range[-1] > self.t1:
            raise ValueError(f"t_range={t_range} out of interpolation range ({self.t0}, {self.t1})")

        idx0, idx1 = self.idx(t_range[0]), self.idx(t_range[-1]) + 1
        return jnp.max(jnp.abs(self.u_range[idx0:idx1 + 1]))

    def integral(self, t: ScalarLike) -> Scalar:
        u_sum = 0.5 * self.dt * (self.u_range[:-1] + self.u_range[1:])
        u_int = jnp.concatenate([[0.], jnp.cumsum(u_sum)])
        
        idx = self.idx(t)
        t_prev = self.t0 + self.dt * idx 
        u_prev = self.u_range[idx]
        u = self.evaluate(t)
        return u_int[idx] + 0.5 * (t - t_prev)  * (u_prev + u)


class ConstantControl(AbstractControl):
    u: Scalar
    t0: Scalar = eqx.field(kw_only=True, default=0.0)

    def evaluate(self, t: ScalarLike) -> Scalar:
        return self.u

    def integral(self, t: ScalarLike) -> Scalar:
        return (t - self.t0) * self.u


class PiecewiseConstantControl(AbstractControl):
    u_range: ArrayLike
    t0: ScalarLike = eqx.field(static=True)
    t1: ScalarLike = eqx.field(static=True)
    has_integral: ClassVar[bool] = True 

    @property
    def num_steps(self) -> int:
        return self.u_range.shape[0]

    @property
    def dt(self) -> Scalar:
        return (self.t1 - self.t0) / (self.num_steps - 1)

    def idx(self, t: ScalarLike) -> int:
        return jnp.clip(jnp.trunc((t - self.t0) / self.dt).astype(int), 0, self.num_steps - 2)

    def evaluate(self, t: ScalarLike) -> Scalar:
        idx = self.idx(t)
        return self.u_range[idx]

    def bound(self, t_range: ArrayLike) -> Scalar:
        if t_range[0] < self.t0 or t_range[-1] > self.t1:
            raise ValueError(f"t_range={t_range} out of interpolation range ({self.t0}, {self.t1})")

        idx0, idx1 = self.idx(t_range[0]), self.idx(t_range[-1]) + 1
        return jnp.max(jnp.abs(self.u_range[idx0:idx1 + 1]))

    def integral(self, t: ScalarLike) -> Scalar:
        idx = self.idx(t)
        return jnp.cumsum(self.dt * self.u_range[:idx])

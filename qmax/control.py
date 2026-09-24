from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._types import RealScalarLike
from .operator import Operator

if TYPE_CHECKING:
    from .timevarying_operator import AbstractTimeVaryingOperator


# TODO
#   1. Add FourierControl


class AbstractControl(eqx.Module):
    """
    Abstract base class for controls.
    """

    def __call__(self, t: RealScalarLike) -> RealScalarLike:
        """
        Args:
            t (Scalar): time at which to evaluate the control

        Returns:
            control input evaluated at time `t`
        """
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
    def evaluate(self, t: RealScalarLike) -> RealScalarLike:
        pass


class ControlFunction(AbstractControl):
    """
    Control input which corresponds to directly evaluating a function `cntrl`

    Attributes:
        cntrl (callable): Function with signature `cntrl(t)` returning a scalar
            which is the control at time t.
    """

    cntrl: Callable[[RealScalarLike], RealScalarLike]

    def evaluate(self, t: RealScalarLike) -> RealScalarLike:
        return self.cntrl(t)


class ConstantControl(AbstractControl):
    u: float = eqx.field(static=True, converter=float)

    def evaluate(self, t: RealScalarLike) -> RealScalarLike:
        return self.u


class AbstractInterpolatedControl(AbstractControl):
    """
    Abstract base class for interpolated controls
    """

    t0: float = eqx.field(static=True, converter=float)
    t1: float = eqx.field(static=True, converter=float)
    u_range: Array

    @classmethod
    def from_function(
        cls,
        u_func: Callable[[RealScalarLike], RealScalarLike],
        t0: RealScalarLike,
        t1: RealScalarLike,
        num_samples: int,
    ) -> AbstractInterpolatedControl:
        """
        Creates an instance of the interpolated control object from a callable
        by sampling it at `num_samples` evenly spaced points in the interval [`t0`, `t1`].

        Args:
            u_func (callable): Control function to sample
            t0 (Scalar): initial time of interpolation interval
            t1 (Scalar): initial time of interpolation interval
            num_samples (int): number of sample points
        """

        t_range = jnp.linspace(t0, t1, num_samples)
        u_range = jnp.asarray(jax.vmap(u_func)(t_range))
        return cls(t0, t1, u_range)

    @property
    def num_steps(self) -> int:
        return self.u_range.shape[0]

    @property
    def dt(self) -> RealScalarLike:
        return (self.t1 - self.t0) / (self.num_steps - 1)

    @property
    def t_range(self) -> Array:
        return jnp.linspace(self.t0, self.t1, self.num_steps)

    def idx(self, t: RealScalarLike) -> RealScalarLike:
        return jnp.clip(jnp.trunc((t - self.t0) / self.dt).astype(int), 0, self.num_steps - 2)


class PiecewiseConstantControl(AbstractInterpolatedControl):
    """
    Piecewise constant interpolation of control values at evenly spaced points
    on the interval [t0, t1]

    Attributes:
        t0 (Scalar): initial time of interpolation interval
        t1 (Scalar): final time of interpolation interval
        u_range (Array): Array of values to interpolate between
    """

    def evaluate(self, t: RealScalarLike) -> RealScalarLike:
        idx = self.idx(t)
        return self.u_range[idx]


class PiecewiseLinearControl(AbstractInterpolatedControl):
    """
    Piecewise linear interpolation of control values at evenly spaced points
    on the interval [t0, t1]

    Attributes:
        t0 (Scalar): initial time of interpolation interval
        t1 (Scalar): final time of interpolation interval
        u_range (Array): Array of values to interpolate between
    """

    def evaluate(self, t: RealScalarLike) -> RealScalarLike:
        idx = self.idx(t)
        t_prev = self.t0 + self.dt * idx
        t_next = self.t0 + self.dt * (idx + 1)
        u_prev = self.u_range[idx]
        u_next = self.u_range[idx + 1]
        return u_prev + (t - t_prev) * (u_next - u_prev) / (t_next - t_prev)

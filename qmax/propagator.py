from typing import ClassVar, Optional, Callable
from abc import abstractmethod
from math import ceil

import equinox as eqx
import jax 
import jax.numpy as jnp

from jaxtyping import Scalar, ScalarLike, PyTree, Array, ArrayLike

from .hilbert_space import AbstractState, AbstractHilbertSpace
from .operator import Operator
from .control import AbstractControl, ControlFunction
from .timestepper import AbstractTimeStepper, Midpoint
from .split import AbstractSplitMethod, Strang
from .spaces.spatial_discretization import SpatialDiscretization
from .tensor import AbstractTensorOperator
from .adapt import adapt


class PropagateResult(eqx.Module):
    y1: AbstractState
    ys: PyTree
    ts: ArrayLike


def _save_y(t, y): 
    return y


class AbstractPropagator(eqx.Module):
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True) 
    hilbert_space: AbstractHilbertSpace
    num_steps: int = eqx.field(static=True)
    hbar: float = eqx.field(default=1.0, kw_only=True)
    timestepper: AbstractTimeStepper = eqx.field(default=Midpoint(), kw_only=True)

    @property
    def weights(self) -> Array:
        return self.timestepper.weights

    @property
    def quad_rule(self) -> Array:
        return self.timestepper.quad_rule

    @property
    def dt(self) -> Scalar:
        return (self.t1 - self.t0) / self.num_steps

    @abstractmethod
    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState, 
        args: PyTree) -> AbstractState:

        pass

    def propagate(
        self, 
        y: AbstractState, 
        params: Optional[PyTree]=None,
        *,
        save_every: Optional[int] = None, 
        save_fn: Callable[[ScalarLike, AbstractState], PyTree] = _save_y) -> PropagateResult:

        if save_every is None:
            save_every = 1

        if not self.num_steps % save_every == 0:
            raise ValueError(f"num_steps={self.num_steps} is not divisible by save_every={save_every}")

        t_range = jnp.linspace(self.t0, self.t1, 
            self.num_steps // save_every + 1, endpoint=True)

        def inner_loop(y, t):
            y_next = self.propagate_stage(t, self.dt, y, params)
            return y_next, None

        def loop(y, args):
            t, t_next = args
            y_next, _ = jax.lax.scan(inner_loop, y, jnp.linspace(t, t_next, save_every, endpoint=False))
            return y_next, save_fn(t_next, y_next)

        y1, ys = jax.lax.scan(loop, y, (t_range[:-1], t_range[1:]))
        return PropagateResult(y1, ys, t_range[1:])


class TimeInvariantPropagator(AbstractPropagator):
    op: Operator

    def __init__(
        self, 
        t0: ScalarLike, 
        t1: ScalarLike, 
        op: Operator,
        hilbert_space: AbstractHilbertSpace,
        *,
        dt_max: Optional[ScalarLike]=None, 
        hbar: ScalarLike=1.0,
        timestepper: AbstractTimeStepper = Midpoint()):

        self.t0 = t0
        self.t1 = t1

        if dt_max is None:
            self.num_steps = 1
        else:
            self.num_steps = ceil((t1 - t0) / dt_max)

        self.timestepper = timestepper
        self.hbar = hbar

        h = self.dt / self.hbar * jnp.max(jnp.abs(jnp.sum(self.weights, axis=1)))
        self.op = adapt_operator(op, hilbert_space, h)
        self.hilbert_space = hilbert_space

    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState, 
        params: Optional[PyTree]=None) -> AbstractState:

        y_next = y

        for i in range(self.weights.shape[0]):
            w = jnp.sum(self.weights[i, :])
            y_next = self.op.exp((-1j / self.hbar) * w * dt, y_next)

        return y_next


class ControlledPropagator(AbstractPropagator):
    op1: Operator
    op2: Operator
    u1_max: Scalar
    u2_max: Scalar
    split_method: AbstractSplitMethod

    def __init__(
        self, 
        t0: ScalarLike, 
        t1: ScalarLike, 
        op1: Operator,
        op2: Operator,
        u1_max: Scalar, 
        u2_max: Scalar,
        hilbert_space: AbstractHilbertSpace,
        *,
        dt_max: Optional[ScalarLike]=None, 
        hbar: ScalarLike=1.0,
        timestepper: AbstractTimeStepper = Midpoint(), 
        split_method: AbstractSplitMethod = Strang()):

        self.t0 = t0
        self.t1 = t1

        if dt_max is None:
            self.num_steps = 1
        else:
            self.num_steps = ceil((t1 - t0) / dt_max)

        self.timestepper = timestepper
        self.hbar = hbar

        self.u1_max = u1_max 
        self.u2_max = u2_max 

        w_max = jnp.max(jnp.sum(jnp.abs(self.weights), axis=1))  
        h1, h2 = split_method.h_scales
        dt1 = h1 * w_max * self.u1_max * self.dt / self.hbar
        dt2 = h2 * w_max * self.u2_max * self.dt / self.hbar

        self.op1 = adapt_operator(op1, hilbert_space, dt1)
        self.op2 = adapt_operator(op2, hilbert_space, dt2)
        self.split_method = split_method
        self.hilbert_space = hilbert_space

    def coeffs(self, t: ScalarLike, dt: ScalarLike, u: AbstractControl) -> Array:
        t_quad, w_quad = self.quad_rule
        u_quad = jax.vmap(u)(t + dt * t_quad)
        coeffs = self.weights @ u_quad
        if not u.has_integral:
            return coeffs 
        c_bar = (u.integral(t + dt) - u.integral(t)) / dt
        return coeffs + jnp.sum(self.weights, axis=1) * (c_bar - w_quad @ u_quad)

    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState, 
        us: tuple[AbstractControl, AbstractControl]) -> AbstractState:

        u1, u2 = us
        c1_coeffs = self.coeffs(t, dt, u1)
        c2_coeffs = self.coeffs(t, dt, u2)
        y_next = y

        for i in range(self.weights.shape[0]):
            y_next = self.split_method.exp(
                c1_coeffs[i] * self.op1 + c2_coeffs[i] * self.op2, (-1j / self.hbar) * dt, y_next)

        return y_next


class QuantumHamiltonianDescent(ControlledPropagator):
    u1: AbstractControl
    u2: AbstractControl

    def __init__(
        self, 
        t0: ScalarLike, 
        t1: ScalarLike, 
        objective: Callable[[ArrayLike], ScalarLike], 
        hilbert_space: SpatialDiscretization,
        *,
        dt_max: Optional[ScalarLike]=None, 
        split_method=Strang(), 
        timestepper=Midpoint()):

        self.u1 = ControlFunction(lambda t: 1.0 / t, lambda t: jnp.log(t))
        self.u2 = ControlFunction(lambda t: t, lambda t: t ** 2 / 2)

        num_steps = ceil((t1 - t0) / dt_max)
        t_range = jnp.linspace(t0, t1, num_steps + 1)
        t_eval = timestepper.eval_points(t_range).flatten()
        u1_max = self.u1.bound(t_eval)
        u2_max = self.u2.bound(t_eval)

        op1 = -0.5 * hilbert_space.laplacian()
        op2 = hilbert_space.potential_energy(objective)

        super().__init__(t0, t1, op1, op2, u1_max, u2_max, hilbert_space, 
            dt_max=dt_max, hbar=1.0, timestepper=timestepper, split_method=split_method)

    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState, 
        params: Optional[PyTree]=None) -> AbstractState:

        c1_coeffs = self.coeffs(t, dt, self.u1)
        c2_coeffs = self.coeffs(t, dt, self.u2)
        y_next = y

        for i in range(self.weights.shape[0]):
            y_next = self.split_method.exp(
                c1_coeffs[i] * self.op1 + c2_coeffs[i] * self.op2, (-1j / self.hbar) * dt, y_next)

        return y_next

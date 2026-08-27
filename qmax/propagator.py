from typing import ClassVar, Optional, Callable
from abc import abstractmethod
from math import ceil

import equinox as eqx
import jax 
import jax.numpy as jnp

import tqdm

from jaxtyping import Scalar, ScalarLike, PyTree, Array, ArrayLike

from ._introspect import CountDict
from .hilbert_space import AbstractState, AbstractHilbertSpace
from .operator import Operator, IncompatibleDomainError
from .control import AbstractControl, ControlFunction
from .timestepper import AbstractTimeStepper, Midpoint
from .exponentiators import AbstractSplitMethod, Strang
from .spaces.spatial_discretization import SpatialDiscretization


# TODO:
#   1. ControlledPropagator should evaluate a Hamiltonian of the form
#           H(t) = H_0 + \sum_{i=1}^{m}u_i(t)H_i
#   2. Progress bars
#   3. propagate() should optionally compute cost functions:
#       a. Running cost c(t, y, args) 
#       b. Terminal cost V(t1, y1, args)
#   4. Checkpoints + explicit adjoints
#



class PropagateResult(eqx.Module):
    y0: AbstractState
    y1: AbstractState
    ys: PyTree
    ts: ArrayLike


def _save_y(t, y): 
    return y


class AbstractPropagator(eqx.Module):
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    timestepper: AbstractTimeStepper = eqx.field(default=Midpoint(), kw_only=True)
    hilbert_space: eqx.AbstractVar[AbstractHilbertSpace]

    @property
    def weights(self) -> Array:
        return self.timestepper.weights

    @property
    def quad_rule(self) -> Array:
        return self.timestepper.quad_rule

    @property
    def dt(self) -> Scalar:
        return (self.t1 - self.t0) / self.num_steps

    @property
    def hbar(self) -> Scalar:
        return self.hilbert_space.hbar

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
        y0: AbstractState, 
        params: Optional[PyTree]=None,
        *,
        save_every: Optional[int] = None, 
        save_fn: Callable[[ScalarLike, AbstractState], PyTree] = _save_y, 
        progressbar: bool=False) -> PropagateResult:

        if save_every is None:
            save_every = self.num_steps

        if not self.num_steps % save_every == 0:
            raise ValueError(f"num_steps={self.num_steps} is not divisible by save_every={save_every}")

        if progressbar:
            BAR = "Propagating: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}, {elapsed}<{remaining}]"
            tqdm_bar = tqdm.tqdm(total=self.num_steps, mininterval=0.2, bar_format=BAR, unit=" steps")

            def update_bar(steps):
                tqdm_bar.update(int(steps))

        def inner_loop(y, t):
            y_next = self.propagate_stage(t, self.dt, y, params)
            if progressbar:
                jax.experimental.io_callback(update_bar, None, 1)

            return y_next, None

        def loop(y, args):
            t, t_next = args
            y_next, _ = jax.lax.scan(inner_loop, y, jnp.linspace(t, t_next, save_every, endpoint=False))
            return y_next, save_fn(t_next, y_next)

        t_range = jnp.linspace(self.t0, self.t1, self.num_steps // save_every + 1, endpoint=True)
        y1, ys = jax.lax.scan(loop, y0, (t_range[:-1], t_range[1:]))
        ys = self.hilbert_space.concatenate((y0, ys))
        
        if progressbar:
            tqdm_bar.close()

        return PropagateResult(y0, y1, ys, t_range)

    @abstractmethod
    def count_stage(
        self, 
        t: ScalarLike, 
        dt: ScalarLike, 
        params: Optional[PyTree]=None) -> CountDict:
        
        pass

    def count(self, params: Optional[PyTree]=None) -> CountDict:
        t_range = jnp.linspace(self.t0, self.t1, self.num_steps + 1, endpoint=True)
        dt = t_range[1] - t_range[0]
        c = self.count_stage(0, dt, params)
        return len(t_range) * c


class TimeInvariantPropagator(AbstractPropagator):
    op: Operator

    def __init__(
        self, 
        t0: ScalarLike,
        t1: ScalarLike,
        op: Operator,
        *,
        dt_max: Optional[ScalarLike]=None,
        timestepper: AbstractTimeStepper = Midpoint()):

        self.t0 = t0
        self.t1 = t1

        if dt_max is None:
            self.num_steps = 1
        else:
            self.num_steps = ceil((t1 - t0) / dt_max)

        self.timestepper = timestepper

        op.check_exponentiable_tree()

        h = self.dt / op.domain.hbar * jnp.max(jnp.abs(jnp.sum(self.weights, axis=1)))
        self.op = op.adapt(h)

    @property
    def hilbert_space(self) -> AbstractHilbertSpace:
        return self.op.domain

    def count_stage(
        self, 
        t: ScalarLike, 
        dt: ScalarLike, 
        params: Optional[PyTree]=None) -> CountDict:

        c = CountDict()
        for i in range(self.weights.shape[0]):
            w = jnp.sum(self.weights[i, :])
            c |= self.op.exp_count((-1j / self.hbar) * w * dt)
        return c

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
        *,
        dt_max: Optional[ScalarLike]=None,
        timestepper: AbstractTimeStepper = Midpoint(),
        split_method: AbstractSplitMethod = Strang()):

        self.t0 = t0
        self.t1 = t1

        if dt_max is None:
            self.num_steps = 1
        else:
            self.num_steps = ceil((t1 - t0) / dt_max)

        self.timestepper = timestepper
        self.u1_max = u1_max
        self.u2_max = u2_max

        if op1.domain != op2.domain:
            raise IncompatibleDomainError(
                f"{type(op1).__name__} acts on {op1.domain}, "
                f"but {type(op2).__name__} acts on {op2.domain}"
            )

        # propagate_stage splits c1 * op1 + c2 * op2, so validate that whole tree up front
        add_op = op1 + op2
        split_method.check_exponentiable_tree(add_op)

        w_max = jnp.max(jnp.sum(jnp.abs(self.weights), axis=1))
        h1, h2 = split_method.h_scales(add_op)
        dt1 = h1 * w_max * self.u1_max * self.dt / op1.domain.hbar
        dt2 = h2 * w_max * self.u2_max * self.dt / op1.domain.hbar

        self.op1 = op1.adapt(dt1)
        self.op2 = op2.adapt(dt2)
        self.split_method = split_method

    @property
    def hilbert_space(self) -> AbstractHilbertSpace:
        return self.op1.domain

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
            y_next = self.split_method(
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

        super().__init__(t0, t1, op1, op2, u1_max, u2_max,
            dt_max=dt_max, timestepper=timestepper, split_method=split_method)

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
            y_next = self.split_method(
                c1_coeffs[i] * self.op1 + c2_coeffs[i] * self.op2, (-1j / self.hbar) * dt, y_next)

        return y_next

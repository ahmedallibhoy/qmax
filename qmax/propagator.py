from typing import Optional, Callable
from math import ceil

import equinox as eqx
import jax
import jax.numpy as jnp

import tqdm

from jaxtyping import Scalar, ScalarLike, PyTree, Array

from ._introspect import CountDict
from .adjoint import AbstractAdjoint, ReversibleAdjoint
from .hilbert_space import AbstractState, AbstractHilbertSpace
from .operator import Operator
from .timevarying_operator import AbstractTimeVaryingOperator, ConstantTimeVaryingOperator
from .timestepper import AbstractTimeStepper, Midpoint


class PropagateResult(eqx.Module):
    y0: AbstractState
    y1: AbstractState
    ys: PyTree
    ts: Array
    cost: Scalar


type CostFunction = Callable[[ScalarLike, AbstractState], Scalar]
type SaveFunction = Callable[[ScalarLike, AbstractState], PyTree]

def _save_y(t, y): 
    return y

def _no_cost(t, y):
    return jnp.zeros(())


class Propagator(eqx.Module):
    t_op: AbstractTimeVaryingOperator
    t0: Scalar = eqx.field(static=True, converter=float)
    t1: Scalar = eqx.field(static=True, converter=float)
    num_steps: int = eqx.field(static=True)
    timestepper: AbstractTimeStepper = eqx.field(default=Midpoint(), kw_only=True)

    def __check_init__(self):
        self.t_op(self.t0).check_exponentiable_tree()

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
        return self.domain.hbar

    @property
    def domain(self) -> AbstractHilbertSpace:
        return self.t_op.domain

    @property
    def ts(self) -> Array:
        return jnp.linspace(self.t0, self.t1, self.num_steps + 1)

    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState) -> AbstractState:

        y_next = y
        t_quad, _ = self.quad_rule

        for i in range(self.weights.shape[0]):
            H = self.t_op.quadrature(t + dt * t_quad, self.weights[i])
            y_next = H.exp((-1j / self.hbar) * dt, y_next)

        return y_next

    def propagate(
        self,
        y0: AbstractState,
        *,
        cost_fn: CostFunction=_no_cost,
        save_fn: SaveFunction=_save_y, 
        save_every: Optional[int]=None,
        progressbar: bool=False, 
        adjoint: AbstractAdjoint=ReversibleAdjoint()) -> PropagateResult:

        if save_every is None:
            save_every = self.num_steps

        if not self.num_steps % save_every == 0:
            raise ValueError(f"num_steps={self.num_steps} is not divisible by save_every={save_every}")

        if progressbar:
            BAR = "Propagating: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}, {elapsed}<{remaining}]"
            tqdm_bar = tqdm.tqdm(total=self.num_steps, mininterval=0.2, bar_format=BAR, unit=" steps")
            def update_bar(n):
                tqdm_bar.update(int(n))

            callback = lambda: jax.experimental.io_callback(update_bar, None, 1)
        else:
            callback = None

        # capture inputs or other variables that cost_fn closes over. 
        cost_fn = eqx.filter_closure_convert(cost_fn, jnp.asarray(self.t0), y0)

        y1, total_cost, ys = adjoint.propagate_fn((self, cost_fn, y0), save_every, save_fn, callback)

        if progressbar:
            tqdm_bar.close()

        return PropagateResult(y0, y1, ys, self.ts[::save_every], total_cost)

    def count_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike) -> CountDict:

        c = CountDict()
        t_quad, _ = self.quad_rule

        for i in range(self.weights.shape[0]):
            H = self.t_op.quadrature(t + dt * t_quad, self.weights[i])            
            c |= H.exp_count((-1j / self.hbar) * dt)

        return c

    def count(self) -> CountDict:
        c = CountDict()
        t_range = jnp.linspace(self.t0, self.t1, self.num_steps + 1, endpoint=True)
        
        if isinstance(self.t_op, ConstantTimeVaryingOperator):
            return self.num_steps * self.count_stage(self.t0, self.dt)

        for t in t_range[:-1]:
            c |= self.count_stage(t, self.dt)
        return c


def propagator(
    op: Operator | AbstractTimeVaryingOperator, 
    t0: ScalarLike, 
    t1: ScalarLike,
    *, 
    num_steps: Optional[int]=None,
    dt_max: Optional[ScalarLike]=None,
    timestepper: AbstractTimeStepper = Midpoint(), 
    adapt: bool=True) -> Propagator:

    if dt_max is not None and num_steps is not None:
        raise ValueError(f"Only one of dt_max or num_steps may not be None")

    if num_steps is None and dt_max is None:
        num_steps = 1
    elif num_steps is None:
        num_steps = ceil((t1 - t0) / dt_max)

    if isinstance(op, Operator):
        if adapt:
            op = op.adapt((t1 - t0) / num_steps) 
        op = ConstantTimeVaryingOperator(op)

    return Propagator(op, t0, t1, num_steps=num_steps, timestepper=timestepper)

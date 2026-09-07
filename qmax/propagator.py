from typing import ClassVar, Optional, Callable
from abc import abstractmethod
from math import ceil

import equinox as eqx
import equinox.internal as eqxi
import jax 
import jax.numpy as jnp

import tqdm

from jaxtyping import Scalar, ScalarLike, PyTree, Array, ArrayLike

from ._introspect import CountDict
from .hilbert_space import AbstractState, AbstractHilbertSpace
from .operator import Operator, AddOperator, IncompatibleDomainError
from .timevarying_operator import AbstractTimeVaryingOperator, ConstantTimeVaryingOperator
from .control import AbstractControl, ControlFunction
from .timestepper import AbstractTimeStepper, Midpoint
from .exponentiators import AbstractSplitMethod, Strang
from .spaces.spatial_discretization import SpatialDiscretization


class PropagateResult(eqx.Module):
    y0: AbstractState
    y1: AbstractState
    ys: PyTree
    ts: Array
    cost: Scalar


type CostFunction = Callable[[ScalarLike, AbstractState, tuple[AbstractControl, ...]], Scalar]


def _save_y(t, y): 
    return y

def _no_cost(t, y):
    return 0.0


class Propagator(eqx.Module):
    t_op: AbstractTimeVaryingOperator
    t0: Scalar = eqx.field(static=True)
    t1: Scalar = eqx.field(static=True)
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
        cost_fn: CostFunction = _no_cost,
        save_fn: Callable[[ScalarLike, AbstractState], PyTree] = _save_y, 
        save_every: Optional[int] = None,
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

        def step(carry, t):
            (y, cost, total_cost) = carry
            y_next = self.propagate_stage(t, self.dt, y)
            cost_next = cost_fn(t + self.dt, y_next)   
            total_cost = total_cost + self.dt * (cost + cost_next) / 2

            if progressbar:
                jax.experimental.io_callback(update_bar, None, 1)

            return (y_next, cost_next, total_cost), None

        def loop(carry, args):
            t, t_next = args
            (y_next, cost_next, total_cost), _ = eqxi.scan(
                step, carry, jnp.linspace(t, t_next, save_every, endpoint=False), kind="checkpointed")
            return (y_next, cost_next, total_cost), save_fn(t_next, y_next)

        t_range = jnp.linspace(self.t0, self.t1, self.num_steps // save_every + 1, endpoint=True)
        (y1, _, total_cost), ys = eqxi.scan(
            loop, (y0, cost_fn(self.t0, y0), 0.0), (t_range[:-1], t_range[1:]), kind="checkpointed")
        
        ys = jax.tree.map(
            lambda a, b: jnp.concatenate([jnp.asarray(a)[None], b], axis=0),
            save_fn(self.t0, y0), ys)
        
        if progressbar:
            tqdm_bar.close()

        return PropagateResult(y0, y1, ys, t_range, total_cost)


def propagator(
    op: Operator | AbstractTimeVaryingOperator, 
    t0: ScalarLike, 
    t1: ScalarLike,
    *, 
    num_steps: Optional[int]=None,
    dt_max: Optional[ScalarLike]=None,
    timestepper: AbstractTimeStepper = Midpoint(), 
    adapt: bool=True) -> AbstractPropagator:

    if dt_max is not None and num_steps is not None:
        raise ValueError(f"Only one of dt_max or num_steps may not be None")

    if num_steps is None and dt_max is None:
        num_steps = 1
    elif dt_max is None:
        num_steps = num_steps
    else:
        num_steps = ceil((t1 - t0) / dt_max)

    if isinstance(op, Operator):
        if adapt:
            op = op.adapt((t1 - t0) / num_steps) 
        op = ConstantTimeVaryingOperator(op)

    return Propagator(op, t0, t1, num_steps=num_steps, timestepper=timestepper)

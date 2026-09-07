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
from .timevarying_operator import AbstractTimeVaryingOperator
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


class AbstractPropagator(eqx.Module):
    t0: Scalar
    t1: Scalar
    num_steps: int
    timestepper: AbstractTimeStepper = eqx.field(default=Midpoint(), kw_only=True)
    domain: eqx.AbstractVar[AbstractHilbertSpace]

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

    @abstractmethod
    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState) -> AbstractState:

        pass

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


class TimeInvariantPropagator(AbstractPropagator):
    op: Operator

    def __init__(
        self, 
        op: Operator,
        t0: ScalarLike,
        t1: ScalarLike,
        *,
        num_steps: Optional[int]=None,
        dt_max: Optional[ScalarLike]=None,
        timestepper: AbstractTimeStepper = Midpoint(), 
        adapt: bool=True):

        self.t0 = t0
        self.t1 = t1

        if dt_max is not None and num_steps is not None:
            raise ValueError(f"Only one of dt_max or num_steps may not be None")

        if num_steps is None and dt_max is None:
            self.num_steps = 1
        elif dt_max is None:
            self.num_steps = num_steps
        else:
            self.num_steps = ceil((t1 - t0) / dt_max)

        self.timestepper = timestepper

        op.check_exponentiable_tree()

        h = self.dt / op.domain.hbar * jnp.max(jnp.abs(jnp.sum(self.weights, axis=1)))

        if adapt:
            self.op = op.adapt(h)
        else:
            self.op = op

    @property
    def domain(self) -> AbstractHilbertSpace:
        return self.op.domain

    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState) -> AbstractState:

        y_next = y

        for i in range(self.weights.shape[0]):
            w = jnp.sum(self.weights[i, :])
            y_next = self.op.exp((-1j / self.hbar) * w * dt, y_next)

        return y_next

    def count_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike) -> CountDict:

        c = CountDict()
        for i in range(self.weights.shape[0]):
            w = jnp.sum(self.weights[i, :])
            c |= self.op.exp_count((-1j / self.hbar) * w * dt)
        return c

    def count(self) -> CountDict:
        c = self.count_stage(0.0, self.dt)
        return self.num_steps * c


class TimeVaryingPropagator(AbstractPropagator):
    t_op: AbstractTimeVaryingOperator

    def __init__(
        self, 
        t_op: AbstractTimeVaryingOperator,
        t0: ScalarLike, 
        t1: ScalarLike, 
        *,
        num_steps: Optional[int]=None,
        dt_max: Optional[ScalarLike]=None,
        timestepper: AbstractTimeStepper = Midpoint()):

        self.t0 = t0
        self.t1 = t1

        if dt_max is not None and num_steps is not None:
            raise ValueError(f"Only one of dt_max or num_steps may not be None")

        if num_steps is None and dt_max is None:
            self.num_steps = 1
        elif dt_max is None:
            self.num_steps = num_steps
        else:
            self.num_steps = ceil((t1 - t0) / dt_max)

        self.timestepper = timestepper

        t_op(t0).check_exponentiable_tree()
        self.t_op = t_op

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
    adapt: bool=True) -> AbstractPropagator:

    if isinstance(op, Operator):
        return TimeInvariantPropagator(op, t0, t1, 
            num_steps=num_steps, dt_max=dt_max, timestepper=timestepper, adapt=adapt)
    else:
        return TimeVaryingPropagator(op, t0, t1,
            num_steps=num_steps, dt_max=dt_max, timestepper=timestepper)

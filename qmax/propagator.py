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
from .operator import Operator, AddOperator, IncompatibleDomainError
from .timevarying_operator import AbstractTimeVaryingOperator
from .control import AbstractControl, ControlFunction
from .timestepper import AbstractTimeStepper, Midpoint
from .exponentiators import AbstractSplitMethod, Strang
from .spaces.spatial_discretization import SpatialDiscretization


# TODO:
#   1. propagate() should optionally compute cost functions:
#       a. Running cost c(t, y, args) 
#       b. Terminal cost V(t1, y1, args)
#   2. Checkpoints + explicit adjoints



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
            y_next = self.propagate_stage(t, self.dt, y)
            if progressbar:
                jax.experimental.io_callback(update_bar, None, 1)

            return y_next, None

        def loop(y, args):
            t, t_next = args
            y_next, _ = jax.lax.scan(inner_loop, y, jnp.linspace(t, t_next, save_every, endpoint=False))
            return y_next, save_fn(t_next, y_next)

        t_range = jnp.linspace(self.t0, self.t1, self.num_steps // save_every + 1, endpoint=True)
        y1, ys = jax.lax.scan(loop, y0, (t_range[:-1], t_range[1:]))
        ys = self.domain.concatenate((y0, ys))
        
        if progressbar:
            tqdm_bar.close()

        return PropagateResult(y0, y1, ys, t_range)

    @abstractmethod
    def count_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike) -> CountDict:

        pass

    def count(self) -> CountDict:
        c = CountDict()
        t_range = jnp.linspace(self.t0, self.t1, self.num_steps + 1, endpoint=True)
        for t in t_range[:-1]:
            c |= self.count_stage(t, self.dt)
        return c


class TimeInvariantPropagator(AbstractPropagator):
    op: Operator

    def __init__(
        self, 
        t0: ScalarLike,
        t1: ScalarLike,
        op: Operator,
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

    def count_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike) -> CountDict:

        c = CountDict()
        for i in range(self.weights.shape[0]):
            w = jnp.sum(self.weights[i, :])
            c |= self.op.exp_count((-1j / self.hbar) * w * dt)
        return c

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


class TimeVaryingPropagator(AbstractPropagator):
    t_op: AbstractTimeVaryingOperator

    def __init__(
        self, 
        t0: ScalarLike, 
        t1: ScalarLike, 
        t_op: AbstractTimeVaryingOperator,
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


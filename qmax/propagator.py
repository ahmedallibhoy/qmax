from typing import Optional, Callable, Iterable
from math import ceil

import equinox as eqx
import jax
import jax.numpy as jnp

import tqdm

from jaxtyping import Scalar, ScalarLike, ArrayLike, PyTree, Array

from ._introspect import CountDict
from .adjoint import AbstractAdjoint, ReversibleAdjoint
from .control import AbstractControl
from .hilbert_space import AbstractState, AbstractHilbertSpace
from .operator import Operator
from .timevarying_operator import AbstractTimeVaryingOperator, ConstantTimeVaryingOperator
from .controlled_operator import ControlledOperator
from .timestepper import AbstractTimeStepper, Midpoint


class PropagateResult(eqx.Module):
    y0: AbstractState
    y1: AbstractState
    ys: PyTree
    ts: Array
    running_cost: Scalar
    terminal_cost: Scalar
    total_cost: Scalar


type CostFunction = Callable[[ScalarLike, AbstractState, ArrayLike], Scalar]
type SaveFunction = Callable[[ScalarLike, AbstractState, ArrayLike], PyTree]
type TerminalCostFunction = Callable[[ScalarLike, AbstractState], Scalar]

def _save_y(t, y, u): 
    return y

def _no_cost(t, y, u):
    return jnp.zeros(())

def _no_term_cost(t, y):
    return jnp.zeros(())


class Propagator(eqx.Module):
    op: ControlledOperator
    t0: Scalar = eqx.field(static=True, converter=float)
    t1: Scalar = eqx.field(static=True, converter=float)
    num_steps: int = eqx.field(static=True)
    timestepper: AbstractTimeStepper = eqx.field(default=Midpoint(), kw_only=True)

    def __init__(self, 
        op: Operator | AbstractTimeVaryingOperator | ControlledOperator, 
        t0: ScalarLike, 
        t1: ScalarLike, 
        *, 
        num_steps: Optional[int]=None,
        dt_max: Optional[ScalarLike]=None,
        timestepper: AbstractTimeStepper=Midpoint(), 
        adapt: bool=True):

        self.t0 = t0
        self.t1 = t1
        self.timestepper = timestepper

        if dt_max is not None and num_steps is not None:
            raise ValueError(f"Only one of dt_max or num_steps may not be None")

        if num_steps is None and dt_max is None:
            self.num_steps = 1
        elif num_steps is None:
            self.num_steps = ceil((t1 - t0) / dt_max)
        else:
            self.num_steps = num_steps

        if isinstance(op, Operator):
            if adapt: 
                op = op.adapt(self.dt)
            op = ControlledOperator(op)

        if isinstance(op, AbstractTimeVaryingOperator):
            op = ControlledOperator(op)

        self.op = op

    def __check_init__(self):
        self.op(self.t0, jnp.zeros((self.op.num_controls,))).check_exponentiable_tree()

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
        return self.op.domain

    @property
    def ts(self) -> Array:
        return jnp.linspace(self.t0, self.t1, self.num_steps + 1)

    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState,
        u_quad: ArrayLike) -> AbstractState:

        y_next = y
        t_quad, _ = self.quad_rule

        for i in range(self.weights.shape[0]):
            H = self.op.quadrature(t + dt * t_quad, u_quad, self.weights[i])
            y_next = H.exp((-1j / self.hbar) * dt, y_next)

        return y_next

    def propagate(
        self,
        y0: AbstractState,
        controls: Iterable[AbstractControl]=(),
        *,
        running_cost_fn: CostFunction=_no_cost,
        terminal_cost_fn: TerminalCostFunction=_no_term_cost,
        save_fn: SaveFunction=_save_y, 
        save_every: Optional[int]=None,
        progressbar: bool=False, 
        adjoint: AbstractAdjoint=ReversibleAdjoint()) -> PropagateResult:

        if save_every is None:
            save_every = self.num_steps

        if not self.num_steps % save_every == 0:
            raise ValueError(f"num_steps={self.num_steps} is not divisible by save_every={save_every}")

        controls = tuple(controls)

        if len(controls) != self.op.num_controls:
            raise ValueError(
                f"Expected {self.op.num_controls} controls but received {len(controls)}")

        if controls:
            t_quads = self.timestepper.eval_points(self.ts)  
            us = jnp.stack([jax.vmap(u)(self.ts) for u in controls], axis=1)
            u_quads = jnp.stack([jax.vmap(jax.vmap(u))(t_quads) for u in controls], axis=1)
        else:
            us = jnp.zeros((self.num_steps + 1, 0))
            u_quads = jnp.zeros((self.num_steps, 0, self.timestepper.num_nodes))

        if progressbar:
            BAR = "Propagating: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}, {elapsed}<{remaining}]"
            tqdm_bar = tqdm.tqdm(total=self.num_steps, mininterval=0.2, bar_format=BAR, unit=" steps")
            def update_bar(n):
                tqdm_bar.update(int(n))

            callback = lambda: jax.experimental.io_callback(update_bar, None, 1)
        else:
            callback = None

        y1, running_cost, ys = adjoint.propagate_fn(
            (y0, us, u_quads), self, running_cost_fn, save_every, save_fn, callback)
        terminal_cost = terminal_cost_fn(self.t1, y1)
        total_cost = running_cost + terminal_cost

        if progressbar:
            tqdm_bar.close()

        return PropagateResult(y0, y1, ys, self.ts[::save_every], running_cost, terminal_cost, total_cost)

    def count_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike) -> CountDict:

        c = CountDict()
        t_quad, _ = self.quad_rule
        u_quad = jnp.zeros((self.op.num_controls, self.timestepper.num_nodes))

        for i in range(self.weights.shape[0]):
            H = self.op.quadrature(t + dt * t_quad, u_quad, self.weights[i])            
            c |= H.exp_count((-1j / self.hbar) * dt)

        return c

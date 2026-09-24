from math import ceil
from typing import Callable, Iterable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import tqdm
from jax.experimental import io_callback
from jaxtyping import Array, PyTree, Scalar, ScalarLike

from ._introspect import CountDict
from ._types import ComplexArrayLike, RealScalarLike
from .adjoint import AbstractAdjoint, ReversibleAdjoint
from .control import AbstractControl
from .controlled_operator import ControlledOperator
from .hilbert_space import AbstractHilbertSpace, AbstractState
from .operator import Operator
from .timestepper import AbstractTimeStepper, Midpoint
from .timevarying_operator import AbstractTimeVaryingOperator

# TODO: enforce batch safety of Propagator and adjoints

class PropagateResult(eqx.Module):
    y0: AbstractState
    y1: AbstractState
    ys: PyTree
    ts: Array
    running_cost: Scalar
    terminal_cost: Scalar
    total_cost: Scalar


type CostFunction = Callable
type SaveFunction = Callable
type TerminalCostFunction = Callable

def _save_y(t, y, u): 
    return y

def _no_cost(t, y, u):
    return jnp.zeros(())

def _no_term_cost(t, y):
    return jnp.zeros(())


class Propagator(eqx.Module):
    r"""
    Object representing the propagator $U\big(t_0, t_1; u(\cdot)\big)$ of the Schrödinger equation. 
    """

    op: ControlledOperator
    t0: float = eqx.field(static=True, converter=float)
    t1: float = eqx.field(static=True, converter=float)
    num_steps: int = eqx.field(static=True)
    timestepper: AbstractTimeStepper = eqx.field(default=Midpoint(), kw_only=True)

    def __init__(self, 
        op: Operator | AbstractTimeVaryingOperator | ControlledOperator, 
        t0: RealScalarLike, 
        t1: RealScalarLike, 
        *, 
        num_steps: Optional[int]=None,
        dt_max: Optional[RealScalarLike]=None,
        timestepper: AbstractTimeStepper=Midpoint(), 
        adapt: bool=True):
        """
        Constructs a Propagator. 

        Args:
            op (Operator | AbstractTimeVaryingOperator | ControlledOperator): The Hamiltonian
                of the system.
            t0 (ScalarLike): The initial time.
            t1 (ScalarLike): The terminal time.
            num_steps (Optional[int]): The number of steps the integration method should take. 
                Cannot be used with `dt_max`. If `num_steps=None` and `dt_max=None` then the number 
                of steps is 1.
            dt_max (Optional[int]):  The maximum stepsize of the integrator. 
                Cannot be used with `num_steps`.
            timestepper (AbstractTimeStepper): The timestepping method, 
                see [Timesteppers](timesteppers.md).
            adapt (bool): Whether the operator should be adapted. 
                This parameter is ignored if the Hamiltonian is a `AbstractTimeVaryingOperator` 
                or `ControlledOperator`. 
        """

        self.t0 = float(t0)
        self.t1 = float(t1)
        self.timestepper = timestepper

        if dt_max is not None and num_steps is not None:
            raise ValueError("Only one of dt_max or num_steps may not be None")

        if num_steps is not None:
            self.num_steps = num_steps
        elif dt_max is not None:
            self.num_steps = ceil((t1 - t0) / dt_max)
        else:
            self.num_steps = 1

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
    def weights(self) -> ComplexArrayLike:
        return self.timestepper.weights

    @property
    def quad_rule(self) -> tuple[ComplexArrayLike, ComplexArrayLike]:
        return self.timestepper.quad_rule

    @property
    def dt(self) -> float:
        return (self.t1 - self.t0) / self.num_steps

    @property
    def hbar(self) -> float:
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
        u_quad: Array) -> AbstractState:

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
        r"""
        Computes $U(t_0, t_1; u)\psi_0$. This method optionally can 
        save intermediate values over the integration interval, and computes 
        a cost function of the form
        $J(\psi_0, u) = V(t_1, \psi(t_1)) + \int_{t_0}^{t_1}\ell(t, \psi(t), u(t))dt$.

        Args:
            y0 (AbstractState): initial condition
            controls (tuple[AbstractControl, ...]): Control inputs to apply to the system
                if `Propagator` was constructed using a `ControlledOperator` (
                see [Controlled Operator](operators/controlled.md)). The number of provided controls
                must equal the number inputs to the controlled Hamiltonian. 
            running_cost_fn (callable): Function with signature `running_cost_fn(t, y, u)` 
                returning a scalar. `Propagator` records the integral of this function over the 
                integration interval.  
            terminal_cost_fn (callable): Function with signature `terminal_cost_fn(t, y)` returning 
                a scalar. `Propagator` records the value `terminal_cost_fn(t1, y1)`.
            save_fn (callable):  Function with signature `save_fn(t, y, u)` returning a PyTree. 
                `Propagator` records `save_function(t, y, u)` every `save_every` steps over the 
                integration interval. 
            save_every (Optional[int]): Number of steps per call to `save_fn`, e.g. if 
                `save_every=2` then every other step is saved. If `save_every=None` then `save_fn` 
                is called only on the first and last steps of the integration. 
            progressbar (bool): Whether to display a tqdm progress bar. 
            adjoint (AbstractAdjoint): How to differentatate `propagate`, 
                see [Adjoints](adjoints.md).

        Returns:
            A `PropagateResult` object containing the following fields:

                - **`y0`** – The initial state 

                - **`y1`** – The terminal state

                - **`ys`** – PyTree of saved values across the integration interval

                - **`ts`** – The times of the saved values

                - **`running_cost`** – The total 
                    running cost $\int_{t_0}^{t_1}\ell(t, u(t), \psi(t))dt$

                - **`terminal_cost`** – The terminal cost $V(t_1, \psi(t_1))$

                - **`total_cost`** – The sum of `running_cost` and `terminal_cost`
        """

        if save_every is None:
            save_every = self.num_steps

        if not self.num_steps % save_every == 0:
            raise ValueError(
                f"num_steps={self.num_steps} is not divisible by save_every={save_every}")

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
            BAR = ("Propagating: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{rate_fmt}, {elapsed}<{remaining}]")
            tqdm_bar = tqdm.tqdm(
                total=self.num_steps, mininterval=0.2, bar_format=BAR, unit=" steps")
            def update_bar(n):
                tqdm_bar.update(int(n))

            callback = lambda: io_callback(update_bar, None, 1)
        else:
            callback = None

        y1, running_cost, ys = adjoint.propagate_fn(
            (y0, us, u_quads), self, running_cost_fn, save_every, save_fn, callback)
        terminal_cost = jnp.asarray(terminal_cost_fn(self.t1, y1))
        total_cost = jnp.asarray(running_cost) + terminal_cost

        if progressbar:
            tqdm_bar.close() # pyright: ignore[reportPossiblyUnboundVariable]

        return PropagateResult(
            y0, y1, ys, self.ts[::save_every], running_cost, terminal_cost, total_cost)

    def count_stage(
        self,
        t: RealScalarLike,
        dt: RealScalarLike) -> CountDict:
        """
        Produces a `CountDict` object tabulating the number of matvecs, adjoint matvecs, 
            exponential actions, and solves required by each operator in the expression tree of the 
            Hamiltonian to compute one stage of the integration method.

        Args:
            t (ScalarLike): The time of the stage
            dt (ScalarLike): The stepsize 

        Returns:
            A `CountDict` object representing the interface counts of the leaves of the Hamiltonian 
                operator for one stage of the time integration method at time `t` with 
                stepsize `dt`.

        Example:
        ```python
        import qmax as qx 

        hilbert_space = qx.spaces.FiniteDifference(x0=-10, x1=10, num_steps=500)
        L = hilbert_space.laplacian()
        V = hilbert_space.potential_energy(lambda x: 0.5 * x ** 2)
        H = -0.5 * L + V

        U = qx.Propagator(H, t0=0.0, t1=1.0, dt_max=0.01)
        print(U.count_stage(U.t0, U.dt).tree())
        ``` 

        ```
        1.0 * (-0.5 * Laplacian + FiniteDifferencePotentialEnergy)
        └─(-0.5 * Laplacian + FiniteDifferencePotentialEnergy)
          ├─FiniteDifferencePotentialEnergy ·························  exp_actions=2
          └─-0.5 * Laplacian
            └─Laplacian
              └─Laplacian1D(axis=0) ·································  actions=1, solves=1
        ─────────────────────────────────────────────────────────────
        total:                                                         actions=1, solves=1, exp_actions=2 
        ```
        """# noqa: E501

        c = CountDict()
        t_quad, _ = self.quad_rule
        u_quad = jnp.zeros((self.op.num_controls, self.timestepper.num_nodes))

        for i in range(self.weights.shape[0]):
            H = self.op.quadrature(t + dt * t_quad, u_quad, self.weights[i])            
            c |= H.exp_count((-1j / self.hbar) * dt)

        return c

from typing import Callable, Optional

import warnings
from abc import abstractmethod
from functools import reduce
from math import ceil

import equinox as eqx
import jax 
import jax.numpy as jnp

from jaxtyping import ScalarLike, Array, ArrayLike, PyTree

from ._internal import _update_field

from .hilbert_space import AbstractHilbertSpace, AbstractState
from .operator import Operator
from .exponentiators import ScaleSquareExponentiator, min_order
from .split import AbstractSplitMethod, Strang
from .timestepper import AbstractTimeStepper, Midpoint
from .spaces.spatial_discretization import SpatialDiscretization
from .spaces.finite_difference import FiniteDifference, FiniteDifferenceLaplacian, FiniteDifferencePotentialEnergy
from .spaces.pseudospectral import PseudoSpectral, PseudoSpectralLaplacian, PseudoSpectralPotentialEnergy
from .tensor import TensorProduct, AbstractTensorOperator


__all__ = [
    "adapt_operator",
    "TimeInvariantSystem", 
    "TimeVaryingSystem", 
    "ScalarSplitTimeVaryingSystem", 
    "QuantumHamiltonianDescent", 
    "PropagateResult"
]


def _adapt_dict(
    op_dict: dict, 
    hilbert_space: AbstractHilbertSpace, 
    dt_max: ScalarLike) -> dict:

    root = op_dict["obj"]

    if isinstance(root, AbstractTensorOperator):
        if "ops" in op_dict and op_dict["ops"] is not None:
            # KroneckerSum
            op_dict["ops"] = tuple(
                _adapt_dict(op, hilbert_space[idx], dt_max) 
                for idx, op in enumerate(op_dict["ops"])
            )
            root = eqx.tree_at(lambda o: o.ops, root, [d["obj"] for d in op_dict["ops"]])

        if "op" in op_dict and op_dict["op"] is not None:
            # Lift
            op_dict["op"] = _adapt_dict(op_dict["op"], hilbert_space[root.factor_idx], dt_max)
            root = eqx.tree_at(lambda o: o.op, root, op_dict["op"]["obj"])
    else:
        for key in ("op", "op1", "op2"):
            if key in op_dict and op_dict[key] is not None:
                op_dict[key] = _adapt_dict(op_dict[key], hilbert_space, dt_max)
                root = eqx.tree_at(lambda o, k=key: getattr(o, k), root, op_dict[key]["obj"])

    op_dict["obj"] = root
    if op_dict["exp_delegated"]:
        return op_dict

    new_exp = root.exponentiator.adapt(root, hilbert_space, dt_max * op_dict["h_scale"])
    op_dict["obj"] = root.with_exponentiator(new_exp)
    return op_dict


def adapt_operator(
    op: Operator, 
    hilbert_space: AbstractHilbertSpace, 
    dt_max: ScalarLike) -> Operator:

    adapted = _adapt_dict(op.to_dict(), hilbert_space, dt_max)
    return adapted["obj"]


class PropagateResult(eqx.Module):
    y1: AbstractState
    ys: PyTree
    ts: ArrayLike


def _save_y(t, y): 
    return y


type AdaptParams = tuple[ScalarLike, ScalarLike, AbstractHilbertSpace, int]

class AbstractSystem(eqx.Module):
    timestepper: AbstractTimeStepper = eqx.field(default=Midpoint(), kw_only=True)
    hbar: ScalarLike = eqx.field(default=1.0, kw_only=True)
    adapt_params: Optional[AdaptParams] = eqx.field(default=None, static=True, kw_only=True)

    def check_consistency(self):
        pass

    def with_timestepper(self, timestepper: AbstractTimeStepper) -> AbstractSystem:
        new_self =_update_field(self, "adapt_params", None)
        return eqx.tree_at(lambda sys: sys.timestepper, new_self,  timestepper)

    @property
    def weights(self) -> Array:
        return self.timestepper.weights

    @property
    def quad_rule(self) -> Array:
        return self.timestepper.quad_rule

    @abstractmethod
    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState) -> AbstractState:

        pass

    def adapt(
        self, 
        t0: ScalarLike, 
        t1: ScalarLike,
        hilbert_space: AbstractHilbertSpace, 
        num_steps: int, 
        timestepper: Optional[AbstractTimeStepper]=None) -> AbstractSystem:

        if timestepper is not None:
            _self = self.with_timestepper(timestepper)
        else:
            _self = self

        new_self = _self._adapt(t0, t1, hilbert_space, num_steps)
        return _update_field(new_self, "adapt_params", (t0, t1, hilbert_space, num_steps))

    def _adapt(
        self, 
        t0: ScalarLike, 
        t1: ScalarLike, 
        hilbert_space: AbstractHilbertSpace, 
        num_steps: int) -> AbstractSystem:

        return self

    def propagate(
        self, 
        t0: ScalarLike, 
        t1: ScalarLike, 
        y0: AbstractState, 
        num_steps: int,
        save_every: int=1, 
        save_fn: Callable[[ScalarLike, AbstractState], PyTree] = _save_y) -> PropagateResult:

        dt = (t1 - t0) / num_steps
        params = (t0, t1, y0.hilbert_space, num_steps)

        if self.adapt_params is not None and self.adapt_params != params:
            t0_a, t1_a, hs_a, num_steps_a = self.adapt_params

            if y0.hilbert_space != hs_a:
                warnings.warn(
                    f"System adapted to Hilbert space {hs_a}"
                    f"but received state of type {y0.hilbert_space}. "
                    f"Accuracy may be less than reported order estimates",
                    stacklevel=3,
                )

            dt_a = (t1_a - t0_a) / num_steps_a

            if abs(dt) > abs(dt_a):
                warnings.warn(
                    f"Stepsize dt={dt} is greater than"
                    f"the stepsize the system was adapted to {dt_a}. "
                    f"Accuracy may be less than reported order estimates",
                    stacklevel=3,
                )

        if not num_steps % save_every == 0:
            raise ValueError(f"num_steps={num_steps} is not divisible by save_every={save_every}")

        self.check_consistency()
        t_range = jnp.linspace(t0, t1, num_steps // save_every + 1, endpoint=True)

        def inner_loop(y, t):
            y_next = self.propagate_stage(t, dt, y)
            return y_next, None

        def loop(y, args):
            t, t_next = args
            y_next, _ = jax.lax.scan(inner_loop, y, jnp.linspace(t, t_next, save_every, endpoint=False))
            return y_next, save_fn(t_next, y_next)

        y1, ys = jax.lax.scan(loop, y0, (t_range[:-1], t_range[1:]))
        return PropagateResult(y1, ys, t_range[1:])


class TimeInvariantSystem(AbstractSystem):
    """
    Simulates a quantum mechanical system with Hamiltonian H. 
    """
    op: Operator

    def _adapt(
        self, 
        t0: ScalarLike, 
        t1: ScalarLike, 
        hilbert_space: AbstractHilbertSpace, 
        num_steps: int) -> AbstractSystem:

        dt = jnp.abs((t1 - t0) / num_steps)
        dt = dt / self.hbar * jnp.max(jnp.abs(jnp.sum(self.weights, axis=1)))
        op = adapt_operator(self.op, hilbert_space, dt)
        
        return eqx.tree_at(lambda sys: sys.op, self, op)

    def check_consistency(self):
        if self.op.exp_order is not None and self.op.exp_order != self.timestepper.order:
            effective_order = min_order(self.op.exp_order, self.timestepper.order)
            warnings.warn(
                f"Orders do not match: timestepper.order={self.timestepper.order} and "
                f"op.exp_order={self.op.exp_order}, so effective order of solution is "
                f"limited to min({self.timestepper.order}, {self.op.exp_order}) = {effective_order}",
                stacklevel=3,
            )

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


type TimeVaryingOperator = Callable[[ScalarLike], Operator]

class TimeVaryingSystem(AbstractSystem):
    r"""
    Simulates a time-varying quantum mechanical system with Hamiltonian H(t). The 
    propagator is estimated using a commutator-free Magnus expansion:

        U(t + dt, t) = \prod_{i} exp(-(1j / hbar) * dt * H_i)
        H_i = \sum_{j}w_{ij} H(t + h_{ij} * dt)

    where h_{ij} are Gauss-Legendre quadrature nodes, and weights are solutions to order 
    conditions. The exponential of \sum_{j}w_{ij} H(t + h_{ij} * dt) is computed using a 
    splitting method. 
    """

    t_op: TimeVaryingOperator
    split_method: AbstractSplitMethod = eqx.field(default=Strang(), kw_only=True)

    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState) -> AbstractState:

        y_next = y
        t_quad, _ = self.quad_rule

        for i in range(self.weights.shape[0]):
            op_list = [w * self.t_op(t + h * dt) for w, h in zip(self.weights[i, :], t_quad)]
            op = reduce(lambda a, b: (a + b).with_exponentiator(self.split_method), op_list)
            y_next = op.exp((-1j / self.hbar) * dt, y_next)

        return y_next


type ScalarTimeDependence = Callable[[ScalarLike], ScalarLike]

class ScalarSplitTimeVaryingSystem(AbstractSystem):
    r"""
    Simulates a time-varying quantum mechanical system with a Hamiltonian of the form 

        H(t) = c1(t) * T + c2(t) * V

    using a commutator-free Magnus expansion to estimate the propagator:

        U(t + dt, t) = \prod_{i} exp(-(1j / hbar) * dt * H_i)
        H_i = \sum_{j}w_{ij} H(t + h_{ij} * dt)

    Because the time-dependence is entirely contained in the scalar coefficients, the 
    Magnus expansion terms simplify to 

        H_i = a_i * T + b_i * V
        a_i = \sum_{j}w_{ij}c1(t + h_{ij} * dt)
        b_i = \sum_{j}w_{ij}c2(t + h_{ij} * dt)

    and a splitting method is used to exponentiate H_i. In the special case where the 
    integral of c_1 or c_2 is known exactly, the coefficients a_i and b_i replaced to 
    obtain a marginally more accurate estimate of the Magnus terms. 
    """

    op1: Operator
    op2: Operator
    c1: ScalarTimeDependence
    c2: ScalarTimeDependence
    c1_int: Optional[ScalarTimeDependence] = None 
    c2_int: Optional[ScalarTimeDependence] = None
    split_method: AbstractSplitMethod = eqx.field(default=Strang(), kw_only=True)

    def stage_coeffs(self, t: ScalarLike, dt: ScalarLike) -> tuple[Array, Array]:
        t_quad, w_quad = self.quad_rule
        
        def _coeffs(c, c_int):
            cs = jax.vmap(c)(t + dt * t_quad)
            coeffs = self.weights @ cs
            if c_int is None:
                return coeffs 
            c_bar = (c_int(t + dt) - c_int(t)) / dt
            return coeffs + jnp.sum(self.weights, axis=1) * (c_bar - w_quad @ cs)

        c1_coeffs = _coeffs(self.c1, self.c1_int)
        c2_coeffs = _coeffs(self.c2, self.c2_int)
        return c1_coeffs, c2_coeffs

    def _adapt(
        self, 
        t0: ScalarLike, 
        t1: ScalarLike, 
        hilbert_space: AbstractHilbertSpace, 
        num_steps: int) -> AbstractSystem:

        def propagate_dt(t, dt):
            c1_coeffs, c2_coeffs = self.stage_coeffs(t, dt)
            dt1 = jnp.max(jnp.abs(c1_coeffs)) * dt / self.hbar
            dt2 = jnp.max(jnp.abs(c2_coeffs)) * dt / self.hbar
            return dt1, dt2

        t_range = jnp.linspace(t0, t1, num_steps + 1, endpoint=True)
        dt = (t1 - t0) / num_steps
        dt1s, dt2s = jax.vmap(propagate_dt, in_axes=(0, None))(t_range[:-1], dt)

        h1, h2 = self.split_method.h_scales
        op1 = adapt_operator(self.op1, hilbert_space, h1 * jnp.max(jnp.abs(dt1s)))
        op2 = adapt_operator(self.op2, hilbert_space, h2 * jnp.max(jnp.abs(dt2s)))

        return eqx.tree_at(lambda sys: (sys.op1, sys.op2), self, (op1, op2))

    def check_consistency(self):
        # Exact exponentiators report order None and place no limit on the
        # solution, so they drop out rather than participating in the minimum.
        orders = [
            order for order in
            (self.op1.exp_order, self.op2.exp_order, self.split_method.order, self.timestepper.order)
            if order is not None
        ]

        if len(set(orders)) > 1:
            effective_order = min_order(*orders)
            warnings.warn(
                f"Orders do not match: op1.exp_order={self.op1.exp_order}, op2.exp_order={self.op2.exp_order}, "
                f"split_method.order={self.split_method.order}, and timestepper.order={self.timestepper.order}, "
                f"so order is limited to "
                f"min({self.op1.exp_order}, {self.op2.exp_order}, {self.split_method.order}, {self.timestepper.order}) "
                f"= {effective_order}.",
                stacklevel=3,
            )

    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState) -> AbstractState:

        c1_coeffs, c2_coeffs = self.stage_coeffs(t, dt)
        y_next = y

        for i in range(self.weights.shape[0]):
            y_next = self.split_method.exp(
                c1_coeffs[i] * self.op1 + c2_coeffs[i] * self.op2, (-1j / self.hbar) * dt, y_next)

        return y_next


class QuantumHamiltonianDescent(ScalarSplitTimeVaryingSystem):
    r"""
    Implementation of Quantum Hamiltonian Descent method for optimization, which is 
    the quantum analog of classical mechanical models of momentum-based optimization 
    algorithms. Simulates the Hamiltonian 

        H = (1/t) * T + t * f

    where f is the function we want to minimize, which is a special case of a 
    ScalarSplitTimeVaryingSystem.

        1. Leng, Jiaqi, et al. "Quantum hamiltonian descent." arXiv preprint arXiv:2303.01471 (2023).
        2. Leng, Jiaqi, and Bin Shi. "Quantum Optimization via Gradient-Based Hamiltonian Descent." 
        arXiv preprint arXiv:2505.14670 (2025).
    """
    
    def __init__(
        self, 
        objective: Callable[[ArrayLike], ScalarLike], 
        hilbert_space: SpatialDiscretization,
        split_method=Strang(), 
        timestepper=Midpoint()):

        if type(hilbert_space) == FiniteDifference:
            op1 = -0.5 * FiniteDifferenceLaplacian(hilbert_space.spatial_dim)
            op2 = FiniteDifferencePotentialEnergy(objective)
        elif type(hilbert_space) == PseudoSpectral:
            op1 = -0.5 * PseudoSpectralLaplacian()
            op2 = PseudoSpectralPotentialEnergy(objective)
        else:
            raise ValueError(f"Invalid Hilbert space, recieved {hilbert_space}")

        self.op1 = op1
        self.op2 = op2

        self.c1 = lambda t: 1.0 / t
        self.c2 = lambda t: t
        self.c1_int = lambda t: jnp.log(t)
        self.c2_int = lambda t: t ** 2 / 2

        self.split_method = split_method
        self.timestepper = timestepper

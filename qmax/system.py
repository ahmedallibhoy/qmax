from typing import Callable, Optional

import warnings
from abc import abstractmethod
from functools import reduce

import equinox as eqx
import jax 
import jax.numpy as jnp

from jaxtyping import ScalarLike, Array, ArrayLike

from .hilbert_space import AbstractHilbertSpace, AbstractState
from .operator import Operator
from .exponentiators import ScaleSquareExponentiator
from .split import AbstractSplitMethod, Strang
from .timestepper import AbstractTimeStepper
from .spaces.spatial_discretization import SpatialDiscretization
from .spaces.finite_difference import FiniteDifference, FiniteDifferenceLaplacian, FiniteDifferencePotentialEnergy
from .spaces.pseudospectral import PseudoSpectral, PseudoSpectralLaplacian, PseudoSpectralPotentialEnergy

from .tensor import TensorProduct, AbstractTensorOperator

def _adapt_dict(
    op_dict: dict, 
    hilbert_space: AbstractHilbertSpace, 
    dt_max: ScalarLike) -> dict:

    root = op_dict["obj"]

    if isinstance(root, AbstractTensorOperator):
        if "ops" in op_dict and op_dict["ops"] is not None:
            # KroneckerSum
            op_dict["ops"] = [_adapt_dict(op, space, dt_max) for (op, space) in zip(op_dict["ops"], hilbert_space.spaces)]
            root = eqx.tree_at(lambda o: o.ops, root, [d["obj"] for d in op_dict["ops"]])

        if "op" in op_dict and op_dict["op"] is not None:
            # Lift
            op_dict["op"] = _adapt_dict(op_dict["op"], hilbert_space.spaces[root.idx], dt_max)
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


class AbstractSystem(eqx.Module):
    hbar: ScalarLike = eqx.field(default=1.0, kw_only=True)

    def check_consistency(self, time_stepper: AbstractTimeStepper):
        pass

    @abstractmethod
    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState,
        weights: ArrayLike,
        quad_rule: ArrayLike) -> AbstractState:

        pass

    def adapt(
        self, 
        t0: ScalarLike, 
        t1: ScalarLike, 
        dt: ScalarLike, 
        y0: AbstractState, 
        time_stepper) -> AbstractSystem:
        
        t_left = jnp.arange(t0, t1, dt)
        t_range = jnp.concatenate([t_left, t_left[-1:] + dt])
        return self.adapt_t_range(t_range, y0, time_stepper)

    def adapt_dt_range(
        self, 
        t0: ScalarLike, 
        dt_range: ArrayLike, 
        y0: AbstractState, 
        time_stepper) -> AbstractSystem:
        
        t_range = jnp.concatenate([jnp.asarray(t0)[None], t0 + jnp.cumsum(dt_range)])
        return self.adapt_t_range(t_range, y0, time_stepper)

    def adapt_t_range(
        self, 
        t_range: ArrayLike, 
        y0: AbstractState, 
        time_stepper) -> AbstractSystem:
        
        return self

    def propagate(
        self, 
        t0: ScalarLike, 
        t1: ScalarLike, 
        dt: ScalarLike, 
        y0: AbstractState, 
        time_stepper) -> AbstractState:

        t_left = jnp.arange(t0, t1, dt)
        t_range = jnp.concatenate([t_left, t_left[-1:] + dt])
        return self.propagate_t_range(t_range, y0, time_stepper)

    def propagate_dt_range(
        self, 
        t0: ScalarLike, 
        dt_range: ArrayLike, 
        y0: AbstractState, 
        time_stepper) -> AbstractState:

        t_range = jnp.concatenate([jnp.asarray(t0)[None], t0 + jnp.cumsum(dt_range)])
        return self.propagate_t_range(t_range, y0, time_stepper)

    def propagate_t_range(   
        self, 
        t_range: ArrayLike,
        y0: AbstractState,
        time_stepper: AbstractTimeStepper) -> tuple[AbstractState, Array]:

        self.check_consistency(time_stepper)
        dt_range = t_range[1:] - t_range[:-1]

        def step(y, args):
            t, dt = args
            y_next = self.propagate_stage(t, dt, y, time_stepper.weights, time_stepper.quad_rule)
            return y_next, y_next

        _, ys = jax.lax.scan(step, y0, (t_range[:-1], dt_range))      
        ys = jax.tree.map(lambda a, b: jnp.concatenate([a[None, :], b]), y0, ys)  

        return ys, t_range


class TimeInvariantSystem(AbstractSystem):
    """
    Simulates a quantum mechanical system with Hamiltonian H. 
    """
    op: Operator

    def adapt_t_range(
        self, 
        t_range: ArrayLike,
        y0: AbstractState, 
        time_stepper) -> AbstractSystem:

        dt_range = t_range[1:] - t_range[:-1]
        dt_max = jnp.max(jnp.abs(dt_range))
        dt_max = dt_max / self.hbar * jnp.max(jnp.abs(jnp.sum(time_stepper.weights, axis=1)))
        op = adapt_operator(self.op, y0.hilbert_space, dt_max)

        return eqx.tree_at(lambda sys: sys.op, self, op)

    def check_consistency(self, time_stepper: AbstractTimeStepper):
        if not jnp.isinf(self.op.exp_order) and self.op.exp_order != time_stepper.order:
            effective_order = min(self.op.exp_order, time_stepper.order)
            warnings.warn(
                f"Orders do not match: time_stepper.order={time_stepper.order} and "
                f"op.exp_order={self.op.exp_order}, so effective order of solution is "
                f"limited to min({time_stepper.order}, {self.op.exp_order}) = {effective_order}",
                stacklevel=3,
            )

    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState,
        weights: ArrayLike,
        quad_rule: ArrayLike) -> AbstractState:

        y_next = y

        # TODO: replace with lax.scan?
        for i in range(weights.shape[0]):
            w = jnp.sum(weights[i, :])
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
        y: AbstractState,
        weights: ArrayLike,
        quad_rule: ArrayLike) -> AbstractState:

        y_next = y
        t_quad, _ = quad_rule

        # TODO: replace with lax.scan?
        for i in range(weights.shape[0]):
            op_list = [w * self.t_op(t + h * dt) for w, h in zip(weights[i, :], t_quad)]
            op = reduce(lambda a, b: (a + b).with_exponentiator(self.split_method), op_list)
            y_next = op.exp((-1j / self.hbar) * dt, y_next)

        return y_next


def _stage_coeffs(c, c_int, weights, quad_rule, t, dt):
    t_quad, _ = quad_rule
    cs = jax.vmap(c)(t + dt * t_quad)
    coeffs = weights @ cs
    if c_int is None:
        return coeffs 
    c_bar = (c_int(t + dt) - c_int(t)) / dt
    return coeffs + jnp.sum(weights, axis=1) * (c_bar - w_quad @ cs)


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

    def adapt_t_range(
        self, 
        t_range: ArrayLike,
        y0: AbstractState, 
        time_stepper) -> AbstractSystem:

        def propagate_dt(t, dt):
            c1_coeffs =_stage_coeffs(self.c1, self.c1_int, weights, time_stepper.quad_rule, t, dt)
            c2_coeffs =_stage_coeffs(self.c2, self.c2_int, weights, time_stepper.quad_rule, t, dt)
            dt1 = jnp.max(jnp.abs(c1_coeffs)) * dt / self.hbar
            dt2 = jnp.max(jnp.abs(c2_coeffs)) * dt / self.hbar
            return dt1, dt2

        dt_range = t_range[1:] - t_range[:-1]
        dt1s, dt2s = jax.vmap(propagate_dt)(t_range[:-1], dt_range)

        op1 = adapt_operator(self.op1, y0.hilbert_space, jnp.max(jnp.abs(dt1s)))
        op2 = adapt_operator(self.op2, y0.hilbert_space, jnp.max(jnp.abs(dt2s)))

        return eqx.tree_at(lambda sys: (sys.op1, sys.op2), self, (op1, op2))

    def check_consistency(self, time_stepper: AbstractTimeStepper):
        if jnp.isinf(self.op1.exp_order) and jnp.isinf(self.op2.exp_order):
            orders = [self.split_method.order, time_stepper.order]
        elif jnp.isinf(self.op1.exp_order):
            orders = [self.op2.exp_order, self.split_method.order, time_stepper.order]
        elif jnp.isinf(self.op2.exp_order):
            orders = [self.op1.exp_order, self.split_method.order, time_stepper.order]
        else:
            orders = [
                self.op1.exp_order, self.op2.exp_order, self.split_method.order, time_stepper.order
            ]

        if len(set(orders)) > 1:
            effective_order = min(orders)
            warnings.warn(
                f"Orders do not match: op1.exp_order={self.op1.exp_order}, op2.exp_order={self.op2.exp_order}, "
                f"split_method.order={self.split_method.order}, and time_stepper.order={time_stepper.order}, "
                f"so order is limited to "
                f"min({self.op1.exp_order}, {self.op2.exp_order}, {self.split_method.order}, {time_stepper.order}) = {effective_order}.",
                stacklevel=3,
            )

    def propagate_stage(
        self,
        t: ScalarLike,
        dt: ScalarLike,
        y: AbstractState,
        weights: ArrayLike,
        quad_rule: ArrayLike) -> AbstractState:

        y_next = y
        coeffs1 = _stage_coeffs(self.c1, self.c1_int, weights, quad_rule, t, dt)
        coeffs2 = _stage_coeffs(self.c2, self.c2_int, weights, quad_rule, t, dt)

        # TODO: replace with lax.scan?
        for i in range(weights.shape[0]):
            y_next = self.split_method.exp(
                coeffs1[i] * self.op1 + coeffs2[i] * self.op2, (-1j / self.hbar) * dt, y_next)

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
        split_method=Strang()):

        if type(hilbert_space) == FiniteDifference:
            op1 = -0.5 * FiniteDifferenceLaplacian()
            op2 = FiniteDifferencePotentialEnergy(objective)
            if hilbert_space.spatial_dim == 2:
                op1 = op1.with_exponentiator(ScaleSquareExponentiator())
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


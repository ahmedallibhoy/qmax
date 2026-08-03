from __future__ import annotations

from typing import Callable, ClassVar, Union
from abc import abstractmethod
from functools import reduce
import itertools

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike, ScalarLike

from .hilbert_space import AbstractHilbertSpace, AbstractState
from .operator import Operator
from .exponentiators import Order, AbstractExponentiator, ExactExponentiator, KrylovExponentiator


def apply_along_tensor(
    fn: Callable[[ArrayLike], ArrayLike],
    tensor: ArrayLike,
    axis: int) -> Array:
    """Apply `fn` to each 1-D slice along `axis`, batched over every other axis.

    `axis` is a literal array axis and may be negative. `fn` may change the axis
    length; the output shape is inferred from `fn`'s output."""

    t = jnp.moveaxis(tensor, axis, 0)
    rest = t.shape[1:]
    out = jax.vmap(fn, in_axes=1, out_axes=1)(t.reshape(t.shape[0], -1))
    return jnp.moveaxis(out.reshape((out.shape[0],) + rest), 0, axis)


def apply_along_state(
    state_fn: Callable[[AbstractState], AbstractState],
    y: TensorState,
    factor_idx: int) -> TensorState:
    """Apply a subspace endomorphism `state_fn` along factor `factor_idx` of a
    tensor product state, wrapping the from_coeffs / .coeffs / from_tensor
    boilerplate.

    `factor_idx` selects a tensor factor, not an axis of `y.coeff_tensor`. The
    factor axes are the trailing ones, so leading batch axes -- time from a
    timestepper, an outer vmap over initial conditions, both at once -- pass
    through untouched, and `state_fn` only ever sees an unbatched subspace
    state. Callers therefore never need to know the batch rank of `y`."""

    num_factors = y.hilbert_space.num_factors
    space = y.hilbert_space[factor_idx]
    fn = lambda col: state_fn(space.from_coeffs(col)).coeffs
    # Count from the back, so the axis is independent of the batch rank.
    axis = factor_idx % num_factors - num_factors
    return y.hilbert_space.from_tensor(apply_along_tensor(fn, y.coeff_tensor, axis))



class TensorState(AbstractState):
    hilbert_space: AbstractTensorSpace = eqx.field(static=True)

    @property
    def coeff_tensor(self) -> Array:
        return self.coeffs.reshape(*self.coeffs.shape[:-1], *self.hilbert_space.dim_list)


class AbstractTensorSpace(AbstractHilbertSpace):
    state_type: ClassVar = TensorState

    @property
    @abstractmethod
    def structure(self):
        pass

    @abstractmethod
    def factor(self, idx: int) -> AbstractHilbertSpace:
        pass 

    def __getitem__(self, idx: int) -> AbstractHilbertSpace:
        return self.factor(idx)

    @property
    @abstractmethod
    def num_factors(self) -> int:
        pass

    @property
    @abstractmethod
    def dim_list(self) -> list[int]:
        pass

    @property
    def dim(self) -> int:
        return math.prod(self.dim_list)

    def from_tensor(self, coeff_tensor: ArrayLike):
        """Inverse of `TensorState.coeff_tensor`: fold the trailing factor
        axes back into a single coefficient axis, leaving leading batch axes
        untouched."""

        coeff_tensor = jnp.asarray(coeff_tensor)
        batch_shape = coeff_tensor.shape[:coeff_tensor.ndim - self.num_factors]
        return self.from_coeffs(coeff_tensor.reshape(*batch_shape, self.dim))

    def product_state(self, y_list: tuple[AbstractState]) -> TensorState:
        """Form the product state y_0 (x) ... (x) y_{n-1}. Leading batch axes are
        broadcast across factors, so batched factors give a batch of product
        states, and a mix of batched and unbatched factors also works."""

        expanded = [
            y.coeffs.reshape(
                *y.coeffs.shape[:-1],
                *(1,) * factor_idx,
                y.coeffs.shape[-1],
                *(1,) * (self.num_factors - factor_idx - 1))
            for factor_idx, y in enumerate(y_list)
        ]
        return self.from_tensor(reduce(lambda a, b: a * b, expanded))


class TensorProduct(AbstractTensorSpace):
    spaces: tuple[AbstractHilbertSpace]

    @property
    def structure(self):
        return tuple(s.structure for s in self.spaces)

    def factor(self, idx: int) -> AbstractHilbertSpace:
        return self.spaces[idx] 

    @property
    def num_factors(self) -> int:
        return len(self.spaces)

    @property
    def dim_list(self) -> list[int]:
        return [hs.dim for hs in self.spaces]


class TensorPower(AbstractTensorSpace):
    factorspace: AbstractHilbertSpace
    power: int

    @property
    def structure(self):
        return tuple(self.factorspace.structure for _ in range(self.power))

    def factor(self, idx: int) -> AbstractHilbertSpace:
        return self.factorspace

    @property
    def num_factors(self) -> int:
        return self.power

    @property
    def dim_list(self) -> list[int]:
        return [self.factorspace.dim for _ in range(self.power)]

    @property
    def dim(self) -> int:
        return self.factorspace.dim ** self.power



class AbstractTensorOperator(Operator):
    """
    """


class LiftExp(AbstractExponentiator):

    def exp(
        self, 
        lift_op: LiftOperator, 
        h: ScalarLike, 
        y: TensorState) -> TensorState:
        
        return apply_along_state(lambda s: lift_op.op.exp(h, s), y, lift_op.factor_idx)

    @property
    def order(self) -> Order:
        return jnp.inf

    def effective_order(self, lift_op: LiftOperator) -> Order:
        return lift_op.op.exponentiator.order


class LiftOperator(AbstractTensorOperator):
    op: Operator
    factor_idx: int
    num_factors: int
    exponentiator: AbstractExponentiator = eqx.field(default=LiftExp(), kw_only=True)

    @property
    def domain(self):
        return tuple(self.op.domain if i == self.factor_idx else AbstractHilbertSpace for i in range(self.num_factors))

    def action(self, y: TensorState) -> TensorState:
        return apply_along_state(lambda s: self.op(s), y, self.factor_idx)

    def exp_action(self, h: ScalarLike, y: TensorState) -> TensorState:
        return apply_along_state(lambda s: self.op.exp_action(h, s), y, self.factor_idx)

    def spectral_bounds(self, hilbert_space: AbstractTensorSpace) -> Array:
        return self.op.spectral_bounds(hilbert_space[self.factor_idx])

    def solve(self, b: TensorState, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> TensorState:
        return apply_along_state(lambda s: self.op.solve(s, scale, shift), b, self.factor_idx)

    def to_matrix(self, hilbert_space: AbstractTensorSpace) -> Array:
        mat_list = [
            jnp.eye(hilbert_space[idx].dim) if idx != self.factor_idx else self.op.to_matrix(hilbert_space[idx])
            for idx in range(hilbert_space.num_factors)
        ]
        return reduce(lambda a, b: jnp.kron(a, b), mat_list)

    def to_dict(self, h_scale=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "exp_delegated": False,
            "op": self.op.to_dict(h_scale)
        }


class KroneckerSumExp(AbstractExponentiator):

    def exp(
        self, 
        kron_op: KroneckerSum, 
        h: ScalarLike, 
        y: TensorState) -> TensorState:
        
        for factor_idx, op in enumerate(kron_op.ops):
            y = apply_along_state(lambda s, op=op: op.exp(h, s), y, factor_idx)

        return y

    @property
    def order(self) -> Order:
        return jnp.inf

    def effective_order(self, kron_op: KroneckerSum) -> Order:
        return min([op.exponentiator.order for op in kron_op.ops])


class KroneckerSum(AbstractTensorOperator):
    ops: tuple[Operator, ...]
    exponentiator: AbstractExponentiator = eqx.field(default=KroneckerSumExp(), kw_only=True)

    @property
    def domain(self):
        return tuple(op.domain for op in self.ops)

    def action(self, y: TensorState) -> TensorState:
        y_list = [
            apply_along_state(lambda s, op=op: op(s), y, factor_idx)
            for factor_idx, op in enumerate(self.ops)
        ]

        return reduce(lambda a, b: a + b, y_list)

    def exp_action(self, h: ScalarLike, y: TensorState) -> TensorState:
        for factor_idx, op in enumerate(self.ops):
            y = apply_along_state(lambda s, op=op: op.exp_action(h, s), y, factor_idx)

        return y

    def spectral_bounds(self, hilbert_space: AbstractTensorSpace) -> Array:
        bounds = [op.spectral_bounds(hilbert_space[idx]) for idx, op in enumerate(self.ops)]
        return jnp.sum(jnp.array(bounds), axis=0)

    def to_matrix(self, hilbert_space: AbstractTensorSpace) -> Array:
        mat = jnp.zeros((hilbert_space.dim, hilbert_space.dim), dtype=complex)
        for factor_idx, op in enumerate(self.ops):
            mat_list = [
                jnp.eye(hilbert_space[idx].dim) if idx != factor_idx else op.to_matrix(hilbert_space[idx])
                for idx in range(hilbert_space.num_factors)
            ]
            mat += reduce(lambda a, b: jnp.kron(a, b), mat_list)
        return mat

    def to_dict(self, h_scale=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "exp_delegated": False,
            "ops": [op.to_dict(h_scale) for op in self.ops]
        }


class KroneckerProduct(AbstractTensorOperator):
    ops: tuple[Operator, ...]
    exponentiator: AbstractExponentiator = eqx.field(default=KrylovExponentiator(), kw_only=True)

    @property
    def domain(self):
        return tuple(op.domain for op in self.ops)

    def action(self, y: TensorState) -> TensorState:
        for factor_idx, op in enumerate(self.ops):
            y = apply_along_state(lambda s, op=op: op(s), y, factor_idx)

        return y

    def spectral_bounds(self, hilbert_space: AbstractTensorSpace) -> Array:
        prods = jnp.array([1.0])
        for idx, op in enumerate(self.ops):
            lo, hi = op.spectral_bounds(hilbert_space[idx])
            prods = jnp.concatenate([prods * lo, prods * hi])   
        return jnp.array([jnp.min(prods), jnp.max(prods)])

    def to_matrix(self, hilbert_space: AbstractTensorSpace) -> Array:
        mat_list = [
            op.to_matrix(hilbert_space[idx]) for idx, op in enumerate(self.ops)
        ]
        return reduce(lambda a, b: jnp.kron(a, b), mat_list)

    def to_dict(self, h_scale=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "exp_delegated": False,
            "ops": None
        }


def stack_ops(ops):
    """Stack structurally identical operators into (static, xs) for lax.scan."""
    dynamic, static = zip(*(eqx.partition(op, eqx.is_inexact_array_like) for op in ops))

    if not all(s == static[0] for s in static[1:]):
        raise ValueError(
            "Operators are not structurally identical, so they cannot be stacked; "
            "use KroneckerSum on a TensorProduct for the unrolled path."
        )

    return static[0], jax.tree.map(lambda *leaves: jnp.stack(leaves), *dynamic)


def rotate_factors(t: ArrayLike, num_factors: int) -> Array:
    """Cyclically shift the factor axes (the trailing `num_factors` axes) by one,
    bringing the next factor to axis `-num_factors`."""
    return jnp.moveaxis(t, -num_factors, -1)

def apply_and_rotate(
    fn: Callable[[ArrayLike], ArrayLike], 
    t: ArrayLike, 
    num_factors: int) -> Array:

    return rotate_factors(apply_along_tensor(fn, t, -num_factors), num_factors)


class BatchKroneckerSumExp(AbstractExponentiator):

    def exp(
        self, 
        kron_op: KroneckerSum, 
        h: ScalarLike, 
        y: TensorState) -> TensorState:
        
        num_factors = y.hilbert_space.num_factors
        factorspace = y.hilbert_space.factorspace
        static, xs = stack_ops(kron_op.ops)

        def body(T, x):
            op = eqx.combine(static, x)
            fn = lambda col: op.exp(h, factorspace.from_coeffs(col)).coeffs
            return apply_and_rotate(fn, T, num_factors), None

        T, _ = jax.lax.scan(body, y.coeff_tensor, xs, length=num_factors)
        return y.hilbert_space.from_tensor(T)

    @property
    def order(self) -> Order:
        return jnp.inf

    def effective_order(self, kron_op: KroneckerSum) -> Order:
        return min([op.exponentiator.order for op in kron_op.ops])


class BatchKroneckerSum(AbstractTensorOperator):
    ops: tuple[Operator, ...]
    exponentiator: AbstractExponentiator = eqx.field(default=BatchKroneckerSumExp(), kw_only=True)

    @property
    def domain(self):
        return tuple(op.domain for op in self.ops)

    def action(self, y: TensorState) -> TensorState:
        num_factors = y.hilbert_space.num_factors
        factorspace = y.hilbert_space.factorspace
        static, xs = stack_ops(self.ops)

        def body(carry, x):
            T, S = carry
            op = eqx.combine(static, x)
            fn = lambda col: op.action(factorspace.from_coeffs(col)).coeffs
            T_next = rotate_factors(T, num_factors)
            S_next = rotate_factors(S + apply_along_tensor(fn, T, -num_factors), num_factors)
            return (T_next, S_next), None

        T0 = y.coeff_tensor
        (_, S), _ = jax.lax.scan(body, (T0, jnp.zeros_like(T0)), xs, length=num_factors)
        return y.hilbert_space.from_tensor(S)

    def exp_action(self, h: ScalarLike, y: TensorState) -> TensorState:
        num_factors = y.hilbert_space.num_factors
        factorspace = y.hilbert_space.factorspace
        static, xs = stack_ops(self.ops)

        def body(T, x):
            op = eqx.combine(static, x)
            fn = lambda col: op.exp_action(h, factorspace.from_coeffs(col)).coeffs
            return apply_and_rotate(fn, T, num_factors), None

        T, _ = jax.lax.scan(body, y.coeff_tensor, xs, length=num_factors)
        return y.hilbert_space.from_tensor(T)

    def spectral_bounds(self, hilbert_space: AbstractTensorSpace) -> Array:
        bounds = [op.spectral_bounds(hilbert_space[idx]) for idx, op in enumerate(self.ops)]
        return jnp.sum(jnp.array(bounds), axis=0)

    def to_matrix(self, hilbert_space: AbstractTensorSpace) -> Array:
        mat = jnp.zeros((hilbert_space.dim, hilbert_space.dim), dtype=complex)
        for factor_idx, op in enumerate(self.ops):
            mat_list = [
                jnp.eye(hilbert_space[idx].dim) if idx != factor_idx else op.to_matrix(hilbert_space[idx])
                for idx in range(hilbert_space.num_factors)
            ]
            mat += reduce(lambda a, b: jnp.kron(a, b), mat_list)
        return mat

    def to_dict(self, h_scale=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "exp_delegated": False,
            "ops": [op.to_dict(h_scale) for op in self.ops]
        }


class BatchKroneckerProduct(AbstractTensorOperator):
    ops: tuple[Operator, ...]
    exponentiator: AbstractExponentiator = eqx.field(default=KrylovExponentiator(), kw_only=True)

    @property
    def domain(self):
        return tuple(op.domain for op in self.ops)

    def action(self, y: TensorState) -> TensorState:
        num_factors = y.hilbert_space.num_factors
        factorspace = y.hilbert_space.factorspace
        static, xs = stack_ops(self.ops)

        def body(T, x):
            op = eqx.combine(static, x)
            fn = lambda col: op.action(factorspace.from_coeffs(col)).coeffs
            return apply_and_rotate(fn, T, num_factors), None

        T, _ = jax.lax.scan(body, y.coeff_tensor, xs, length=num_factors)
        return y.hilbert_space.from_tensor(T)

    def spectral_bounds(self, hilbert_space: AbstractTensorSpace) -> Array:
        lo, hi = 1.0, 1.0
        for op in self.ops:
            a, b = op.spectral_bounds(hilbert_space.factorspace)
            c = jnp.array([lo*a, lo*b, hi*a, hi*b])
            lo, hi = jnp.min(c), jnp.max(c)
        return jnp.array([jnp.min(prods), jnp.max(prods)])

    def to_matrix(self, hilbert_space: AbstractTensorSpace) -> Array:
        mat_list = [
            op.to_matrix(hilbert_space[idx]) for idx, op in enumerate(self.ops)
        ]
        return reduce(lambda a, b: jnp.kron(a, b), mat_list)

    def to_dict(self, h_scale=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "exp_delegated": False,
            "ops": None
        }

from __future__ import annotations

from typing import Callable, ClassVar
from functools import reduce

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike, ScalarLike

from .hilbert_space import AbstractHilbertSpace, AbstractState
from .operator import Operator
from .exponentiators import Order, AbstractExponentiator, ExactExponentiator, KrylovExponentiator


class TensorProductState(AbstractState):
    hilbert_space: TensorProduct = eqx.field(static=True)

    @property
    def coeff_tensor(self) -> Array:
        return self.coeffs.reshape(*self.coeffs.shape[:-1], *self.hilbert_space.dim_list)

    @property
    def spaces(self) -> list[AbstractHilbertSpace]:
        return self.hilbert_space.spaces


class TensorProduct(AbstractHilbertSpace):
    state_type: ClassVar = TensorProductState
    spaces: tuple[AbstractHilbertSpace]

    @property
    def dim(self) -> int:
        return math.prod(self.dim_list)

    @property
    def dim_list(self) -> list[int]:
        return [h.dim for h in self.spaces]

    def from_tensor(self, coeff_tensor: ArrayLike):
        return self.from_coeffs(coeff_tensor.reshape((self.dim,)))

    def product_state(self, y_list: tuple[AbstractState]) -> TensorProductState:
        coeffs = reduce(lambda a, b: jnp.outer(a, b).flatten(), [y.coeffs for y in y_list])
        return self.from_coeffs(coeffs)


def apply_along_axis(
    fn: Callable[[ArrayLike, int], ArrayLike], 
    y: TensorProductState, 
    axis: int) -> TensorProductState:

    tensor_in = jnp.moveaxis(y.coeff_tensor, axis, 0)
    shape = tensor_in.shape
    tensor_out = jax.vmap(fn, in_axes=(1, None), out_axes=1)(tensor_in.reshape(shape[0], -1), axis)
    coeffs_tensor = jnp.moveaxis(tensor_out.reshape(shape), 0, axis)

    return y.hilbert_space.from_tensor(coeffs_tensor)


class LiftExp(AbstractExponentiator):

    def exp(
        self, 
        lift_op: LiftOperator, 
        dt: ScalarLike, 
        y: TensorProductState) -> TensorProductState:
        
        def fn(coeffs, idx):
            y_i = y.spaces[idx].from_coeffs(coeffs)
            return lift_op.op.exp(dt, y_i).coeffs

        return apply_along_axis(fn, y, lift_op.idx)

    @property
    def order(self) -> Order:
        return jnp.inf

    def effective_order(self, lift_op: LiftOperator) -> Order:
        return lift_op.op.exponentiator.order


class KroneckerSumExp(AbstractExponentiator):

    def exp(
        self, 
        kron_op: KroneckerSum, 
        dt: ScalarLike, 
        y: TensorProductState) -> TensorProductState:
        
        def fn(coeffs, idx):
            y_i = y.spaces[idx].from_coeffs(coeffs)
            return kron_op.ops[idx].exp(dt, y_i).coeffs

        for idx, op in enumerate(kron_op.ops):
            y = apply_along_axis(fn, y, idx)

        return y

    @property
    def order(self) -> Order:
        return jnp.inf

    def effective_order(self, kron_op: KroneckerSum) -> Order:
        return min([op.exponentiator.order for op in kron_op.ops])


class AbstractTensorOperator(Operator):
    domain: ClassVar = TensorProduct

    def _check_axis_domain(self, y: TensorProductState, idx: int, sub_op: Operator):
        hilbert_space = y.spaces[idx]

        if not isinstance(hilbert_space, sub_op.domain):
            raise TypeError(
                f"{type(sub_op).__name__} acts on {sub_op.domain.__name__}, "
                f"but received a state on {type(hilbert_space).__name__}"
            )

    def _check_op_list(self, y: TensorProductState, op_list):
        if len(y.spaces) != len(op_list):
            raise TypeError(
                f"Number of spaces in tensor product differs from number of operands in Kronecker operator: "
                f"len(y.spaces) = {len(y.spaces)} but len(self.ops) = {len(op_list)}"
            )

        for idx, op in enumerate(op_list):
            self._check_axis_domain(y, idx, op)


class LiftOperator(AbstractTensorOperator):
    domain: ClassVar = TensorProduct
    exponentiator: AbstractExponentiator = eqx.field(default=LiftExp(), kw_only=True)
    op: Operator
    idx: int

    def _check_domain(self, y: TensorProductState):
        self._check_axis_domain(y, self.idx, self.op)

    def action(self, y: TensorProductState) -> TensorProductState:
        def fn(coeffs, idx):
            y_i = y.spaces[idx].from_coeffs(coeffs)
            return self.op(y_i).coeffs

        return apply_along_axis(fn, y, self.idx)

    def exp_action(self, dt: ScalarLike, y: TensorProductState) -> TensorProductState:
        def fn(coeffs, idx):
            y_i = y.spaces[idx].from_coeffs(coeffs)
            return self.op.exp_action(dt, y_i).coeffs

        return apply_along_axis(fn, y, self.idx)

    def spectral_bounds(self, hilbert_space: TensorProduct) -> Array:
        return self.op.spectral_bounds(hilbert_space.spaces[self.idx])

    def solve(self, b: TensorProductState, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> TensorProductState:
        def fn(coeffs, idx):
            b_i = b.spaces[idx].from_coeffs(coeffs)
            return self.op.solve(b_i, scale, shift).coeffs

        return apply_along_axis(fn, b, self.idx)

    def to_matrix(self, hilbert_space: TensorProduct) -> Array:
        mat_list = [
            jnp.eye(hs.dim) if jdx != self.idx else self.op.to_matrix(hs) 
            for jdx, hs in enumerate(hilbert_space.spaces)
        ]
        return reduce(lambda a, b: jnp.kron(a, b), mat_list)


class KroneckerSum(AbstractTensorOperator):
    domain: ClassVar = TensorProduct
    exponentiator: AbstractExponentiator = eqx.field(default=KroneckerSumExp(), kw_only=True)
    ops: tuple[Operator, ...]

    def _check_domain(self, y: TensorProductState):
        self._check_op_list(y, self.ops)

    def action(self, y: TensorProductState) -> TensorProductState:
        def fn(coeffs, idx):
            y_i = y.spaces[idx].from_coeffs(coeffs)
            return self.ops[idx](y_i).coeffs

        y_list = [apply_along_axis(fn, y, idx) for idx, op in enumerate(self.ops)]
        return reduce(lambda a, b: a + b, y_list) 

    def exp_action(self, dt: ScalarLike, y: TensorProductState) -> TensorProductState:
        def fn(coeffs, idx):
            y_i = y.spaces[idx].from_coeffs(coeffs)
            return self.ops[idx].exp_action(dt, y_i).coeffs

        for idx, op in enumerate(self.ops):
            y = apply_along_axis(fn, y, idx)

        return y

    def spectral_bounds(self, hilbert_space: TensorProduct) -> Array:
        bounds = [op.spectral_bounds(space) for op, space in zip(self.ops, hilbert_space.spaces)]
        return jnp.sum(jnp.array(bounds), axis=0)

    def to_matrix(self, hilbert_space: TensorProduct) -> Array:
        mat = jnp.zeros((hilbert_space.dim, hilbert_space.dim), dtype=complex)
        for idx, op in enumerate(self.ops):
            mat_list = [
                jnp.eye(hs.dim) if jdx != idx else op.to_matrix(hs) 
                for jdx, hs in enumerate(hilbert_space.spaces)
            ]
            mat += reduce(lambda a, b: jnp.kron(a, b), mat_list)
        return mat


class KroneckerProduct(AbstractTensorOperator):
    domain: ClassVar = TensorProduct
    exponentiator: AbstractExponentiator = eqx.field(default=KrylovExponentiator(), kw_only=True)
    ops: tuple[Operator, ...]

    def _check_domain(self, y: TensorProductState):
        self._check_op_list(y, self.ops)

    def action(self, y: TensorProductState) -> TensorProductState:
        def fn(coeffs, idx):
            y_i = y.spaces[idx].from_coeffs(coeffs)
            return self.ops[idx](y_i).coeffs

        for idx, op in enumerate(self.ops):
            y = apply_along_axis(fn, y, idx)

        return y

    def spectral_bounds(self, hilbert_space: TensorProduct) -> Array:
        prods = jnp.array([1.0])
        for op, space in zip(self.ops, hilbert_space.spaces):
            lo, hi = op.spectral_bounds(space)
            prods = jnp.concatenate([prods * lo, prods * hi])   
        return jnp.array([jnp.min(prods), jnp.max(prods)])

    def to_matrix(self, hilbert_space: TensorProduct) -> Array:
        mat_list = [
            op.to_matrix(hs) for op, hs in zip(self.ops, hilbert_space.spaces)
        ]
        return reduce(lambda a, b: jnp.kron(a, b), mat_list)


class TensorPower(TensorProduct):
    state_type: ClassVar = TensorProductState
    subspace: AbstractHilbertSpace
    power: int

    @property
    def dim(self) -> int:
        return self.subspace.dim ** self.power

    @property
    def dim_list(self) -> list[int]:
        return [self.subspace.dim for h in range(self.power)]




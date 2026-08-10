from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike, ScalarLike, PRNGKeyArray

if TYPE_CHECKING:
    from .operator import Operator


type Index = Union[int, tuple[int, ...]]


def _to_tuple(idx: Index) -> tuple[int, ...]:
    if isinstance(idx, tuple):
        return idx 

    return (idx,)


def _coeff_index(idx: Index) -> tuple[int, ...]:
    return _to_tuple(idx) + (slice(None),)


class AbstractHilbertSpace(eqx.Module):
    state_type: eqx.AbstractClassVar[type[AbstractState]]

    @property
    def structure(self) -> type[AbstractHilbertSpace]:
        return type(self)

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    def batch_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return shape + (self.dim,)

    def innerp(self, y1: AbstractState, y2: AbstractState) -> ScalarLike:
        return jnp.sum(jnp.conj(y1.coeffs) * y2.coeffs, axis=-1)

    def norm(self, y: AbstractState) -> ScalarLike:
        return jnp.sqrt(self.norm2(y))

    def norm2(self, y: AbstractState) -> ScalarLike:
        return jnp.real(self.innerp(y, y))

    def expected_value(self, op: Operator, y: AbstractState) -> ScalarLike:
        return jnp.real(self.innerp(y, op(y)))

    def from_coeffs(self, coeffs: ArrayLike) -> AbstractState:
        return self.state_type(coeffs, self)

    def zeros(self, shape: tuple[int, ...]=()) -> AbstractState:
        batch_shape = self.batch_shape(shape)
        return self.from_coeffs(jnp.zeros(batch_shape))

    def random(self, key: PRNGKeyArray, shape: tuple[int, ...]=()) -> AbstractState:
        batch_shape = self.batch_shape(shape)
        random_coeffs = jax.random.normal(key, shape=batch_shape, dtype=complex)
        return self.from_coeffs(random_coeffs)

    def zeros_like(self, y: AbstractState) -> AbstractState:
        return self.state_type(jnp.zeros_like(y.coeffs), self)


class AbstractState(eqx.Module):
    coeffs: Array
    hilbert_space: AbstractHilbertSpace = eqx.field(static=True)

    def __init__(self, coeffs, hilbert_space):
        self.coeffs = jnp.asarray(coeffs, dtype=complex)
        self.hilbert_space = hilbert_space

    def binary_op(
        self, 
        other: AbstractState, 
        fn: Callable, 
        on_coeffs: bool=True) -> Union[AbstractState, ScalarLike]:

        if not isinstance(other, AbstractState):
            return NotImplemented

        if self.hilbert_space != other.hilbert_space:
            raise ValueError("Cannot compose vectors from different spaces")

        if on_coeffs:
            return self.hilbert_space.from_coeffs(fn(self.coeffs, other.coeffs))
        else:
            return fn(self, other)

    def innerp(self, y: AbstractState) -> ScalarLike:
        return self.hilbert_space.innerp(self, y)

    def norm(self) -> Array:
        return self.hilbert_space.norm(self)

    def norm2(self) -> Array:
        return self.hilbert_space.norm2(self)

    def expected_value(self, op: Operator) -> Array:
        return self.hilbert_space.expected_value(op, self)

    def __add__(self, other: AbstractState) -> AbstractState:
        return self.binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: AbstractState) -> AbstractState:
        return self.binary_op(other, lambda a, b: b + a)

    def __sub__(self, other: AbstractState) -> AbstractState:
        return self.binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: AbstractState) -> AbstractState:
        return self.binary_op(other, lambda a, b: b - a)

    def __matmul__(self, other: AbstractState) -> ScalarLike:
        return self.binary_op(
            other, lambda a, b, hs=self.hilbert_space: hs.innerp(a, b), on_coeffs=False)

    def __rmatmul__(self, other: AbstractState) -> ScalarLike:
        return self.binary_op(
            other, lambda a, b, hs=self.hilbert_space: hs.innerp(b, a), on_coeffs=False)

    def __mul__(self, other: ScalarLike) -> AbstractState:
        if not jnp.isscalar(other):
            return NotImplemented

        return self.hilbert_space.from_coeffs(other * self.coeffs)

    def __rmul__(self, other: ScalarLike) -> AbstractState:
        if not jnp.isscalar(other):
            return NotImplemented

        return self.hilbert_space.from_coeffs(other * self.coeffs)

    def __truediv__(self, other: ScalarLike) -> AbstractState:
        if not jnp.isscalar(other):
            return NotImplemented

        return self.hilbert_space.from_coeffs(self.coeffs / other)

    def __neg__(self) -> AbstractState:
        return self.hilbert_space.from_coeffs(-self.coeffs)

    def contract(
        self, 
        weights: ArrayLike, 
        axes: Union[int, tuple]=(0, 0)) -> AbstractState:

        rank = len(self.shape)

        if isinstance(axes, int):
            w_axes, c_axes = tuple(range(axes)), tuple(range(axes))
        else:
            w_axes, c_axes = axes
            w_axes, c_axes = _to_tuple(w_axes), _to_tuple(c_axes)

        c_axes = tuple(a + rank if a < 0 else a for a in c_axes)

        if any(not 0 <= a < rank for a in c_axes):
            raise ValueError(
                f"axes {c_axes} out of range for a state with {rank} batch axes; "
                "the coefficient axis cannot be contracted")

        new_coeffs = jnp.tensordot(weights, self.coeffs, axes=(w_axes, c_axes))
        return self.hilbert_space.from_coeffs(new_coeffs)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.coeffs.shape[:-1]

    def __getitem__(self, idx: Index) -> AbstractState:            
        return self.hilbert_space.from_coeffs(self.coeffs[_coeff_index(idx)])

    @property
    def at(self) -> _AbstractStateIndexHelper:
        return _AbstractStateIndexHelper(self)


class _AbstractStateIndexHelper:

    def __init__(self, state: AbstractState):
        self.state = state 

    def __getitem__(self, idx: Index) -> _AbstractStateIndexSetter:
        return _AbstractStateIndexSetter(self.state, idx)


class _AbstractStateIndexSetter:

    def __init__(self, state: AbstractState, idx: Index):
        self.state = state 
        self.idx = idx

    def set(self, values: AbstractState) -> AbstractState:
        if self.state.hilbert_space != values.hilbert_space:
            raise ValueError("Cannot set batched state with values from a different Hilbert space")

        new_coeffs = self.state.coeffs.at[_coeff_index(self.idx)].set(values.coeffs)
        return self.state.hilbert_space.from_coeffs(new_coeffs)

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar, Union
from abc import abstractmethod

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike, Scalar, ScalarLike

if TYPE_CHECKING:
    from .operator import Operator


class AbstractHilbertSpace(eqx.Module):
    state_type: eqx.AbstractClassVar[type[AbstractState]]

    @property
    def structure(self) -> type[AbstractHilbertSpace]:
        return type(self)

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    def innerp(self, y1: AbstractState, y2: AbstractState) -> ScalarLike:
        return self.from_coeffs(jnp.sum(jnp.conj(y1.coeffs) * y2.coeffs, axis=-1))

    def norm(self, y: AbstractState) -> ScalarLike:
        return jnp.sqrt(jnp.real(self.innerp(y, y)))

    def expected_value(self, op: Operator, y: AbstractState) -> Array:
        return jnp.real(self.innerp(y, op(y)))

    def from_coeffs(self, coeffs) -> AbstractState:
        return self.state_type(coeffs, self)

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

        if self.hilbert_space != other.hilbert_space:
            raise ValueError("Cannot compose vectors from different spaces")

        if not isinstance(other, AbstractState):
            return NotImplemented

        if on_coeffs:
            return self.hilbert_space.from_coeffs(fn(self.coeffs, other.coeffs))
        else:
            return fn(self, other)

    def innerp(self, y: AbstractState) -> ScalarLike:
        return self.hilbert_space.innerp(self, y)

    def norm(self) -> Array:
        return self.hilbert_space.norm(self)

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

    @property
    def shape(self) -> tuple[int, ...]:
        return self.coeffs.shape[:-1]

    def __getitem__(self, idx) -> AbstractState:
        if not isinstance(idx, tuple):
            idx = (idx,)
        return self.hilbert_space.from_coeffs(self.coeffs[idx + (slice(None),)])


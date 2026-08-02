from __future__ import annotations

from typing import Callable, ClassVar
from abc import abstractmethod

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike, Scalar, ScalarLike


class AbstractHilbertSpace(eqx.Module):
    state_type: eqx.AbstractClassVar[type[AbstractState]]

    @property
    def structure(self) -> type[AbstractHilbertSpace]:
        return type(self)

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

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

    def binary_op(self, other: AbstractState, op: Callable) -> AbstractState:
        if self.hilbert_space != other.hilbert_space:
            raise ValueError("Cannot compose vectors from different spaces")

        return self.hilbert_space.from_coeffs(op(self.coeffs, other.coeffs))

    def __add__(self, other: AbstractState) -> AbstractState:
        return self.binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: AbstractState) -> AbstractState:
        return self.binary_op(other, lambda a, b: b + a)

    def __sub__(self, other: AbstractState) -> AbstractState:
        return self.binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: AbstractState) -> AbstractState:
        return self.binary_op(other, lambda a, b: b - a)

    def __mul__(self, other: ScalarLike) -> AbstractState:
        return self.hilbert_space.from_coeffs(other * self.coeffs)

    def __rmul__(self, other: ScalarLike) -> AbstractState:
        return self.hilbert_space.from_coeffs(other * self.coeffs)

    def __truediv__(self, other: ScalarLike) -> AbstractState:
        return self.hilbert_space.from_coeffs(self.coeffs / other)

    def __neg__(self) -> AbstractState:
        return self.hilbert_space.from_coeffs(-self.coeffs)

    def __getitem__(self, idx) -> AbstractState:
        return self.hilbert_space.from_coeffs(self.coeffs[idx, :])

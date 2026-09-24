from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Iterable, Self, Sequence, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar, ScalarLike

if TYPE_CHECKING:
    from .operator import Operator


type Shape = tuple[int, ...]

type Index = Any


def _to_tuple(idx: Index) -> tuple[Index, ...]:
    if isinstance(idx, tuple):
        return idx 
    return (idx,)


def _coeff_index(idx: Index) -> tuple[Index, ...]:
    return _to_tuple(idx) + (slice(None),)


class AbstractHilbertSpace[S: AbstractState[Any]](eqx.Module):
    state_type: eqx.AbstractClassVar[type[AbstractState]]
    hbar: float = eqx.field(default=1.0, converter=float, kw_only=True)

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    def batch_shape(self, shape: Shape) -> Shape:
        return shape + (self.dim,)

    def innerp(self, y1: S, y2: S) -> Scalar:
        return jnp.sum(jnp.conj(y1.coeffs) * y2.coeffs, axis=-1)

    def norm(self, y: S) -> Scalar:
        return jnp.sqrt(self.norm2(y))

    def norm2(self, y: S) -> Scalar:
        return jnp.real(self.innerp(y, y))

    def expected_value(self, op: Operator, y: S) -> Scalar:
        return self.innerp(y, op(y))

    def from_coeffs(self, coeffs: ArrayLike) -> S:
        coeffs = jnp.asarray(coeffs, dtype=complex)
        # state_type is a ClassVar, which cannot mention S
        return cast(S, self.state_type(coeffs, hilbert_space=self))

    def zeros(self, shape: Shape=()) -> S:
        batch_shape = self.batch_shape(shape)
        return self.from_coeffs(jnp.zeros(batch_shape))

    def random(self, key: PRNGKeyArray, shape: Shape=(), dtype=complex) -> S:
        batch_shape = self.batch_shape(shape)
        random_coeffs = jax.random.normal(key, shape=batch_shape, dtype=dtype)
        return self.from_coeffs(random_coeffs)

    def zeros_like(self, y: S) -> S:
        return self.from_coeffs(jnp.zeros_like(y.coeffs))

    def stack(self, ys: Sequence[S], axis=0) -> S:
        if any(y.hilbert_space != self for y in ys):
            raise ValueError("Cannot join states from different spaces")    
        
        max_rank = max([y.rank for y in ys])

        if axis < 0:
            axis += max_rank + 1

        if not 0 <= axis < max_rank + 1:
            raise ValueError(f"axis {axis} out of range for sequence with of states with maximum rank {max_rank}")
        
        return self.from_coeffs(jnp.stack([y.coeffs for y in ys], axis))

    def concatenate(self, ys: Sequence[S], axis=0) -> S:
        if any(y.hilbert_space != self for y in ys):
            raise ValueError("Cannot join states from different spaces")

        max_rank = max([y.rank for y in ys])

        if axis < 0:
            axis += max_rank

        if not 0 <= axis < max_rank:
            raise ValueError(f"axis {axis} out of range for sequence with of states with maximum rank {max_rank}")

        coeffs = [y.coeffs[(None,) * (max_rank - y.rank) + (Ellipsis,)] for y in ys]
        return self.from_coeffs(jnp.concatenate(coeffs, axis))

    def identity(self) -> Operator[S]:
        from .operator import Identity
        return Identity(self)

    def zero_operator(self) -> Operator[S]:
        from .operator import Zero
        return Zero(self)


class AbstractState[H: AbstractHilbertSpace[Any]](eqx.Module):
    coeffs: Array
    hilbert_space: H = eqx.field(static=True, kw_only=True)

    def __init__(self, coeffs: ArrayLike, hilbert_space: H):
        self.coeffs = jnp.asarray(coeffs, dtype=complex)
        self.hilbert_space = hilbert_space

    def _check_compatible(self, other: AbstractState):
        if self.hilbert_space != other.hilbert_space:
            raise ValueError("Cannot compose vectors from different spaces")

    def binary_op(self, other: Self, fn: Callable) -> Self:
        if not isinstance(other, AbstractState):
            return NotImplemented

        self._check_compatible(other)
        return self.hilbert_space.from_coeffs(fn(self.coeffs, other.coeffs))

    def innerp(self, y: Self) -> Scalar:
        return self.hilbert_space.innerp(self, y)

    def norm(self) -> Scalar:
        return self.hilbert_space.norm(self)

    def norm2(self) -> Scalar:
        return self.hilbert_space.norm2(self)

    def expected_value(self, op: Operator) -> Scalar:
        return self.hilbert_space.expected_value(op, self)

    def __add__(self, other: Self) -> Self:
        return self.binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: Self) -> Self:
        return self.binary_op(other, lambda a, b: b + a)

    def __sub__(self, other: Self) -> Self:
        return self.binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: Self) -> Self:
        return self.binary_op(other, lambda a, b: b - a)

    def __matmul__(self, other: Self) -> Scalar:
        if not isinstance(other, AbstractState):
            return NotImplemented

        self._check_compatible(other)

        return self.hilbert_space.innerp(self, other)

    def __mul__(self, other: ScalarLike) -> Self:
        if not jnp.isscalar(other):
            return NotImplemented

        return self.hilbert_space.from_coeffs(other * self.coeffs)

    def __rmul__(self, other: ScalarLike) -> Self:
        if not jnp.isscalar(other):
            return NotImplemented

        return self.hilbert_space.from_coeffs(other * self.coeffs)

    def __truediv__(self, other: ScalarLike) -> Self:
        if not jnp.isscalar(other):
            return NotImplemented

        return self.hilbert_space.from_coeffs(self.coeffs / other)

    def __neg__(self) -> Self:
        return self.hilbert_space.from_coeffs(-self.coeffs)

    def contract(
        self,
        weights: Array,
        axes: int | Iterable[int]=(0, 0)) -> Self:
        """
        Takes a linear combination of states corresponding to batch axes
        """

        if isinstance(axes, int):
            w_axes, c_axes = tuple(range(axes)), tuple(range(axes))
        else:
            w_axes, c_axes = axes
            w_axes, c_axes = _to_tuple(w_axes), _to_tuple(c_axes)

        c_axes = tuple(self.coeff_axis(a) for a in c_axes)
        new_coeffs = jnp.tensordot(weights, self.coeffs, axes=(w_axes, c_axes))
        return self.hilbert_space.from_coeffs(new_coeffs)

    @property
    def shape(self) -> Shape:
        return self.coeffs.shape[:-1]

    @property
    def rank(self) -> int:
        return len(self.shape)

    def coeff_axis(self, axis: int) -> int:
        if axis < 0:
            axis += self.rank 

        if not 0 <= axis < self.rank:
            raise ValueError(f"axis={axis} out of range for batched state vector of rank {self.rank}")

        return axis

    def __getitem__(self, idx: Index) -> Self:
        return self.hilbert_space.from_coeffs(self.coeffs[_coeff_index(idx)])

    @property
    def at(self) -> _AbstractStateIndexHelper[Self]:
        return _AbstractStateIndexHelper(self)


class _AbstractStateIndexHelper[S: AbstractState[Any]]:

    def __init__(self, state: S):
        self.state = state

    def __getitem__(self, idx: Index) -> _AbstractStateIndexSetter[S]:
        return _AbstractStateIndexSetter(self.state, idx)


class _AbstractStateIndexSetter[S: AbstractState[Any]]:

    def __init__(self, state: S, idx: Index):
        self.state = state
        self.idx = idx

    def set(self, values: S) -> S:
        if self.state.hilbert_space != values.hilbert_space:
            raise ValueError("Cannot set batched state with values from a different Hilbert space")

        new_coeffs = self.state.coeffs.at[_coeff_index(self.idx)].set(values.coeffs)
        return self.state.hilbert_space.from_coeffs(new_coeffs)

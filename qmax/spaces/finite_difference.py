from typing import Callable, ClassVar, Union

from functools import reduce

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Scalar, ScalarLike

from ..hilbert_space import AbstractHilbertSpace, AbstractState
from ..operator import Operator
from ..exponentiators import AbstractExponentiator, ExactExponentiator, CrankNicolson
from ..tensor import TensorProduct, TensorProductState, KroneckerSum
from ..utils import over_batch
from .spatial_discretization import SpatialDiscretization, SpatiallyDiscretizedState, _to_tuple


class FiniteDifference1DState(SpatiallyDiscretizedState):

    @property
    def values(self) -> Array:
        return self.coeffs


class FiniteDifference1D(SpatialDiscretization):
    state_type: ClassVar = FiniteDifference1DState
    endpoint: ClassVar[bool] = True

    def __init__(self, x0: ScalarLike, xf: ScalarLike, mesh_size: int):
        self.x0 = (float(x0),)
        self.xf = (float(xf),)
        self.mesh_size = (mesh_size,)

    @property
    def dim(self) -> Array:
        return self.mesh_size[0]

    def from_values(self, values: ArrayLike) -> FiniteDifference1DState:
        return self.from_coeffs(values)

    @property
    def dx(self) -> ScalarLike:
        return self.dx_range[0]


class FiniteDifference1DLaplacian(Operator):
    domain: ClassVar[type[AbstractHilbertSpace]] = FiniteDifference1D
    exponentiator: AbstractExponentiator = eqx.field(default=CrankNicolson(), kw_only=True)

    def action(self, y: FiniteDifference1DState) -> FiniteDifference1DState:
        values_next = jnp.concatenate(
            [y.values[..., 1:], jnp.zeros_like(y.values[..., :1])], axis=-1)
        values_prev = jnp.concatenate(
            [jnp.zeros_like(y.values[..., :1]), y.values[..., :-1]], axis=-1)
        lapl_values = (values_next - 2 * y.values + values_prev) / (y.hilbert_space.dx ** 2)
        return y.hilbert_space.from_values(lapl_values)

    def solve(
        self, 
        b: FiniteDifference1DState, 
        scale: ScalarLike=-1.0, 
        shift: ScalarLike=0.0) -> FiniteDifference1DState:

        hilbert_space = b.hilbert_space
        dim = hilbert_space.dim
        dx = hilbert_space.dx

        diag = jnp.full(dim, shift - 2 * scale / (dx ** 2), dtype=complex)
        lower_diag = scale * jnp.full(dim - 1, 1 / (dx ** 2), dtype=complex)
        upper_diag = scale * jnp.full(dim - 1, 1 / (dx ** 2), dtype=complex)

        lx_op = lx.TridiagonalLinearOperator(diag, lower_diag, upper_diag)

        # lineax operators carry a fixed input/output structure, so a batched
        # vector has to be mapped over rather than passed through.
        def fn(b_i):
            sol = lx.linear_solve(lx_op, b_i.values)
            return hilbert_space.from_values(sol.value)

        return over_batch(fn, b)

    def spectral_bounds(self, hilbert_space: FiniteDifference1D) -> Array:
        return jnp.array([-4 / hilbert_space.dx ** 2, 0.0])

    def to_matrix(self, hilbert_space: FiniteDifference1D) -> Array:
        dim = hilbert_space.dim
        dx = hilbert_space.dx
        L = (-2 * jnp.eye(dim) + jnp.eye(dim, k=1) + jnp.eye(dim, k=-1)) / (dx ** 2)
        return L


class FiniteDifferenceState(SpatiallyDiscretizedState, TensorProductState):

    @property
    def values(self) -> Array:
        return self.hilbert_space.to_grid(self.coeffs)


class FiniteDifference(SpatialDiscretization, TensorProduct):
    state_type: ClassVar[type[AbstractState]] = FiniteDifferenceState
    endpoint: ClassVar[bool] = True

    def __init__(self, x0s: ArrayLike, xfs: ArrayLike, num_steps: Union[int, tuple[int]]):
        if isinstance(num_steps, int):
            self.spaces = (FiniteDifference1D(float(x0s), float(xfs), num_steps),)
        else:
            self.spaces = tuple(FiniteDifference1D(float(x0), float(xf), n) for x0, xf, n in zip(x0s, xfs, num_steps))

        self.x0 = _to_tuple(x0s)
        self.xf = _to_tuple(xfs)
        self.mesh_size = num_steps

    @property
    def structure(self):
        return type(self)

    @property
    def dim(self) -> Array:
        return reduce(lambda a, b: a * b, self.mesh_size)

    def from_values(self, values: ArrayLike) -> FiniteDifferenceState:
        return self.from_coeffs(self.flatten(values))


class FiniteDifferenceLaplacian(KroneckerSum):
    domain: ClassVar[type[AbstractHilbertSpace]] = FiniteDifference

    def __init__(self, spatial_dim: int):
        self.ops = tuple(FiniteDifference1DLaplacian() for _ in range(spatial_dim))


class FiniteDifferencePotentialEnergy(Operator):
    domain: ClassVar[type[AbstractHilbertSpace]] = FiniteDifference
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)
    potential: Callable[[ArrayLike], ScalarLike]

    def values(self, fd: FiniteDifference) -> Array:
        return fd.eval(self.potential)

    def action(self, y: FiniteDifferenceState) -> FiniteDifferenceState:
        return y.hilbert_space.from_values(self.values(y.hilbert_space) * y.values)

    def exp_action(self, h: ScalarLike, y: FiniteDifferenceState) -> FiniteDifferenceState:
        return y.hilbert_space.from_values(jnp.exp(h * self.values(y.hilbert_space)) * y.values)

    def solve(
        self, 
        b: FiniteDifferenceState, 
        scale: ScalarLike=-1.0, 
        shift: ScalarLike=0.0) -> FiniteDifferenceState:

        hilbert_space = b.hilbert_space
        values = self.values(hilbert_space)
        return hilbert_space.from_coeffs(b.coeffs / (scale * hilbert_space.flatten(values) + shift))

    def spectral_bounds(self, fd: FiniteDifference) -> Array:
        values = self.values(fd)
        return jnp.array([jnp.min(values), jnp.max(values)])

    def to_matrix(self, fd: FiniteDifference) -> Array:
        return jnp.diag(fd.flatten(self.values(fd)))

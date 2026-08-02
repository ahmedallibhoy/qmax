from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ScalarLike

from ..hilbert_space import AbstractState, AbstractHilbertSpace
from ..operator import Operator
from ..exponentiators import AbstractExponentiator, ExactExponentiator


class TwoLevelState(AbstractState):
    """
    """


class TwoLevel(AbstractHilbertSpace):
    state_type: ClassVar = TwoLevelState

    @property
    def dim(self) -> int:
        return 2


S_X = jnp.array([[0., 1.], [1., 0.]], dtype=complex)
S_Y = jnp.array([[0., -1j], [1j, 0.]], dtype=complex)
S_Z = jnp.array([[1., 0.], [0., -1.]], dtype=complex)


class PauliOperator(Operator):
    domain: ClassVar = TwoLevel
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)
    matrix: Array

    def __init__(self, axis):
        if axis not in ("x", "y", "z"):
            raise ValueError(
                f"Invalid axis={axis!r}; axis must be one of 'x', 'y', or 'z'."
            )

        match axis:
            case "x":
                self.matrix = S_X
            case "y":
                self.matrix = S_Y
            case "z":
                self.matrix = S_Z

    def action(self, y: TwoLevelState) -> TwoLevelState:
        return y.hilbert_space.from_coeffs((self.matrix @ y.coeffs[..., None])[..., 0])

    def exp_action(self, h: ScalarLike, y: TwoLevelState) -> TwoLevelState:
        return jnp.cosh(h) * y + jnp.sinh(h) * self.action(y)

    def spectral_bounds(self, hilbert_space: TwoLevel) -> Array:
        return jnp.array([-1, 1])

    def solve(self, b: TwoLevelState, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> AbstractState:
        A = scale * self.matrix + shift * jnp.eye(2)
        return b.hilbert_space.from_coeffs(jnp.linalg.solve(A, b.coeffs[..., None])[..., 0])

    def to_matrix(self, hilbert_space: TwoLevel) -> Array:
        return self.matrix

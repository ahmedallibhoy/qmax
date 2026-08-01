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


class PauliOperator(Operator):
    domain: ClassVar = TwoLevel
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)
    axis: str

    def __check_init__(self):
        if self.axis not in ("x", "y", "z"):
            raise ValueError(
                f"Invalid axis={self.axis!r}; axis must be one of 'x', 'y', or 'z'."
            )

    def action(self, y: TwoLevelState) -> TwoLevelState:
        return TwoLevelState(self.to_matrix(y.hilbert_space) @ y.coeffs, y.hilbert_space)

    def exp_action(self, h: ScalarLike, y: TwoLevelState) -> TwoLevelState:
        return jnp.cosh(h) * y + jnp.sinh(h) * self.action(y)

    def spectral_bounds(self, hilbert_space: TwoLevel) -> Array:
        return jnp.array([-1, 1])

    def solve(self, b: TwoLevel, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> AbstractState:
        A = shift * jnp.eye(2) + scale * self.to_matrix(b.hilbert_space)
        lx_op = lx.MatrixLinearOperator(A)
        lx_op = lx.TaggedLinearOperator(lx_op, lx.symmetric_tag)
        sol = lx.linear_solve(lx_op, b.coeffs)
        return b.hilbert_space.from_coeffs(sol.value)

    def to_matrix(self, hilbert_space: TwoLevel) -> Array:
        match self.axis:
            case "x":
                return jnp.array([[0., 1.], [1., 0.]], dtype=complex)
            case "y":
                return jnp.array([[0., -1j], [1j, 0.]], dtype=complex)
            case "z":
                return jnp.array([[1., 0.], [0., -1.]], dtype=complex)

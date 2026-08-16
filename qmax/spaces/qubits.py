from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ScalarLike

from ..hilbert_space import AbstractState, AbstractHilbertSpace
from ..operator import Operator, Identity
from ..exponentiators import AbstractExponentiator, ExactExponentiator
from ..tensor import TensorState, TensorPower, KroneckerProduct, BatchKroneckerProduct


class TwoLevelState(AbstractState):
    """
    """


class TwoLevel(AbstractHilbertSpace):
    state_type: ClassVar = TwoLevelState

    @property
    def dim(self) -> int:
        return 2


S_I = jnp.eye(2, dtype=complex)
S_X = jnp.array([[0., 1.], [1., 0.]], dtype=complex)
S_Y = jnp.array([[0., -1j], [1j, 0.]], dtype=complex)
S_Z = jnp.array([[1., 0.], [0., -1.]], dtype=complex)


class AbstractPauliOperator(Operator):

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def is_unitary(self) -> bool:
        return True

    def exp_action(self, h: ScalarLike, y: QubitsState) -> QubitsState:                    
        return jnp.cosh(h) * y + jnp.sinh(h) * self.action(y)

    def spectral_bounds(self, hilbert_space: Qubits) -> Array:     
        return jnp.array([-1.0, 1.0])

    def solve(self, 
        b: QubitsState, 
        scale: ScalarLike=-1.0, 
        shift: ScalarLike=0.0) -> Qubits:           
        
        return (shift * b - scale * self.action(b)) / (shift ** 2 - scale ** 2)


class PauliOperator(AbstractPauliOperator):
    domain: ClassVar = TwoLevel
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)
    matrix: Array

    def __init__(self, axis):
        if axis not in ("i", "x", "y", "z"):
            raise ValueError(
                f"Invalid axis={axis!r}; axis must be one of 'i', 'x', 'y', or 'z'."
            )

        match axis:
            case "i":
                self.matrix = S_I
            case "x":
                self.matrix = S_X
            case "y":
                self.matrix = S_Y
            case "z":
                self.matrix = S_Z

    def action(self, y: TwoLevelState) -> TwoLevelState:
        return y.hilbert_space.from_coeffs((self.matrix @ y.coeffs[..., None])[..., 0])

    def to_matrix(self, hilbert_space: TwoLevel) -> Array:
        return self.matrix


class QubitsState(TensorState):

    @property
    def probabilities(self) -> Array:
        return jnp.abs(self.coeffs) ** 2

    def bit_probability(self, idx: int) -> Array:
        n = self.hilbert_space.num_factors
        p = self.probabilities
        left, right = 2 ** idx, 2 ** (n - idx - 1)
        return p.reshape(*p.shape[:-1], left, 2, right).sum(axis=(-3, -1))[..., 1]

    @property
    def bit_probabilities(self) -> Array:
        n = self.hilbert_space.num_factors
        return jnp.stack([self.bit_probability(i) for i in range(n)], axis=-1)


class Qubits(TensorPower):
    state_type: ClassVar = QubitsState

    def __init__(self, num_bits: int):
        self.factorspace = TwoLevel()
        self.power = num_bits


class PauliProduct(AbstractPauliOperator, KroneckerProduct):
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    def __init__(self, ax_list: list[str]):
        self.ops = tuple(
            Identity() if ax.lower() == "i" else PauliOperator(ax.lower()) for ax in ax_list
        )


class BatchPauliProduct(AbstractPauliOperator, BatchKroneckerProduct):
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    def __init__(self, ax_list: list[str]):
        self.ops = tuple(PauliOperator(ax.lower()) for ax in ax_list)

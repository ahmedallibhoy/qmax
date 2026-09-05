from typing import ClassVar, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ScalarLike

from ..hilbert_space import AbstractState, AbstractHilbertSpace
from ..operator import Operator, Identity, AbstractHermitianOperator
from ..exponentiators import AbstractExponentiator, ExactExponentiator
from ..tensor import TensorState, TensorPower, KroneckerProduct
from .nlevel import NLevel, NLevelState


__all__ = ["TwoLevel", "Qubits"]


PAULI_MATRICES = {
    "i": jnp.eye(2, dtype=complex),
    "x": jnp.array([[0., 1.], [1., 0.]], dtype=complex),
    "y": jnp.array([[0., -1j], [1j, 0.]], dtype=complex),
    "z": jnp.array([[1., 0.], [0., -1.]], dtype=complex)
}


class TwoLevelState(NLevelState):
    """
    """


class TwoLevel(NLevel):
    state_type: ClassVar = TwoLevelState

    def __init__(self):
        self._dim = 2

    def pauli(self, axis) -> PauliOperator:
        return PauliOperator(self, axis)


class AbstractPauliOperator(AbstractHermitianOperator):

    def exp_action(self, h: ScalarLike, y: QubitsState) -> QubitsState:                    
        return jnp.cosh(h) * y + jnp.sinh(h) * self.action(y)

    @property
    def spectral_bounds(self) -> Array:
        return jnp.array([-1.0, 1.0])

    def _solve(self,
        b: QubitsState,
        scale: ScalarLike=-1.0,
        shift: ScalarLike=0.0) -> QubitsState:

        return (shift * b - scale * self.action(b)) / (shift ** 2 - scale ** 2)


class PauliOperator(AbstractPauliOperator):
    axis: str
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    def __init__(
        self,
        domain: TwoLevel,
        axis: str,
        *,
        exponentiator: AbstractExponentiator=ExactExponentiator(),
        name: Optional[str]=None):

        self.domain = domain
        self.axis = axis
        self.exponentiator = exponentiator

        if name is not None:
            self.name = name
        else:
            self.name = f"σ_{axis}" if axis != "i" else "Id"

    def __check_init__(self):
        if self.axis not in ("i", "x", "y", "z"):
            raise ValueError(
                f"Invalid axis={self.axis!r}; axis must be one of 'i', 'x', 'y', or 'z'."
            )

    def action(self, y: TwoLevelState) -> TwoLevelState:
        return self.domain.from_coeffs((self.to_matrix() @ y.coeffs[..., None])[..., 0])

    def to_matrix(self) -> Array:
        return PAULI_MATRICES[self.axis]


class QubitsState(TensorState):
    """
    """


class Qubits(TensorPower):
    state_type: ClassVar = QubitsState

    def __init__(self, num_bits: int=1):
        self.factorspace = TwoLevel()
        self.power = num_bits

    def pauli_product(self, ax_list: list[str]) -> PauliProduct:
        return PauliProduct(self, ax_list)


class PauliProduct(AbstractPauliOperator, KroneckerProduct):
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    def __init__(
        self,
        domain: Qubits,
        ax_list: list[str],
        exponentiator: AbstractExponentiator=ExactExponentiator(),
        name: Optional[str]=None):

        self.domain = domain
        self.children = tuple(
            domain[idx].identity() if ax.lower() == "i" else PauliOperator(domain[idx], ax.lower())
            for idx, ax in enumerate(ax_list)
        )
        self.exponentiator = exponentiator
        self.name = name

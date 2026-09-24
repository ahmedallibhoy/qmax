from typing import Any, ClassVar, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._types import ComplexScalarLike
from ..exponentiators import AbstractExponentiator, ExactExponentiator
from ..hilbert_space import AbstractState
from ..operator import AbstractHermitianOperator
from ..tensor import KroneckerProduct, TensorPower, TensorState
from .nlevel import NLevel, NLevelState

__all__ = ["TwoLevel", "Qubits"]


PAULI_MATRICES = {
    "i": np.eye(2, dtype=complex),
    "x": np.array([[0., 1.], [1., 0.]], dtype=complex),
    "y": np.array([[0., -1j], [1j, 0.]], dtype=complex),
    "z": np.array([[1., 0.], [0., -1.]], dtype=complex)
}


class TwoLevelState(NLevelState["TwoLevel"]):
    """
    """


class TwoLevel(NLevel[TwoLevelState]):
    state_type: ClassVar = TwoLevelState

    def __init__(self):
        self._dim = 2

    def pauli(self, axis) -> PauliOperator:
        return PauliOperator(self, axis)


class AbstractPauliOperator[S: AbstractState[Any]](AbstractHermitianOperator[S]):

    def exp_action(self, h: ComplexScalarLike, y: S) -> S:                    
        return jnp.cosh(h) * y + jnp.sinh(h) * self.action(y)

    @property
    def spectral_bounds(self) -> Array:
        return jnp.array([-1.0, 1.0])

    def _solve(self,
        b: S,
        scale: ComplexScalarLike=-1.0,
        shift: ComplexScalarLike=0.0) -> S:

        return (shift * b - scale * self.action(b)) / (shift ** 2 - scale ** 2)


class PauliOperator(AbstractPauliOperator[TwoLevelState]):
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


class QubitsState(TensorState["Qubits"]):
    """
    """


class Qubits(TensorPower[QubitsState]):
    state_type: ClassVar = QubitsState

    def __init__(self, num_bits: int=1):
        self.factorspace = TwoLevel()
        self.power = num_bits

    def pauli_product(self, ax_list: list[str]) -> PauliProduct:
        return PauliProduct(self, ax_list)


class PauliProduct(AbstractPauliOperator[QubitsState], KroneckerProduct[QubitsState]):
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    def __init__(
        self,
        domain: Qubits,
        ax_list: list[str],
        exponentiator: AbstractExponentiator=ExactExponentiator(),
        name: Optional[str]=None):

        self.domain = domain
        self.children = tuple(
            domain[idx].identity() if ax.lower() == "i" else PauliOperator(domain[idx], ax.lower()) # pyright: ignore[reportArgumentType]
            for idx, ax in enumerate(ax_list)
        )
        self.exponentiator = exponentiator
        self.name = name

from abc import abstractmethod
from typing import Union, Optional

import equinox as eqx
import jax 
import jax.numpy as jnp
import lineax as lx

from jaxtyping import ScalarLike, Array

from ._internal import _update_field, _overrides

from .hilbert_space import AbstractHilbertSpace, AbstractState
from .exponentiators import (
    Order, AbstractExponentiator, ExactExponentiator, NoExponentiator,
    ShiftScaleExponentiator, TruncatedTaylorExponentiator, NotExponentiableError,
    Strang
)
from .utils import over_batch


BRANCH = "├"
PIPE = "|"
ANGLE = "└"
DASH = "─"
PAD = "    "


class IncompatibleDomainError(TypeError):
    pass


class NoExactExponentialError(NotExponentiableError):
    pass 


class NoRealSpectrumError(Exception):
    pass


def _as_shift(x: Union[Operator, ScalarLike]) -> Optional[ScalarLike]:
    """The coefficient c if x is c*I -- as a bare scalar, Identity, or a scalar
    multiple of one -- else None."""
    if jnp.isscalar(x):
        return x
    if isinstance(x, Identity):
        return 1.0
    if isinstance(x, ShiftScaleOperator) and isinstance(x.op, Identity):
        return x.shift + x.scale
    return None


class Operator(eqx.Module):
    domain: AbstractHilbertSpace = eqx.field(static=True)
    exponentiator: AbstractExponentiator = eqx.field(default=NoExponentiator(), kw_only=True)

    def _check_domain(self, y: AbstractState):
        if self.domain != y.hilbert_space:
            raise IncompatibleDomainError(
                f"{type(self).__name__} acts on {self.domain}, "
                f"but received a state on {y.hilbert_space}"
            )

    def _check_compatible(self, other: Union[ScalarLike, Operator]):
        if isinstance(other, Operator) and other.domain != self.domain:
            raise IncompatibleDomainError(
                f"{type(self).__name__} acts on {self.domain}, "
                f"but {type(other).__name__} acts on {other.domain}"
            )

    def with_exponentiator(self, exponentiator: AbstractExponentiator):
        # The usual eqx.tree_at breaks on eqx.Module with fields that have no leaves
        return _update_field(self, "exponentiator", exponentiator)

    @property
    def has_exact_exponential(self) -> bool:
        return _overrides(type(self), "exp_action", Operator)

    # --------------------------------------------------------------------------------------------
    # Exponentiator delegators
    # --------------------------------------------------------------------------------------------

    @property
    def can_exponentiate(self) -> bool:
        return self.exponentiator.can_exponentiate(self)

    def check_exponentiable_tree(self) -> None:
        self.exponentiator.check_exponentiable_tree(self)

    @property
    def tree_order(self) -> Order:
        return self.exponentiator.tree_order(self)

    def adapt(self, dt_max: ScalarLike) -> Operator:
        return self.exponentiator.adapt_tree(self, dt_max)

    # --------------------------------------------------------------------------------------------
    # Public, domain-checked entry points
    # --------------------------------------------------------------------------------------------

    def __call__(self, y: AbstractState) -> AbstractState:
        self._check_domain(y)
        return self.action(y)

    def exp(self, h: ScalarLike, y: AbstractState) -> AbstractState:
        self._check_domain(y)
        return self.exponentiator(self, h, y)

    def solve(self, b: AbstractState, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> AbstractState:
        self._check_domain(b)
        return self._solve(b, scale, shift)

    # --------------------------------------------------------------------------------------------
    # Interfaces
    # --------------------------------------------------------------------------------------------

    @abstractmethod
    def action(self, y: AbstractState) -> AbstractState:
        pass

    @abstractmethod
    def adj_action(self, y: AbstractState) -> AbstractState:
        pass

    def exp_action(self, h: ScalarLike, y: AbstractState) -> AbstractState:
        raise NoExactExponentialError(
            f"Exact exponential cannot be computed: {type(self).__name__} does not override base exp_action"
        )

    def _solve(self, b: AbstractState, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> AbstractState:
        """
        Solves (shift * I + scale * A)y = b. 

        Warning: This is unreliable and slow. Operators using implicit exponentiators should 
        override this method when a more efficient implementation exists. 
        """
        func = lambda y: shift * y + scale * self(y)

        def fn(b_i):
            shape = jax.eval_shape(lambda: b_i)
            lx_op = lx.FunctionLinearOperator(func, input_structure=shape)
            sol = lx.linear_solve(lx_op, b_i, solver=lx.GMRES(rtol=1e-9, atol=1e-9))
            return sol.value

        return over_batch(fn, b)

    @property
    def spectral_bounds(self) -> Array:
        """
        Returns interval (lambda_min, lambda_max) containing all eigenvalues of operator
        """
        raise NotImplementedError

    def expected_value(self, y: AbstractState) -> ScalarLike:
        return y.expected_value(self)

    def to_matrix(self) -> Array:
        raise NotImplementedError

    # --------------------------------------------------------------------------------------------
    # Operator Algebra
    # --------------------------------------------------------------------------------------------

    def __add__(self, other: Union[Operator, ScalarLike]) -> Operator:
        self._check_compatible(other)

        c = _as_shift(other)
        if c is not None:
            return ShiftScaleOperator(self, shift=c)
        if not isinstance(other, Operator):
            return NotImplemented

        # self may be the identity term instead
        c = _as_shift(self)
        if c is not None:
            return ShiftScaleOperator(other, shift=c)  
            
        return AddOperator(self, other)

    def __radd__(self, other: Union[Operator, ScalarLike]) -> Operator:
        self._check_compatible(other)

        c = _as_shift(other)
        if c is not None:
            return ShiftScaleOperator(self, shift=c)
        if not isinstance(other, Operator):
            return NotImplemented

        # self may be the identity term instead
        c = _as_shift(self)
        if c is not None:
            return ShiftScaleOperator(other, shift=c)

        return AddOperator(other, self)

    def __sub__(self, other: Union[Operator, ScalarLike]) -> Operator:
        self._check_compatible(other)

        c = _as_shift(other)
        if c is not None:
            return ShiftScaleOperator(self, shift=-c)
        if isinstance(other, Operator):
            return self + (-other) 

        return NotImplemented

    def __rsub__(self, other: Union[Operator, ScalarLike]) -> Operator:
        self._check_compatible(other)

        c = _as_shift(other)
        if c is not None:
            return ShiftScaleOperator(self, shift=c, scale=-1.0)
        if isinstance(other, Operator):
            return other + (-self) 

        return NotImplemented

    def __mul__(self, other: ScalarLike) -> Operator:
        if not jnp.isscalar(other):
            return NotImplemented

        return ShiftScaleOperator(self, scale=other)

    def __rmul__(self, other: ScalarLike) -> Operator:
        if not jnp.isscalar(other):
            return NotImplemented

        return ShiftScaleOperator(self, scale=other) 

    def __matmul__(self, other: Operator) -> Operator:
        self._check_compatible(other)

        if not isinstance(other, Operator):
            return NotImplemented

        c = _as_shift(other)
        if c is not None:
            return c * self 

        c = _as_shift(self)
        if c is not None:
            return c * other

        return MatMulOperator(self, other)

    def __rmatmul__(self, other: Operator) -> Operator:
        self._check_compatible(other)

        if not isinstance(other, Operator):
            return NotImplemented

        # A @ (c * I) == c * A
        c = _as_shift(other)
        if c is not None:
            return c * self 

        # (c * I) @ B == c * B
        c = _as_shift(self)
        if c is not None:
            return c * other

        return MatMulOperator(other, self)

    def adjoint(self) -> Operator:
        return AdjOperator(self)

    @property
    def H(self) -> Operator:
        return self.adjoint()

    def __truediv__(self, other: ScalarLike) -> Operator:
        if not jnp.isscalar(other):
            return NotImplemented

        return ShiftScaleOperator(self, scale=1.0 / other) 

    def __neg__(self) -> Operator:
        return ShiftScaleOperator(self, scale=-1.0)

    def to_str(self) -> str:
        return f"{type(self).__name__}"

    def draw_tree(self, prefix="") -> str:
        return f"{self.to_str()}"

    def __str__(self) -> str:
        return self.draw_tree()


class AbstractHermitianOperator(Operator):

    def adj_action(self, y: AbstractState) -> AbstractState:
        return self.action(y)

    def adjoint(self) -> AbstractHermitianOperator:
        return self


class ShiftScaleOperator(Operator):
    """
    Implements shift * Identity() + scale * op
    """
    op: Operator
    shift: ScalarLike = 0.0
    scale: ScalarLike = 1.0

    def __init__(
        self, 
        op: Operator, 
        shift: ScalarLike=0.0,
        scale: ScalarLike=1.0, 
        exponentiator: Optional[AbstractExponentiator]=None):

        self.domain = op.domain

        if isinstance(op, ShiftScaleOperator):
            # If op = s0 * I + c0 * A then 
            # s1 * I + c1 * op can be collapsed to (s1 + c1 * s0) * I + c0 * c1 * A 
            self.op = op.op
            self.shift = shift + scale * op.shift
            self.scale = scale * op.scale
        else:
            self.op = op 
            self.shift = shift 
            self.scale = scale 

        if exponentiator is not None:
            self.exponentiator = exponentiator
        else:
            self.exponentiator = ShiftScaleExponentiator()

    def action(self, y: AbstractState) -> AbstractState:
        return self.scale * self.op.action(y) + self.shift * y

    def adj_action(self, y: AbstractState) -> AbstractState:
        return jnp.conj(self.scale) * self.op.adj_action(y) + jnp.conj(self.shift) * y

    def _solve(self, b, scale=-1.0, shift=0.0):
        return self.op._solve(b, scale * self.scale, shift + scale * self.shift)

    @property
    def spectral_bounds(self):
        if jnp.iscomplexobj(self.shift) or jnp.iscomplexobj(self.scale):
            raise NoRealSpectrumError(
                f"scale * {type(self.op).__name__} + shift * Identity() does not have a real spectrum "
                f"since shift={self.shift} or scale={self.scale} is complex"
            )

        return jnp.sort(self.scale * self.op.spectral_bounds + self.shift)

    def to_matrix(self):
        return self.scale * self.op.to_matrix() + self.shift * jnp.eye(self.domain.dim)

    def adjoint(self) -> Operator:
        return jnp.conj(self.shift) + jnp.conj(self.scale) * self.op.adjoint()

    def to_str(self) -> str:
        return f"{type(self).__name__}(shift={self.shift}, scale={self.scale})"

    def draw_tree(self, prefix="") -> str:
        op_tree = self.op.draw_tree(prefix + PAD + " ")
        return f"{self.to_str()}\n{prefix}{PAD + ANGLE + DASH}{op_tree}"


class AddOperator(Operator):
    op1: Operator
    op2: Operator

    def __init__(
        self, 
        op1: Operator, 
        op2: Operator, 
        exponentiator: Optional[AbstractExponentiator]=None):

        if op1.domain != op2.domain:
            raise IncompatibleDomainError(
                f"Cannot add operators on different domains: op1={type(self.op1).__name__} acts on {op1.domain}, "
                f"but op2={type(self.op2).__name__} acts on {op2.domain},"
            )

        self.domain = op1.domain
        self.op1 = op1 
        self.op2 = op2

        if exponentiator is not None:
            self.exponentiator = exponentiator
        elif not op1.can_exponentiate or not op2.can_exponentiate:
            self.exponentiator = TruncatedTaylorExponentiator()
        else:
            self.exponentiator = Strang()

    def action(self, y: AbstractState) -> AbstractState:
        return self.op1.action(y) + self.op2.action(y)

    def adj_action(self, y: AbstractState) -> AbstractState:
        return self.op1.adj_action(y) + self.op2.adj_action(y)

    @property
    def spectral_bounds(self) -> Array:
        return self.op1.spectral_bounds + self.op2.spectral_bounds

    def to_matrix(self) -> Array:
        return self.op1.to_matrix() + self.op2.to_matrix()

    def adjoint(self) -> Operator:
        return self.op1.adjoint() + self.op2.adjoint()

    def draw_tree(self, prefix="") -> str:
        op1_tree = self.op1.draw_tree(prefix + PAD + PIPE)
        op2_tree = self.op2.draw_tree(prefix + PAD + " ")
        return f"{self.to_str()}\n{prefix}{PAD + BRANCH + DASH}{op1_tree}\n{prefix}{PAD + ANGLE + DASH}{op2_tree}"


class MatMulOperator(Operator):
    op1: Operator
    op2: Operator

    def __init__(self, op1, op2, exponentiator=TruncatedTaylorExponentiator()):
        if op1.domain != op2.domain:
            raise IncompatibleDomainError(
                f"Cannot compose operators on different domains: op1={type(self.op1).__name__} acts on {op1.domain}, "
                f"but op2={type(self.op2).__name__} acts on {op2.domain},"
            )
        
        self.domain = op1.domain
        self.op1 = op1 
        self.op2 = op2 
        self.exponentiator = exponentiator

    def action(self, y: AbstractState) -> AbstractState:
        return self.op1.action(self.op2.action(y))

    def adj_action(self, y: AbstractState) -> AbstractState:
        return self.op2.adj_action(self.op1.adj_action(y))

    def to_matrix(self) -> Array:
        return self.op1.to_matrix() @ self.op2.to_matrix()

    def adjoint(self) -> Operator:
        return self.op2.adjoint() @ self.op1.adjoint()

    def draw_tree(self, prefix="") -> str:
        op1_tree = self.op1.draw_tree(prefix + PAD + PIPE)
        op2_tree = self.op2.draw_tree(prefix + PAD + " ")
        return f"{self.to_str()}\n{prefix}{PAD + BRANCH + DASH}{op1_tree}\n{prefix}{PAD + ANGLE + DASH}{op2_tree}"


class AdjOperator(Operator):
    op: Operator

    def __init__(
        self, 
        op: Operator, 
        exponentiator: Optional[AbstractExponentiator]=None):

        self.domain = op.domain
        self.op = op

        if exponentiator is not None:
            self.exponentiator = exponentiator
        else:
            self.exponentiator = NoExponentiator()

    def action(self, y: AbstractState) -> AbstractState:
        return self.op.adj_action(y)

    def adj_action(self, y: AbstractState) -> AbstractState:
        return self.op.action(y)

    @property
    def spectral_bounds(self) -> Array:
        return self.op.spectral_bounds

    def to_matrix(self) -> Array:
        return jnp.conj(self.op.to_matrix().T)

    def adjoint(self) -> Operator:
        return self.op

    def draw_tree(self, prefix="") -> str:
        op_tree = self.op.draw_tree(prefix + PAD + " ")
        return f"{self.to_str()}\n{prefix}{PAD + ANGLE + DASH}{op_tree}"


class Identity(AbstractHermitianOperator):
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    def action(self, y: AbstractState) -> AbstractState:
        return y

    def exp_action(self, h: ScalarLike, y: AbstractState) -> AbstractState:
        return jnp.exp(h) * y

    def _solve(self, b: AbstractState, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> AbstractState:
        return b / (shift + scale)

    @property
    def spectral_bounds(self) -> Array:
        return jnp.array([1.0, 1.0])

    def to_matrix(self) -> Array:
        return jnp.eye(self.domain.dim)


class AbstractDiagonalOperator(AbstractHermitianOperator):
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    @property
    @abstractmethod
    def eigvals(self) -> Array:
        # Assumed that eigvals are real
        pass

    def action(self, y: AbstractState) -> AbstractState:
        return self.domain.from_coeffs(self.eigvals * y.coeffs)

    def exp_action(self, h: ScalarLike, y: AbstractState) -> AbstractState:
        coeffs = jnp.exp(h * self.eigvals) * y.coeffs
        return self.domain.from_coeffs(coeffs)

    def _solve(
        self, 
        b: AbstractState, 
        scale: ScalarLike=-1.0, 
        shift: ScalarLike=0.0) -> AbstractState:

        return self.domain.from_coeffs(b.coeffs / (scale * self.eigvals + shift))

    @property
    def spectral_bounds(self) -> Array:
        return jnp.array([jnp.min(self.eigvals), jnp.max(self.eigvals)])

    def to_matrix(self) -> Array:
        return jnp.diag(self.eigvals)

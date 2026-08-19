from abc import abstractmethod
from typing import Callable, TypeVar, ClassVar, Union, Any, Optional

import dataclasses

import equinox as eqx
import jax 
import jax.numpy as jnp
import lineax as lx

from jaxtyping import ScalarLike, Array, ArrayLike, PRNGKeyArray

from ._internal import _update_field

from .hilbert_space import AbstractHilbertSpace, AbstractState
from .exponentiators import (
    Order, AbstractExponentiator, ExactExponentiator, NoExponentiator,
    ShiftScaleExponentiator, TruncatedTaylorExponentiator, NotExponentiableError,
    Strang
)
from .utils import over_batch


class IncompatibleDomainError(TypeError):
    pass


class NoExactExponentialError(NotExponentiableError):
    pass 


class NoRealSpectrumError(Exception):
    pass


def _reconcile_domains(A, B):                   
    if A is AbstractHilbertSpace: 
        return B             
    if B is AbstractHilbertSpace: 
        return A

    if isinstance(A, tuple) and isinstance(B, tuple):
        if len(A) != len(B): 
            raise IncompatibleDomainError
        return tuple(_reconcile_domains(x, y) for x, y in zip(A, B))

    if isinstance(A, tuple) or isinstance(B, tuple):
        raise IncompatibleDomainError

    if issubclass(A, B): 
        return A       
    if issubclass(B, A): 
        return B

    raise IncompatibleDomainError


def _get_name(struct: Union[tuple, type[Any]]) -> str:
    if isinstance(struct, tuple):
        return "(" + ", ".join(_get_name(s) for s in struct) + ")"
    return struct.__name__


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
    domain: ClassVar = AbstractHilbertSpace
    exponentiator: AbstractExponentiator = eqx.field(default=NoExponentiator(), kw_only=True)

    def _check_domain(self, y: AbstractState):
        try:
            _reconcile_domains(self.domain, y.hilbert_space.structure)
        except IncompatibleDomainError:
            raise IncompatibleDomainError(
                f"{type(self).__name__} acts on {_get_name(self.domain)}, "
                f"but received a state on {_get_name(y.hilbert_space.structure)}"
            )

    def with_exponentiator(self, exponentiator: AbstractExponentiator):
        # The usual eqx.tree_at breaks on eqx.Module with fields that have no leaves
        return _update_field(self, "exponentiator", exponentiator)

    # --------------------------------------------------------------------------------------------
    # Exponentiator delegators
    #   Convenience wrappers for querying properties of which are operator dependent. 
    #   These should be thin wrappers which pass self to the corresponding method of AbstractExponentiator
    # --------------------------------------------------------------------------------------------
    
    @property
    def can_exponentiate(self) -> bool:
        return self.exponentiator.can_exponentiate(self)

    def check_exponentiable_tree(self) -> None:
        self.exponentiator.check_exponentiable_tree(self)

    @property
    def tree_order(self) -> Order:
        return self.exponentiator.tree_order(self)

    def adapt(self, hilbert_space: AbstractHilbertSpace, dt_max: ScalarLike) -> Operator:
        return self.exponentiator.adapt_tree(self, hilbert_space, dt_max)

    # --------------------------------------------------------------------------------------------
    # Public, domain-checked entry points
    # --------------------------------------------------------------------------------------------

    def __call__(self, y: AbstractState) -> AbstractState:
        self._check_domain(y)
        return self.action(y)

    def exp(self, h: ScalarLike, y: AbstractState) -> AbstractState:
        self._check_domain(y)
        return self.exponentiator(self, h, y)

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
        raise NoExactExponentialError(f"Exact exponential cannot be computed: {type(self).__name__} does not override base exp_action")

    def solve(self, b: AbstractState, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> AbstractState:
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

    def spectral_bounds(self, hilbert_space: AbstractHilbertSpace) -> Array:
        """
        Returns interval (lambda_min, lambda_max) containing all eigenvalues of operator
        """
        raise NotImplementedError

    def expected_value(self, y: AbstractState) -> ScalarLike:
        return y.expected_value(self)

    def to_matrix(self, hilbert_space: AbstractHilbertSpace) -> Array:
        raise NotImplementedError

    # --------------------------------------------------------------------------------------------
    # Operator Algebra
    # --------------------------------------------------------------------------------------------

    def __add__(self, other: Union[Operator, ScalarLike]) -> Operator:
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
        c = _as_shift(other)
        if c is not None:
            return ShiftScaleOperator(self, shift=-c)
        if isinstance(other, Operator):
            return self + (-other) 
        return NotImplemented

    def __rsub__(self, other: Union[Operator, ScalarLike]) -> Operator:
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

    @property
    def domain(self): 
        return self.op.domain

    def action(self, y: AbstractState) -> AbstractState:
        return self.scale * self.op.action(y) + self.shift * y

    def adj_action(self, y: AbstractState) -> AbstractState:
        return jnp.conj(self.scale) * self.op.adj_action(y) + jnp.conj(self.shift) * y

    def solve(self, b, scale=-1.0, shift=0.0):
        return self.op.solve(b, scale * self.scale, shift + scale * self.shift)

    def spectral_bounds(self, hilbert_space):
        if jnp.iscomplexobj(self.shift) or jnp.iscomplexobj(self.scale):
            raise NoRealSpectrumError(
                f"scale * {type(self.op).__name__} + shift * Identity() does not have a real spectrum since shift={self.shift} or scale={self.scale} is complex"
            )

        return jnp.sort(self.scale * self.op.spectral_bounds(hilbert_space) + self.shift)

    def to_matrix(self, hilbert_space):
        return self.scale * self.op.to_matrix(hilbert_space) + self.shift * jnp.eye(hilbert_space.dim)

    def adjoint(self) -> Operator:
        return jnp.conj(self.shift) + jnp.conj(self.scale) * self.op.adjoint()

class AddOperator(Operator):
    op1: Operator
    op2: Operator

    def __init__(
        self, 
        op1: Operator, 
        op2: Operator, 
        exponentiator: Optional[AbstractExponentiator]=None):

        self.op1 = op1 
        self.op2 = op2

        if exponentiator is not None:
            self.exponentiator = exponentiator
        elif not op1.can_exponentiate or not op2.can_exponentiate:
            self.exponentiator = TruncatedTaylorExponentiator()
        else:
            self.exponentiator = Strang()

    def __check_init__(self):
        try:
            _reconcile_domains(self.op1.domain, self.op2.domain)
        except IncompatibleDomainError:
            raise IncompatibleDomainError(
                f"Incompatible domains: {type(self.op1).__name__} acts on {_get_name(self.op1.domain)}, "
                f"but {type(self.op2).__name__} acts on {_get_name(self.op2.domain)},"
            )

    @property
    def domain(self):
        return _reconcile_domains(self.op1.domain, self.op2.domain)

    def action(self, y: AbstractState) -> AbstractState:
        return self.op1.action(y) + self.op2.action(y)

    def adj_action(self, y: AbstractState) -> AbstractState:
        return self.op1.adj_action(y) + self.op2.adj_action(y)

    def spectral_bounds(self, hilbert_space: AbstractHilbertSpace) -> Array:
        return self.op1.spectral_bounds(hilbert_space) + self.op2.spectral_bounds(hilbert_space)

    def to_matrix(self, hilbert_space: AbstractHilbertSpace) -> Array:
        return self.op1.to_matrix(hilbert_space) + self.op2.to_matrix(hilbert_space)

    def adjoint(self) -> Operator:
        return self.op1.adjoint() + self.op2.adjoint()


class MatMulOperator(Operator):
    op1: Operator
    op2: Operator
    exponentiator: AbstractExponentiator = eqx.field(default=TruncatedTaylorExponentiator(), kw_only=True)

    @property
    def domain(self):
        return _reconcile_domains(self.op1.domain, self.op2.domain)

    def action(self, y: AbstractState) -> AbstractState:
        return self.op1.action(self.op2.action(y))

    def adj_action(self, y: AbstractState) -> AbstractState:
        return self.op2.adj_action(self.op1.adj_action(y))

    def to_matrix(self, hilbert_space: AbstractHilbertSpace) -> Array:
        return self.op1.to_matrix(hilbert_space) @ self.op2.to_matrix(hilbert_space)

    def adjoint(self) -> Operator:
        return self.op2.adjoint() @ self.op1.adjoint()


class AdjOperator(Operator):
    op: Operator

    def __init__(self, op: Operator, exponentiator: Optional[AbstractExponentiator]=None):
        self.op = op

        if exponentiator is not None:
            self.exponentiator = exponentiator
        else:
            self.exponentiator = NoExponentiator()

    @property
    def domain(self):
        return self.op.domain

    def action(self, y: AbstractState) -> AbstractState:
        return self.op.adj_action(y)

    def adj_action(self, y: AbstractState) -> AbstractState:
        return self.op.action(y)

    def spectral_bounds(self, hilbert_space: AbstractHilbertSpace) -> Array:
        return self.op.spectral_bounds(hilbert_space)

    def to_matrix(self, hilbert_space: AbstractHilbertSpace) -> Array:
        return jnp.conj(self.op.to_matrix(hilbert_space).T)

    def adjoint(self) -> Operator:
        return self.op


class Identity(AbstractHermitianOperator):
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    def action(self, y: AbstractState) -> AbstractState:
        return y

    def exp_action(self, h: ScalarLike, y: AbstractState) -> AbstractState:
        return jnp.exp(h) * y

    def solve(self, b: AbstractState, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> AbstractState:
        return b / (shift + scale)

    def spectral_bounds(self, hilbert_space: AbstractHilbertSpace) -> Array:
        return jnp.array([1.0, 1.0])

    def to_matrix(self, hilbert_space: AbstractHilbertSpace) -> Array:
        return jnp.eye(hilbert_space.dim)


class Zero(AbstractHermitianOperator):
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    def action(self, y: AbstractState) -> AbstractState:
        return y.hilbert_space.zeros_like(y)

    def exp_action(self, h: ScalarLike, y: AbstractState) -> AbstractState:
        return y

    def solve(self, b: AbstractState, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> AbstractState:
        return b / shift

    def spectral_bounds(self, hilbert_space: AbstractHilbertSpace) -> Array:
        return jnp.array([0.0, 0.0])

    def to_matrix(self, hilbert_space: AbstractHilbertSpace) -> Array:
        return jnp.zeros((hilbert_space.dim, hilbert_space.dim))


class AbstractDiagonalOperator(AbstractHermitianOperator):
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    @abstractmethod
    def eigvals(self, hilbert_space: AbstractHilbertSpace) -> Array:
        # Assumed that eigvals are real
        pass

    def action(self, y: AbstractState) -> AbstractState:
        return y.hilbert_space.from_coeffs(self.eigvals(y.hilbert_space) * y.coeffs)

    def exp_action(self, h: ScalarLike, y: AbstractState) -> AbstractState:
        coeffs = jnp.exp(h * self.eigvals(y.hilbert_space)) * y.coeffs
        return y.hilbert_space.from_coeffs(coeffs)

    def solve(
        self, 
        b: AbstractState, 
        scale: ScalarLike=-1.0, 
        shift: ScalarLike=0.0) -> AbstractState:

        hilbert_space = b.hilbert_space
        eigvals = self.eigvals(hilbert_space)
        return hilbert_space.from_coeffs(b.coeffs / (scale * eigvals + shift))

    def spectral_bounds(self, hilbert_space: AbstractHilbertSpace) -> Array:
        eigvals = self.eigvals(hilbert_space)
        return jnp.array([jnp.min(eigvals), jnp.max(eigvals)])

    def to_matrix(self, hilbert_space: AbstractHilbertSpace) -> Array:
        return jnp.diag(self.eigvals(hilbert_space))

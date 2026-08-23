from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Union, Optional, TypeVar

import equinox as eqx
import jax 
import jax.numpy as jnp
import lineax as lx

from jaxtyping import ScalarLike, Array

from ._internal import _update_field, _overrides
from ._introspect import (
    Count, CountDict, CountType, InterfaceCount, Path, RenderTree, _rows
)
from .hilbert_space import AbstractHilbertSpace, AbstractState
from .exponentiators import (
    Order, 
    AbstractExponentiator, 
    ExactExponentiator, 
    NoExponentiator,
    ShiftScaleExponentiator, 
    NotExponentiableError,
    Strang
)
from .utils import over_batch


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
    if isinstance(x, ShiftScaleOperator) and isinstance(x.children[0], Identity):
        return x.shift + x.scale
    return None


T = TypeVar("T")


class Operator(eqx.Module):
    domain: AbstractHilbertSpace = eqx.field(static=True)
    children: tuple[Operator, ...] = eqx.field(default=(), kw_only=True)
    exponentiator: AbstractExponentiator = eqx.field(default=NoExponentiator(), kw_only=True)
    _label: Optional[str] = eqx.field(default=None, kw_only=True)

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

    def with_label(
        self, 
        label: str,
        path: Path=Path(), 
        parent_path=None, 
        child_idx=None) -> Operator:

        if path:
            index, new_path = path.descend()
            fn = lambda o: o.children[index]

            child = fn(self)
            new_child = child.with_label(
                label, new_path, self.path(parent_path, child_idx), index)
            new_self = eqx.tree_at(fn, self, new_child)
            return new_self

        return _update_field(self, "_label", label)

    def with_exponentiator(
        self, 
        exponentiator: AbstractExponentiator, 
        path: Path=Path(), 
        parent_path=None, 
        child_idx=None) -> Operator:

        try:
            if path:
                index, new_path = path.descend()
                fn = lambda o: o.children[index]

                child = fn(self)
                new_child = child.with_exponentiator(
                    exponentiator, new_path, self.path(parent_path, child_idx), index)
                new_self = eqx.tree_at(fn, self, new_child)
                return new_self

            if not isinstance(exponentiator, NoExponentiator): # NoExponentiator will raise but is assignable
                exponentiator.check_exponentiable_tree(self, parent_path, child_idx)    

            # The usual eqx.tree_at breaks on eqx.Module with fields that have no leaves
            return _update_field(self, "exponentiator", exponentiator)
        except NotExponentiableError as e:
            raise e.from_path(path) from None

    @property
    def has_exact_exponential(self) -> bool:
        return _overrides(type(self), "exp_action", Operator)

    # --------------------------------------------------------------------------------------------
    # Exponentiator delegators
    # --------------------------------------------------------------------------------------------

    def can_exponentiate(self, parent_path: Optional[Path]=None, child_idx: Optional[int]=None) -> bool:
        return self.exponentiator.can_exponentiate(self, parent_path, child_idx)

    def check_exponentiable_tree(self, parent_path: Optional[Path]=None, child_idx: Optional[int]=None):
        self.exponentiator.check_exponentiable_tree(self, parent_path, child_idx)

    @property
    def tree_order(self) -> Order:
        return self.exponentiator.tree_order(self)

    def exp_count(self, h: ScalarLike, parent_path: Optional[Path]=None, child_idx: Optional[int]=None) -> CountDict:
        return self.exponentiator.tree_count(self, h, parent_path, child_idx)

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

    # --------------------------------------------------------------------------------------------
    # Introspection
    # --------------------------------------------------------------------------------------------

    #------------------- Do not override -------------------

    def __repr__(self) -> str:
        return "\n".join(line for line, _ in _rows(self))

    def path(self, parent_path: Optional[Path]=None, child_idx: Optional[int]=None) -> Path:
        # parent_path is None at the entry point of a traversal, where self is the root
        if parent_path is None:
            return Path(self.label)
        return parent_path.append(child_idx, self.label)

    def _exp_action_count(self, path: Path) -> CountType:
        return {path: Count(exp_actions=1)} if self.has_exact_exponential else NotImplemented

    def leaves(self, parent_path: Optional[Path]=None, child_idx: Optional[int]=None) -> list[Path]:
        path = self.path(parent_path, child_idx)

        if not self.children:
            return [path]

        leaves = [op.leaves(path, idx) for idx, op in enumerate(self.children)]
        return [leaf for leaf_list in leaves for leaf in leaf_list]

    def child_at(self, path: Path) -> Operator:
        if not path:
            return self
        index, rest = path.descend()
        return self.children[index].child_at(rest)

    @property
    def label(self) -> str:
        return self.default_label if self._label is None else self._label

    #------------- Should override if necessary -------------
    @property
    def default_label(self) -> str:
        return type(self).__name__

    def interface_count(self, parent_path: Optional[Path]=None, child_idx: Optional[int]=None) -> InterfaceCount:
        # for each interface (i.e. action, adj_action, solve, exp_action), 
        # recursively counts number of calls to each interface of a leaves of expression tree
        path = self.path(parent_path, child_idx)

        return InterfaceCount(
            action={path: Count(actions=1)},
            adj_action={path: Count(adj_actions=1)},
            solve={path: Count(solves=1)},
            exp_action=self._exp_action_count(path),
        )


class AbstractHermitianOperator(Operator):

    def adj_action(self, y: AbstractState) -> AbstractState:
        return self.action(y)

    def adjoint(self) -> AbstractHermitianOperator:
        return self


class ShiftScaleOperator(Operator):
    """
    Implements shift * Identity() + scale * op
    """
    shift: ScalarLike = 0.0
    scale: ScalarLike = 1.0

    def __init__(
        self,
        op: Operator,
        shift: ScalarLike=0.0,
        scale: ScalarLike=1.0,
        exponentiator: Optional[AbstractExponentiator]=ShiftScaleExponentiator()):

        self.domain = op.domain

        if isinstance(op, ShiftScaleOperator):
            # If op = s0 * I + c0 * A then
            # s1 * I + c1 * op can be collapsed to (s1 + c1 * s0) * I + c0 * c1 * A
            self.children = op.children
            self.shift = shift + scale * op.shift
            self.scale = scale * op.scale
        else:
            self.children = (op,)
            self.shift = shift
            self.scale = scale

        self.exponentiator = exponentiator

    def action(self, y: AbstractState) -> AbstractState:
        (A,) = self.children
        return self.scale * A.action(y) + self.shift * y

    def adj_action(self, y: AbstractState) -> AbstractState:
        (A,) = self.children
        return jnp.conj(self.scale) * A.adj_action(y) + jnp.conj(self.shift) * y

    def _solve(self, b, scale=-1.0, shift=0.0):
        (A,) = self.children
        return A._solve(b, scale * self.scale, shift + scale * self.shift)

    @property
    def spectral_bounds(self):
        (A,) = self.children
        if jnp.iscomplexobj(self.shift) or jnp.iscomplexobj(self.scale):
            raise NoRealSpectrumError(
                f"scale * {type(A).__name__} + shift * Identity() does not have a real spectrum "
                f"since shift={self.shift} or scale={self.scale} is complex"
            )

        return jnp.sort(self.scale * A.spectral_bounds + self.shift)

    def to_matrix(self):
        (A,) = self.children
        return self.scale * A.to_matrix() + self.shift * jnp.eye(self.domain.dim)

    def adjoint(self) -> Operator:
        (A,) = self.children
        return jnp.conj(self.shift) + jnp.conj(self.scale) * A.adjoint()

    @property
    def default_label(self) -> str:
        (A,) = self.children
        if self.shift == 0 and self.scale == 0:
             return "0"
        elif self.shift == 0:
            return f"{self.scale} * {A.label}"
        elif self.scale == 0:
            return f"{A.label} + {self.shift} * Identity()"
        else:
            return f"{self.scale} * {A.label} + {self.shift} * Identity()"

    def interface_count(self, parent_path: Optional[Path]=None, child_idx: Optional[int]=None) -> InterfaceCount:
        (A,) = self.children
        path = self.path(parent_path, child_idx)
        c = A.interface_count(path, 0)

        return InterfaceCount(
            action     = c.action,
            adj_action = c.adj_action,
            solve      = c.solve,
            exp_action = self._exp_action_count(path),
        )


class AddOperator(Operator):

    def __init__(
        self,
        A: Operator,
        B: Operator,
        exponentiator: Optional[AbstractExponentiator]=None, 
        label: Optional[str]=None):

        if A.domain != B.domain:
            raise IncompatibleDomainError(
                f"Cannot add operators on different domains: A={type(A).__name__} acts on {A.domain}, "
                f"but B={type(B).__name__} acts on {B.domain},"
            )

        self.domain = A.domain
        self.children = (A, B)

        if exponentiator is not None:
            self.exponentiator = exponentiator
        elif not A.can_exponentiate() or not B.can_exponentiate():
            self.exponentiator = NoExponentiator()
        else:
            self.exponentiator = Strang()

        self._label = label

    def action(self, y: AbstractState) -> AbstractState:
        A, B = self.children
        return A.action(y) + B.action(y)

    def adj_action(self, y: AbstractState) -> AbstractState:
        A, B = self.children
        return A.adj_action(y) + B.adj_action(y)

    @property
    def spectral_bounds(self) -> Array:
        A, B = self.children
        return A.spectral_bounds + B.spectral_bounds

    def to_matrix(self) -> Array:
        A, B = self.children
        return A.to_matrix() + B.to_matrix()

    def adjoint(self) -> Operator:
        A, B = self.children
        return A.adjoint() + B.adjoint()

    @property
    def default_label(self) -> str:
        A, B = self.children
        return f"({A.label} + {B.label})"

    def interface_count(self, parent_path: Optional[Path]=None, child_idx: Optional[int]=None) -> InterfaceCount:
        path = self.path(parent_path, child_idx)
        c1, c2 = (op.interface_count(path, idx) for idx, op in enumerate(self.children))

        return InterfaceCount(
            action     = c1.action | c2.action,
            adj_action = c1.adj_action | c2.adj_action,
            solve      = {path: Count(solves=1)},
            exp_action = self._exp_action_count(path),
        )


class MatMulOperator(Operator):

    def __init__(
        self, 
        A: Operator, 
        B: Operator, 
        exponentiator=NoExponentiator(),
        label: Optional[str]=None):

        if A.domain != B.domain:
            raise IncompatibleDomainError(
                f"Cannot compose operators on different domains: A={type(A).__name__} acts on {A.domain}, "
                f"but B={type(B).__name__} acts on {B.domain},"
            )

        self.domain = A.domain
        self.children = (A, B)
        self.exponentiator = exponentiator
        self._label = label

    def action(self, y: AbstractState) -> AbstractState:
        A, B = self.children
        return A.action(B.action(y))

    def adj_action(self, y: AbstractState) -> AbstractState:
        A, B = self.children
        return B.adj_action(A.adj_action(y))

    def to_matrix(self) -> Array:
        A, B = self.children
        return A.to_matrix() @ B.to_matrix()

    def adjoint(self) -> Operator:
        A, B = self.children
        return B.adjoint() @ A.adjoint()

    @property
    def default_label(self) -> str:
        A, B = self.children
        return f"({A.label} @ {B.label})"

    def interface_count(self, parent_path: Optional[Path]=None, child_idx: Optional[int]=None) -> InterfaceCount:
        path = self.path(parent_path, child_idx)
        c1, c2 = (op.interface_count(path, idx) for idx, op in enumerate(self.children))

        return InterfaceCount(
            action     = c1.action | c2.action,
            adj_action = c1.adj_action | c2.adj_action,
            solve      = {path: Count(solves=1)},
            exp_action = self._exp_action_count(path),
        )


class AdjOperator(Operator):

    def __init__(
        self,
        op: Operator,
        exponentiator: Optional[AbstractExponentiator]=NoExponentiator(), 
        label: Optional[str]=None):

        self.domain = op.domain
        self.children = (op,)
        self.exponentiator = exponentiator
        self._label = label

    def action(self, y: AbstractState) -> AbstractState:
        (A,) = self.children
        return A.adj_action(y)

    def adj_action(self, y: AbstractState) -> AbstractState:
        (A,) = self.children
        return A.action(y)

    @property
    def spectral_bounds(self) -> Array:
        (A,) = self.children
        return A.spectral_bounds

    def to_matrix(self) -> Array:
        (A,) = self.children
        return jnp.conj(A.to_matrix().T)

    def adjoint(self) -> Operator:
        (A,) = self.children
        return A

    @property
    def default_label(self) -> str:
        (A,) = self.children
        return f"({A.label})^H"


    def interface_count(self, parent_path: Optional[Path]=None, child_idx: Optional[int]=None) -> InterfaceCount:
        (A,) = self.children
        path = self.path(parent_path, child_idx)
        c = A.interface_count(path, 0)

        return InterfaceCount(
            action     = c.adj_action,
            adj_action = c.action,
            solve      = {path: Count(solves=1)},
            exp_action = self._exp_action_count(path),
        )


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


class AbstractDiagonalOperator(Operator):
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    @property
    @abstractmethod
    def eigvals(self) -> Array:
        pass

    def action(self, y: AbstractState) -> AbstractState:
        return self.domain.from_coeffs(self.eigvals * y.coeffs)

    def adj_action(self, y: AbstractState) -> AbstractState:
        return self.domain.from_coeffs(jnp.conj(self.eigvals) * y.coeffs)

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
        if not jnp.iscomplexobj(self.eigvals):
            return jnp.array([jnp.min(self.eigvals), jnp.max(self.eigvals)])

        raise NotImplementedError

    def to_matrix(self) -> Array:
        return jnp.diag(self.eigvals)

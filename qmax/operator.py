from abc import abstractmethod
from typing import Callable, TypeVar, ClassVar, Union, Any, Optional

import dataclasses

import equinox as eqx
import jax 
import jax.numpy as jnp
import lineax as lx

from jaxtyping import ScalarLike, Array, ArrayLike

from ._internal import _update_field

from .hilbert_space import AbstractHilbertSpace, AbstractState
from .exponentiators import AbstractExponentiator, ExactExponentiator, ScaleSquareExponentiator
from .split import AbstractSplitMethod, Strang
from .utils import over_batch


class IncompatibleDomainError(TypeError):
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
    if isinstance(x, ScalarMulOperator) and isinstance(x.op, Identity):
        return x.c
    return None


class Operator(eqx.Module):
    domain: ClassVar = AbstractHilbertSpace
    lx_tags: ClassVar = []
    exponentiator: eqx.AbstractVar[AbstractExponentiator]

    def _check_domain(self, y: AbstractState):
        try:
            _reconcile_domains(self.domain, y.hilbert_space.structure)
        except IncompatibleDomainError as e:
            raise IncompatibleDomainError(
                f"{type(self).__name__} acts on {_get_name(self.domain)}, "
                f"but received a state on {_get_name(y.hilbert_space.structure)}"
            )

    def with_exponentiator(self, exponentiator: AbstractExponentiator):
        # Rebuild field-wise rather than via eqx.tree_at: exponentiators are
        # leaf-less modules, so tree_at can't locate them by leaf identity and
        # falls through to a branch that calls len() on the module. Bypassing
        # __init__ is safe -- only the exponentiator changes, and nothing that
        # __check_init__ validates (e.g. domain compatibility) is affected.

        return _update_field(self, "exponentiator", exponentiator)

    # Public, domain-checked entry points
    def __call__(self, y: AbstractState) -> AbstractState:
        self._check_domain(y)
        return self.action(y)

    def exp(self, h: ScalarLike, y: AbstractState) -> AbstractState:
        self._check_domain(y)
        return self.exponentiator.exp(self, h, y)

    # Interfaces
    @abstractmethod
    def action(self, y: AbstractState) -> AbstractState:
        pass

    def exp_action(self, h: ScalarLike, y: AbstractState) -> AbstractState:
        raise NotImplementedError

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
            
            if len(self.lx_tags) > 0:
                lx_op = lx.TaggedLinearOperator(lx_op, self.lx_tags)

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

    @property
    def exp_order(self):
        return self.exponentiator.order

    def to_matrix(self, hilbert_space: AbstractHilbertSpace) -> Array:
        raise NotImplementedError

    # Operator algebra
    def __add__(self, other: Union[Operator, ScalarLike]) -> Operator:
        c = _as_shift(other)
        if c is not None:
            return ShiftOperator(self, c)
        if not isinstance(other, Operator):
            return NotImplemented

        # self may be the identity term instead
        c = _as_shift(self)
        if c is not None:
            return ShiftOperator(other, c)  
            
        return AddOperator(self, other)

    def __radd__(self, other: Union[Operator, ScalarLike]) -> Operator:
        c = _as_shift(other)
        if c is not None:
            return ShiftOperator(self, c)
        if not isinstance(other, Operator):
            return NotImplemented

        # self may be the identity term instead
        c = _as_shift(self)
        if c is not None:
            return ShiftOperator(other, c)

        return AddOperator(other, self)

    def __sub__(self, other: Union[Operator, ScalarLike]) -> Operator:
        c = _as_shift(other)
        if c is not None:
            return ShiftOperator(self, -c)
        if isinstance(other, Operator):
            return self + (-other) 
        return NotImplemented

    def __rsub__(self, other: Union[Operator, ScalarLike]) -> Operator:
        c = _as_shift(other)
        if c is not None:
            return ShiftOperator(-self, c)
        if isinstance(other, Operator):
            return other + (-self) 
        return NotImplemented

    def __mul__(self, other: ScalarLike) -> ScalarMulOperator:
        if not jnp.isscalar(other):
            return NotImplemented

        return ScalarMulOperator(self, other)

    def __rmul__(self, other: ScalarLike) -> ScalarMulOperator:
        if not jnp.isscalar(other):
            return NotImplemented

        return ScalarMulOperator(self, other) 

    def __truediv__(self, other: ScalarLike) -> ScalarMulOperator:
        if not jnp.isscalar(other):
            return NotImplemented

        return ScalarMulOperator(self, 1.0 / other) 

    def __neg__(self) -> ScalarMulOperator:
        return ScalarMulOperator(self, -1.0)

    def to_dict(self, h_scale: int=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "exp_delegated": False
        }


class ShiftOperator(Operator):
    """
    Implements op + c * Identity()
    """
    op: Operator
    c: ScalarLike

    def __init__(self, op: Operator, c: ScalarLike):
        if isinstance(op, ShiftOperator):
            # If op is a ShiftOperator, i.e. op = A + c0 * I
            # then shift can be collapsed to A + (c0 + c) * I
            self.op = op.op
            self.c = c + op.c 
        else:
            self.op = op 
            self.c = c

    @property
    def domain(self): 
        return self.op.domain

    @property
    def exponentiator(self): 
        return self.op.exponentiator

    @property
    def exp_order(self):
        return self.op.exp_order

    def with_exponentiator(self, exponentiator: AbstractExponentiator):
        # exponentiator is a property delegating to self.op, so push the swap
        # into the wrapped operator and re-wrap with the shift.
        return ShiftOperator(self.op.with_exponentiator(exponentiator), self.c)

    def action(self, y):
        return self.op.action(y) + self.c * y

    def exp(self, h: ScalarLike, y: AbstractState) -> AbstractState:
        # The shift factors out exactly, so delegate to the base operator's own
        # exponentiator rather than inheriting Operator.exp, which would hand
        # *this* operator to it. A split method would then look for op1/op2 here
        # and not find them.
        self._check_domain(y)
        return jnp.exp(h * self.c) * self.op.exp(h, y)

    def exp_action(self, h, y):
        return jnp.exp(h * self.c) * self.op.exp_action(h, y)

    def solve(self, b, scale=-1.0, shift=0.0):
        return self.op.solve(b, scale, shift + scale * self.c)

    def spectral_bounds(self, hilbert_space):
        if jnp.iscomplexobj(self.c):
            raise NotImplementedError(
                f"spectral_bounds assumes a real spectrum, but c={self.c} is complex, "
                f"so {type(self.op).__name__} + c * Identity() is not Hermitian"
            )

        return self.op.spectral_bounds(hilbert_space) + self.c

    def to_matrix(self, hilbert_space):
        return self.op.to_matrix(hilbert_space) + self.c * jnp.eye(hilbert_space.dim)

    def to_dict(self, h_scale: int=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "exp_delegated": True, 
            "op": self.op.to_dict(h_scale)
        }


class AddOperator(Operator):
    op1: Operator
    op2: Operator
    exponentiator: AbstractExponentiator = eqx.field(default=Strang(), kw_only=True)

    def __check_init__(self):
        try:
            _reconcile_domains(self.op1.domain, self.op2.domain)
        except IncompatibleDomainError as e:
            raise IncompatibleDomainError(
                f"Incompatible domains: {type(self.op1).__name__} acts on {_get_name(self.op1.domain)}, "
                f"but {type(self.op2).__name__} acts on {_get_name(self.op2.domain)},"
            )

    @property
    def domain(self):
        return _reconcile_domains(self.op1.domain, self.op2.domain)

    @property
    def exp_order(self):
        return self.exponentiator.effective_order(self)

    def action(self, y: AbstractState) -> AbstractState:
        return self.op1.action(y) + self.op2.action(y)

    def spectral_bounds(self, hilbert_space: AbstractHilbertSpace) -> Array:
        return self.op1.spectral_bounds(hilbert_space) + self.op2.spectral_bounds(hilbert_space)

    def to_matrix(self, hilbert_space: AbstractHilbertSpace) -> Array:
        return self.op1.to_matrix(hilbert_space) + self.op2.to_matrix(hilbert_space)

    def to_dict(self, h_scale: int=1.0) -> dict:
        if isinstance(self.exponentiator, AbstractSplitMethod):
            h1, h2 = self.exponentiator.h_scales
            op1_val = self.op1.to_dict(h1 * h_scale)
            op2_val = self.op2.to_dict(h2 * h_scale)
        else:
            op1_val = None
            op2_val = None
        
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "exp_delegated": False,
            "op1": op1_val,
            "op2": op2_val
        }


class ScalarMulOperator(Operator):
    op: Operator
    c: ScalarLike

    def __init__(self, op: Operator, c: ScalarLike):
        if isinstance(op, ScalarMulOperator):
            # If op is a ScalarMulOperator, i.e. op = c0 * A
            # then scalar multiplication can be collapsed to c * c0 * A
            self.op = op.op
            self.c = c * op.c 
        else:
            self.op = op 
            self.c = c

    @property
    def domain(self):
        return self.op.domain

    @property
    def exponentiator(self):
        return self.op.exponentiator

    @property
    def exp_order(self):
        return self.op.exp_order

    def with_exponentiator(self, exponentiator: AbstractExponentiator):
        return self.c * self.op.with_exponentiator(exponentiator)

    def action(self, y: AbstractState) -> AbstractState:
        return self.c * self.op.action(y)

    def exp(self, h: ScalarLike, y: AbstractState) -> AbstractState:
        return self.op.exp(self.c * h, y)

    def exp_action(self, h: ScalarLike, y: AbstractState) -> AbstractState:
        return self.op.exp_action(self.c * h, y)

    def solve(self, b: AbstractState, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> AbstractState:
        return self.op.solve(b, self.c * scale, shift)

    def spectral_bounds(self, hilbert_space: AbstractHilbertSpace) -> Array:
        if jnp.iscomplexobj(self.c):
            raise NotImplementedError(
                f"spectral_bounds assumes a real spectrum, but c={self.c} is complex, "
                f"so c * {type(self.op).__name__} is not Hermitian"
            )

        return jnp.sort(self.c * self.op.spectral_bounds(hilbert_space))

    def to_matrix(self, hilbert_space: AbstractHilbertSpace) -> Array:
        return self.c * self.op.to_matrix(hilbert_space)

    def to_dict(self, h_scale: int=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "exp_delegated": True,
            "op": self.op.to_dict(h_scale=jnp.abs(self.c) * h_scale),
        }


class Identity(Operator):
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

from typing import Iterable
from functools import reduce

import equinox as eqx
from jaxtyping import ArrayLike, ScalarLike

from ._internal import _update_field
from .hilbert_space import AbstractHilbertSpace
from .operator import Operator, AddOperator
from .timevarying_operator import AbstractTimeVaryingOperator, ConstantTimeVaryingOperator
from .exponentiators import AbstractSplitMethod, Strang


class ControlledOperator(eqx.Module):
    drift_op: AbstractTimeVaryingOperator
    controlled_ops: tuple[AbstractTimeVaryingOperator, ...] = eqx.field(default=())
    split_method: AbstractSplitMethod = eqx.field(default=Strang(), kw_only=True)

    def __init__(
        self, 
        drift_op: Operator | AbstractTimeVaryingOperator, 
        controlled_ops: Iterable[Operator | AbstractTimeVaryingOperator]=(), 
        split_method=Strang()):
        
        if isinstance(drift_op, Operator):
            drift_op = ConstantTimeVaryingOperator(drift_op)

        controlled_ops = tuple(
            ConstantTimeVaryingOperator(op) if isinstance(op, Operator) else op
            for op in controlled_ops)

        self.drift_op = drift_op 
        self.controlled_ops = controlled_ops
        self.split_method = split_method

    def __check_init__(self):
        for idx, op in enumerate(self.controlled_ops):
            if not op.domain == self.drift_op.domain:
                raise ValueError(
                    f"All operators must act on the same domain but drift_op.domain={self.drift_op.domain} "
                    f"and controlled_ops[{idx}].domain={op.domain}")

    @property
    def domain(self) -> AbstractHilbertSpace:
        return self.drift_op.domain

    @property
    def num_controls(self) -> int:
        return len(self.controlled_ops)

    def with_split_method(self, split_method: AbstractSplitMethod):
        return _update_field(self, "split_method", split_method)

    def quadrature(
        self, 
        u_quad: ArrayLike, 
        t_quad: ArrayLike, 
        weights: ArrayLike) -> Operator:

        # u_quad.shape == (len(self.controlled_ops), num_nodes)

        H0 = self.drift_op.quadrature(t_quad, weights)
        H_list = [H0] + [
            H.quadrature(t_quad, u * weights) for (H, u) in zip(self.controlled_ops, u_quad)]

        return reduce(lambda a, b: AddOperator(a, b, exponentiator=self.split_method), H_list)

    def __call__(
        self, 
        t: ScalarLike,
        controls: ArrayLike) -> Operator | AbstractTimeVaryingOperator:

        op = reduce(
            lambda a, b: (a + b).with_split_method(self.split_method), 
            [self.drift_op] + [u * op for (u, op) in zip(controls, self.controlled_ops)])

        return op(t)

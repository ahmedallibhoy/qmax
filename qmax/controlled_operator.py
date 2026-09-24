from functools import reduce
from typing import Iterable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ScalarLike

from ._internal import _update_field
from ._types import ComplexArrayLike
from .exponentiators.split import AbstractSplitMethod, Strang
from .hilbert_space import AbstractHilbertSpace
from .operator import AddOperator, Operator
from .timevarying_operator import AbstractTimeVaryingOperator, ConstantTimeVaryingOperator

__all__ = ["ControlledOperator"]


class ControlledOperator(eqx.Module):
    r"""
    Controlled operator of the form $H(t, u) = H_0(t) + \sum_{j=1}^{m}u_jH_j(t)$.

    Attributes:
        drift_op (Operator | AbstractTimeVaryingOperator): The drift operator $H_0(t)$
        controlled_ops (Iterable[Operator | AbstractTimeVaryingOperator]): An iterable of
            controlled operators $H_j(t)$.
        split_method (Optional[AbstractSplitMethod]): The split method used to exponentiate
            the sum $H_0(t) + \sum_{j=1}^{m}u_jH_j(t)$, (see [Split](../exponentiators/split.md)
            for details). If `split_method=None` then [qmax.exponentiators.Strang][] is used.
    """

    drift_op: AbstractTimeVaryingOperator
    controlled_ops: tuple[AbstractTimeVaryingOperator, ...] = eqx.field(default=())
    split_method: AbstractSplitMethod = eqx.field(default=Strang(), kw_only=True)

    def __init__(
        self,
        drift_op: Operator | AbstractTimeVaryingOperator,
        controlled_ops: Iterable[Operator | AbstractTimeVaryingOperator] = (),
        split_method: AbstractSplitMethod = Strang(),
    ):

        if isinstance(drift_op, Operator):
            drift_op = ConstantTimeVaryingOperator(drift_op)

        controlled_ops = tuple(
            ConstantTimeVaryingOperator(op) if isinstance(op, Operator) else op
            for op in controlled_ops
        )

        self.drift_op = drift_op
        self.controlled_ops = controlled_ops
        self.split_method = split_method

    def __check_init__(self):
        for idx, op in enumerate(self.controlled_ops):
            if not op.domain == self.drift_op.domain:
                raise ValueError(
                    f"All operators must act on the same domain "
                    f"but drift_op.domain={self.drift_op.domain} "
                    f"and controlled_ops[{idx}].domain={op.domain}"
                )

    @property
    def domain(self) -> AbstractHilbertSpace:
        return self.drift_op.domain

    @property
    def num_controls(self) -> int:
        return len(self.controlled_ops)

    def with_split_method(self, split_method: AbstractSplitMethod):
        return _update_field(self, "split_method", split_method)

    def quadrature(
        self, t_quad: ComplexArrayLike, u_quad: ComplexArrayLike, weights: ComplexArrayLike
    ) -> Operator:

        # u_quad.shape == (len(self.controlled_ops), num_nodes)

        H0 = self.drift_op.quadrature(t_quad, weights)
        H_list = [H0] + [
            H.quadrature(t_quad, u * weights) for (H, u) in zip(self.controlled_ops, u_quad)
        ]

        return reduce(lambda a, b: AddOperator(a, b, exponentiator=self.split_method), H_list)

    def __call__(self, t: ScalarLike, controls: ComplexArrayLike) -> Operator:
        """
        Evaluates the controlled operator given a time t and an array of control inputs `controls`.

        Args:
            t (Scalar): Evaluation time
            controls (Array): Array of shape `(num_controls,)`

        Returns:
            the operator $H(t, u)$.
        """
        controls = jnp.asarray(controls)

        op = reduce(
            lambda a, b: (a + b).with_split_method(self.split_method),
            [self.drift_op] + [u * op for (u, op) in zip(controls, self.controlled_ops)],
        )

        return op(t)

import equinox as eqx

from jaxtyping import ScalarLike

from .hilbert_space import AbstractHilbertSpace
from .operator import Operator
from .tensor import AbstractTensorOperator


def _adapt_dict(
    op_dict: dict, 
    hilbert_space: AbstractHilbertSpace, 
    dt_max: ScalarLike) -> dict:

    root = op_dict["obj"]

    if isinstance(root, AbstractTensorOperator):
        if "ops" in op_dict and op_dict["ops"] is not None:
            # KroneckerSum
            op_dict["ops"] = tuple(
                _adapt_dict(op, hilbert_space[idx], dt_max) 
                for idx, op in enumerate(op_dict["ops"])
            )
            root = eqx.tree_at(lambda o: o.ops, root, [d["obj"] for d in op_dict["ops"]])

        if "op" in op_dict and op_dict["op"] is not None:
            # Lift
            op_dict["op"] = _adapt_dict(op_dict["op"], hilbert_space[root.factor_idx], dt_max)
            root = eqx.tree_at(lambda o: o.op, root, op_dict["op"]["obj"])
    else:
        for key in ("op", "op1", "op2"):
            if key in op_dict and op_dict[key] is not None:
                op_dict[key] = _adapt_dict(op_dict[key], hilbert_space, dt_max)
                root = eqx.tree_at(lambda o, k=key: getattr(o, k), root, op_dict[key]["obj"])

    op_dict["obj"] = root
    if op_dict["exp_delegated"]:
        return op_dict

    new_exp = root.exponentiator.adapt(root, hilbert_space, dt_max * op_dict["h_scale"])
    op_dict["obj"] = root.with_exponentiator(new_exp)
    return op_dict


def adapt_operator(
    op: Operator, 
    hilbert_space: AbstractHilbertSpace, 
    dt_max: ScalarLike) -> Operator:

    adapted = _adapt_dict(op.to_dict(), hilbert_space, dt_max)
    return adapted["obj"]

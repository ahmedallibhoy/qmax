from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable
import warnings

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray, Array, Scalar, ArrayLike, ScalarLike

from .hilbert_space import AbstractHilbertSpace, AbstractState
if TYPE_CHECKING:
    from .operator import Operator


def op_lanczos(
    op: Operator,
    hilbert_space: AbstractHilbertSpace, 
    num_iterations: int, 
    orthogonalize: bool=True,
    key: PRNGKeyArray=jax.random.key(0), 
    w0: Optional[AbstractState]=None) -> tuple[Array, Array, AbstractState]:

    """
    Implements the Lanczos tridiagonalization algorithm as described in Chapter 10 of: 
        Golub, Gene H., and Charles F. Van Loan. Matrix computations. JHU press, 2013.
    """

    if num_iterations > hilbert_space.dim:
        warnings.warn(
            f"Received num_iterations={num_iterations} which is greater than hilbert_space.dim={hilbert_space.dim}. "
            f"Iterations will be capped at {hilbert_space.dim}",
            stacklevel=2,
        )
        num_iterations = hilbert_space.dim

    def gram_schmidt2(y, Z):
        # Gram-Schmidt process
        y1 =  y - Z.contract(Z @ y)

        # redo Gram-Schmidt for numerical stability
        y2 =  y1 - Z.contract(Z @ y1)

        return y2

    def lanczos_step(beta, q, w):
        q_next = w / jnp.where(beta == 0, 1.0, beta)
        z = op.action(q_next)        
        alpha_next = jnp.real(q_next @ z)
        w_next = z - alpha_next * q_next - beta * q
        return alpha_next, q_next, w_next

    def lanczos_outer_loop(carry, _):
        idx, beta, Q, w = carry

        alpha_next, q_next, w_next = lanczos_step(beta, Q[idx], w)
        Q_next = Q.at[idx + 1].set(q_next) 

        if orthogonalize:
            w_next = gram_schmidt2(w_next, Q_next)

        beta_next = w_next.norm()
        carry_next = (idx + 1, beta_next, Q_next, w_next)

        return carry_next, (alpha_next, beta_next) 

    if w0 is None:
        w0 = hilbert_space.random(key)

    Q0 = hilbert_space.zeros((num_iterations + 1,))

    init = (0, w0.norm(), Q0, w0)
    carry, (alpha, beta) = jax.lax.scan(lanczos_outer_loop, init, length=num_iterations)
    _, _, Q, _ = carry

    return alpha, beta, Q[1:]


from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import warnings

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray, Array

from .hilbert_space import AbstractHilbertSpace, AbstractState
if TYPE_CHECKING:
    from .operator import Operator


def gram_schmidt2(y, Z):
    # Gram-Schmidt process
    y1 =  y - Z.contract(Z @ y)
    # redo Gram-Schmidt for numerical stability
    y2 =  y1 - Z.contract(Z @ y1)

    return y2


def op_lanczos(
    op: Operator,
    hilbert_space: AbstractHilbertSpace, 
    num_iterations: int, 
    *,
    orthogonalize: bool=True,
    key: PRNGKeyArray=jax.random.key(0), 
    w0: Optional[AbstractState]=None,
    Q0: Optional[AbstractState]=None,
    idx0: int=0) -> tuple[Array, Array, AbstractState]:

    """
    Implements the Lanczos tridiagonalization algorithm as described in Chapter 10 of: 
        Golub, Gene H., and Charles F. Van Loan. Matrix computations. JHU press, 2013.
    """

    if num_iterations > hilbert_space.dim - idx0:
        warnings.warn(
            f"Received num_iterations={num_iterations} which is greater than hilbert_space.dim - idx0={hilbert_space.dim - idx0}. ",
            stacklevel=2,
        )

    def lanczos_step(beta, q, w):
        q_next = w / jnp.where(beta == 0, 1.0, beta)
        z = op.action(q_next)        
        alpha_next = jnp.real(q_next @ z)
        w_next = z - alpha_next * q_next - beta * q
        return alpha_next, q_next, w_next

    def lanczos_loop(carry, _):
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

    if Q0 is None:
        Q0 = hilbert_space.zeros((num_iterations + 1,))

    init = (idx0, w0.norm(), Q0, w0)
    carry, (alpha, beta) = jax.lax.scan(lanczos_loop, init, length=num_iterations)
    _, _, Q, w = carry

    return alpha, beta, Q[1:], w


def _select_indices(select, theta, num):
    # Return the `num` indices selected by the given strategy (sorted by its key).
    match select:
        case "smallest":
            key = theta
        case "largest":
            key = -theta
        case "largest_magnitude":
            key = -jnp.abs(theta)
    return jnp.argsort(key)[:num]


def restart_lanczos(
    op: Operator,
    hilbert_space: AbstractHilbertSpace, 
    num_ritz: int,
    max_krylov_dim: int=100,
    num_restarts: int=4,
    select: str="smallest",
    key: PRNGKeyArray=jax.random.key(0)) -> tuple[Array, AbstractState, Array]:

    """
    Essentially reproduces the method described in:
        Wu, Kesheng, and Horst Simon. "Thick-restart Lanczos method for large symmetric eigenvalue problems." 
        SIAM Journal on Matrix Analysis and Applications 22.2 (2000): 602-616.
    """

    # need (max_krylov_dim - num_ritz) ~ num_ritz

    if num_ritz >= max_krylov_dim:
        raise ValueError(f"num_ritz={num_ritz} must be < max_krylov_dim={max_krylov_dim}")

    def selectk(theta, Y, Q):
        indices = _select_indices(select, theta, num_ritz)
        Yk, thetak = Y[:, indices], theta[indices]
        sk = Yk[-1, :]
        Qk = Q.contract(Yk)
        return Qk, sk, thetak

    def lanczos_loop(sk, beta_m, Qk, wk):
        q0 = wk / jnp.where(beta_m == 0, 1.0, beta_m)
        z = op.action(q0)
        alpha0 = jnp.real(q0 @ z)
        w0 = z - alpha0 * q0 - beta_m * Qk.contract(sk)

        Q0 = hilbert_space.zeros((max_krylov_dim + 1,))
        Q0 = Q0.at[1:num_ritz + 1].set(Qk)
        Q0 = Q0.at[num_ritz + 1].set(q0)

        w0 = gram_schmidt2(w0, Q0)
        beta0 = w0.norm()

        alpha, beta, Q, w = op_lanczos(op, hilbert_space, max_krylov_dim - num_ritz - 1,
            orthogonalize=True, w0=w0, Q0=Q0, idx0=num_ritz + 1)

        alpha = jnp.append(alpha0, alpha)
        beta = jnp.append(beta0, beta)
        return alpha, beta, Q, w

    def reconstruct(alpha, beta, sk, thetak, beta_m):
        tri = jnp.diag(alpha) + jnp.diag(beta[:-1], k=1) + jnp.diag(beta[:-1], k=-1)
        border = jnp.zeros((num_ritz, max_krylov_dim - num_ritz)).at[:, 0].set(jnp.real(beta_m * sk))
        T_next = jnp.block([[jnp.diag(thetak), border], [border.T, tri]])
        return T_next

    def restart(carry, _):
        Qk, sk, thetak, wk = carry
        beta_m = wk.norm()

        alpha, beta, Q, w = lanczos_loop(sk, beta_m, Qk, wk)
        T_next = reconstruct(alpha, beta, sk, thetak, beta_m)
        theta_next, Y_next = jnp.linalg.eigh(T_next)

        Qk_next, sk_next, thetak_next = selectk(theta_next, Y_next, Q)
        carry_next = (Qk_next, sk_next, thetak_next, w)

        return carry_next, None

    alpha, beta, Q, w = op_lanczos(op, hilbert_space, max_krylov_dim, orthogonalize=True, key=key)

    theta, Y = jax.scipy.linalg.eigh_tridiagonal(alpha, beta[:-1])
    Qk, sk, thetak = selectk(theta, Y, Q)
    init_restart = (Qk, sk, thetak, w)
    (Qk, sk, thetak, wk), _ = jax.lax.scan(restart, init_restart, length=num_restarts)
    
    beta_m = wk.norm()
    eigvals, eigvecs, residuals = thetak, Qk, beta_m * jnp.abs(sk)
    indices = _select_indices(select, thetak, num_ritz)
    eigvals, eigvecs, residuals = eigvals[indices], eigvecs[indices], residuals[indices]

    return eigvals, eigvecs, residuals

from __future__ import annotations
from typing import Optional, Callable

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray, Array, Scalar, ArrayLike, ScalarLike


type Matvec = Callable[[ArrayLike], Array]


def _lanczos_dtype(is_complex: bool):
    real_dtype = jnp.zeros(()).dtype
    return jnp.result_type(real_dtype, 1j) if is_complex else real_dtype


def _select_indices(select, theta, residuals, num):
    match select:
        case "smallest":
            key = theta
        case "largest":
            key = -theta
        case "largest_magnitude":
            key = -jnp.abs(theta)
        case "smallest_residuals":
            key = residuals
        case _:
            raise ValueError(f"Invalid select={select!r}; expected one of 'smallest', "
                "'largest', 'largest_magnitude', 'smallest_residuals'")
    return jnp.argsort(key)[:num]


def safe_norm(v: ArrayLike, tol: float) -> Scalar:
    # ensures norm is differentiable when v == 0
    norm2 = jnp.real(v.conj() @ v)
    is_zero = (norm2 <= tol * tol)
    norm2_safe = jnp.where(is_zero, 1.0, norm2)
    return jnp.where(is_zero, 0.0, jnp.sqrt(norm2_safe))


def restart_lanczos(
    matvec: Matvec,
    dim: int,
    num_eig: int,
    num_iterations: int=100,
    *,
    buffer: int=25,
    num_restarts: int=4,
    select: str="smallest",
    is_complex: bool=False,
    key: PRNGKeyArray=jax.random.key(0)) -> tuple[Array, Array, Array]:

    """
    Essentially reproduces the method described in:
        Wu, Kesheng, and Horst Simon. "Thick-restart Lanczos method for large symmetric eigenvalue problems." 
        SIAM Journal on Matrix Analysis and Applications 22.2 (2000): 602-616.

    The 'thick' mode is a faithful recreation of the algorithm in the above referenced paper. 
    """

    num_ritz = num_eig + buffer
    dtype = _lanczos_dtype(is_complex)

    if num_ritz >= num_iterations:
        raise ValueError(f"num_ritz (num_eig+buffer={num_ritz}) must be < num_iterations={num_iterations}")

    def selectk(theta, Y, beta, Q):
        residuals = jnp.abs(beta[-1] * Y[-1, :])
        indices = _select_indices(select, theta, residuals, num_ritz)
        Yk, thetak = Y[:, indices], theta[indices]
        sk = Yk[-1, :]
        Qk = Q @ Yk
        return Qk, sk, thetak

    def first_iteration(sk, beta_m, Qk, wk):
        q_next = wk / jnp.where(beta_m == 0, 1.0, beta_m)
        z = matvec(q_next)
        alpha_next = jnp.real(q_next.conj() @ z)
        w_next = z - alpha_next * q_next - beta_m * (Qk @ sk)
        beta_next = jnp.linalg.norm(w_next)
        return alpha_next, beta_next, q_next, w_next

    def lanczos_loop(sk, beta_m, Qk, wk):
        alpha0, beta0, q0, w0 = first_iteration(sk, beta_m, Qk, wk)
        Q0 = jnp.zeros((dim, num_iterations,), dtype=dtype)
        Q0 = Q0.at[:, :num_ritz].set(Qk)
        Q0 = Q0.at[:, num_ritz].set(q0)
        init = (num_ritz, beta0, Q0, w0)

        alpha, beta, Q, w = lanczos(matvec, dim, num_iterations - num_ritz - 1,
            orthogonalize=True, return_ritz=True, return_residual=True,
            is_complex=is_complex, init=init)

        alpha = jnp.append(alpha0, alpha)
        beta = jnp.append(beta0, beta)
        return alpha, beta, Q, w

    def reconstruct(alpha, beta, sk, thetak, beta_m):
        tri = jnp.diag(alpha) + jnp.diag(beta[:-1], k=1) + jnp.diag(beta[:-1], k=-1)
        border = jnp.zeros((num_ritz, num_iterations - num_ritz)).at[:, 0].set(jnp.real(beta_m * sk))
        T_next = jnp.block([[jnp.diag(thetak), border], [border.T, tri]])
        return T_next

    def restart(carry, _):
        Qk, sk, thetak, wk = carry
        beta_m = jnp.linalg.norm(wk)

        alpha, beta, Q, w = lanczos_loop(sk, beta_m, Qk, wk)
        T_next = reconstruct(alpha, beta, sk, thetak, beta_m)
        theta_next, Y_next = jnp.linalg.eigh(T_next)

        Qk_next, sk_next, thetak_next = selectk(theta_next, Y_next, beta, Q)
        carry_next = (Qk_next, sk_next, thetak_next, w)

        return carry_next, None

    alpha, beta, Q, w = lanczos(matvec, dim, num_iterations,
        orthogonalize=True, return_ritz=True, return_residual=True,
        is_complex=is_complex, key=key)

    theta, Y = jax.scipy.linalg.eigh_tridiagonal(alpha, beta[:-1])
    Qk, sk, thetak = selectk(theta, Y, beta, Q[:, 1:])
    init_restart = (Qk, sk, thetak, w)
    (Qk, sk, thetak, wk), _ = jax.lax.scan(restart, init_restart, length=num_restarts)
    beta_m = jnp.linalg.norm(wk) 

    eigvals, eigvecs, residuals = thetak, Qk, beta_m * jnp.abs(sk)
    indices = _select_indices(select, eigvals, residuals, num_eig)
    eigvals, eigvecs, residuals = eigvals[indices], eigvecs[:, indices], residuals[indices]
    return eigvals, eigvecs, residuals


def eigh_lanczos_matvec(
    matvec: Matvec,
    dim: int,
    num_eig: int,
    num_iterations: int=200,
    *,
    eigvals_only: bool=False,
    return_residuals: bool=False,
    select: str="smallest",
    orthogonalize: bool=True,
    grad_safe: bool=False,
    tol: Optional[float]=None,
    is_complex: bool=False,
    key: PRNGKeyArray=jax.random.key(0)) -> tuple[Array, Optional[Array], Optional[Array]]:

    result = lanczos(matvec, dim, num_iterations, orthogonalize=orthogonalize,
        return_ritz=not eigvals_only, grad_safe=grad_safe, tol=tol,
        is_complex=is_complex, key=key)
        
    if eigvals_only:
        alpha, beta = result
    else:
        alpha, beta, Q = result

    need_vectors = (not eigvals_only) or return_residuals or (select == "smallest_residuals")

    if need_vectors:
        eigvals, Y = jax.scipy.linalg.eigh_tridiagonal(alpha, beta[:-1])
        residuals = jnp.abs(beta[-1] * Y[-1, :])
    else:
        eigvals = jax.scipy.linalg.eigh_tridiagonal(alpha, beta[:-1], eigvals_only=True)
        residuals = None

    indices = _select_indices(select, eigvals, residuals, num_eig)
    eigvals = eigvals[indices]
    residuals = residuals[indices] if return_residuals else None
    eigvecs = None if eigvals_only else Q[:, 1:] @ Y[:, indices]
    return eigvals, eigvecs, residuals


def lanczos(
    matvec: Matvec,
    dim: int, 
    num_iterations: int=200, 
    orthogonalize: bool=False, 
    return_ritz: bool=False, 
    return_residual: bool=False,
    grad_safe: bool=False,
    tol: Optional[float]=None,
    is_complex: bool=False,
    key: PRNGKeyArray=jax.random.key(0),
    init: Optional[tuple[int, ScalarLike, ArrayLike, ArrayLike]]=None) -> tuple[Array, ...]:

    """
    Implements the Lanczos tridiagonalization algorithm as described in Chapter 10 of:
        Golub, Gene H., and Charles F. Van Loan. Matrix computations. JHU press, 2013.
    """

    if init is not None:
        # init supplies (idx, beta, Q, w), which only the Q-tracking branch
        # consumes, so return_ritz is genuinely required. orthogonalize is not:
        # that branch applies Gram-Schmidt only when asked, and forcing it here
        # silently ignored the caller's choice.
        return_ritz = True

    if grad_safe:
        eps = jnp.finfo(jnp.zeros((0)).dtype).eps
        tol = jnp.sqrt(eps * dim) if tol is None else tol
    else:
        tol = 0.0

    dtype = _lanczos_dtype(is_complex)

    def gram_schmidt2(y, Z):
        # Gram-Schmidt process
        y1 =  y - Z @ (Z.conj().T @ y)

        # redo Gram-Schmidt for numerical stability
        y2 =  y1 - Z @ (Z.conj().T @ y1) 

        return y2

    def lanczos_step(beta, q, w):
        q_next = w / jnp.where(jnp.abs(beta) <= tol, 1.0, beta)
        z = matvec(q_next)        
        alpha_next = jnp.real(q_next.conj() @ z)
        w_next = z - alpha_next * q_next - beta * q
        return alpha_next, q_next, w_next

    if orthogonalize or return_ritz:
        if init is None:
            idx0 = 0
            beta0 = 1.0
            w0 = jax.random.normal(key, shape=(dim,), dtype=dtype)
            w0 = w0 / jnp.linalg.norm(w0)
            Q0 = jnp.zeros(shape=(dim, num_iterations + 1), dtype=dtype)
            init = (idx0, beta0, Q0, w0)

        def lanczos_outer_loop(carry, _):
            idx, beta, Q, w = carry

            alpha_next, q_next, w_next = lanczos_step(beta, Q[:, idx], w)
            Q_next = Q.at[:, idx + 1].set(q_next) 

            if orthogonalize:
                w_next = gram_schmidt2(w_next, Q_next)

            beta_next = safe_norm(w_next, tol)
            carry_next = (idx + 1, beta_next, Q_next, w_next)

            return carry_next, (alpha_next, beta_next) 

        carry, (alpha, beta) = jax.lax.scan(lanczos_outer_loop, init, length=num_iterations)

    else:
        w0 = jax.random.normal(key, shape=(dim,), dtype=dtype)
        w0 = w0 / jnp.linalg.norm(w0)
        q0 = jnp.zeros(shape=(dim,), dtype=dtype)
        init = (1.0, q0, w0)

        def lanczos_outer_loop(carry, _):
            beta, q, w = carry

            alpha_next, q_next, w_next = lanczos_step(beta, q, w)
            beta_next = safe_norm(w_next, tol)
            carry_next = (beta_next, q_next, w_next)

            return carry_next, (alpha_next, beta_next) 

        carry, (alpha, beta) = jax.lax.scan(lanczos_outer_loop, init, length=num_iterations)

    if orthogonalize or return_ritz:
        _, _, Q, w = carry
    else:
        _, _, w = carry

    if return_ritz and return_residual:
        return alpha, beta, Q, w
    elif return_ritz:
        return alpha, beta, Q 
    elif return_residual:
        return alpha, beta, w
    else:
        return alpha, beta 
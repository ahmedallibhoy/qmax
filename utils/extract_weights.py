import jax.numpy as jnp
import numpy as np


def _gl_nodes(n):
    x, _ = np.polynomial.legendre.leggauss(n)   
    return jnp.array((x + 1) / 2) 


def _build_weights(n, F):
    # Convert expansion coefficients F[i, k] (Omega_i = sum_k F[i,k] A_k, where A_k are
    # the shifted-Legendre time-moments of H) into nodal weights on the n-point
    # Gauss-Legendre grid.  A. Alvermann and H. Fehske, "High-order commutator-free
    # exponential time-propagation of driven quantum systems," J. Comput. Phys.,
    # vol. 230, no. 15, pp. 5930-5956, 2011, Eq. (60):
    #   g[i,m] = w_m * sum_k (2k+1) P_k(x_m) f[i,k],   P_k shifted Legendre on [0, 1].
    F = np.asarray(F)
    x, w = np.polynomial.legendre.leggauss(n)
    tau = (x + 1) / 2                                   # GL nodes on [0, 1]
    K = F.shape[1]
    P = np.array([np.polynomial.legendre.Legendre.basis(k)(2 * tau - 1) for k in range(K)])
    G = (w / 2)[None, :] * (2 * np.arange(K) + 1)[:, None] * P     # (K, n)
    return jnp.asarray(F @ G)

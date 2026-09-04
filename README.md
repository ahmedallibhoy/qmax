# qmax

qmax is a JAX-based library for 
simulating quantum mechanical systems. Features include

- Full compatibility with JAX transformations including just-in-time 
compilation, vmap, and autodiff. 
- A suite of geometric and structure preserving integrators including 
operator splitting methods, composed methods, for approximating 
the exponential action of a Hamiltonian, as well as commutator-free 
quasi-Magnus expansions for simulating time-dependent Hamiltonians.
- Additional exponential approximations including Krylov subspace methods, 
Chebyshev and Laguerre polynomial approximations, and truncated Taylor expansions

The distinguishing feature of qmax is that it provides a convenient interface 
to hiearchically compose numerical methods to simulate complex systems. Consider 
the problem of simulating the Schrodinger equation
$$
-i\dot{\psi} = H(t)\psi
$$
where $H(t) = u_1(t)H_1 + u_2(t)H_2$. In the case where $H_1$ and $H_2$ are exactly exponentiable, 
a solver for this system be built as follows:

1. Implement the exact exponentials $\exp(-i\tau H_1)$ and $\exp(-i\tau H_2)$
2. Approximate the exponential of $u_1(t)H_1 + u_2H_2(t)$ using *Strang splitting*
3. Use a composition method to increase the order of the split approximation 
4. Use a commutator-free quasi-Magnus expansion approximate the propagator of $H(t)$ using 
the exponential approximations. 

## Simple Example
```python
import jax.numpy as jnp
import qmax as qx

hilbert_space = qx.spaces.FiniteDifference(x0=-10, xf=10, num_steps=1000)

T = -0.5 * hilbert_space.laplacian()
V = hilbert_space.potential_energy(lambda x: 0.5 * x ** 2)
H = T + V

U = qx.propagator(H, t0=0, t1=2 * jnp.pi, dt_max=0.01)
y0 = hilbert_space.from_function(lambda x: jnp.exp(-0.5 * (x - 2) ** 2))
y0 = y0 / y0.norm

result = U.propagate(y0)
y1 = result.y1
```
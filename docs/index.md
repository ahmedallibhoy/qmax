# Overview
qmax is a JAX-based library for simulating quantum mechanical systems. 

The library aims to provide a convenient interface to define complex Hamiltonians 
and compose numerical methods to compute their spectra and exponential actions, 
as well as a comprehensive suite of geometric numerical integration schemes for 
fast and efficient simulation of the Schrödinger equation 

Features include:

- Suite of methods to approximate exponentials of operators, including
    - Splitting methods
    - High-order composition methods 
    - Krylov subspace approximations
    - Chebyshev polynomial approximations
    - Scaled and truncated Taylor expansions
- Commutator-free quasi-Magnus expansions for time-varying Hamiltonians
- Memory efficient matrix-free eigensolvers
- Adjoint methods for solving optimal control problems
- Tools for static analysis of numerical methods
- Full compatibility with JAX transformations (JIT, vmap, autodiff)


## Installation

```
pip install qmax
```

## Quick example

```python
import jax.numpy as jnp
import qmax as qx 

hilbert_space = qx.spaces.FiniteDifference(x0=-10, x1=10, num_steps=500)
L = hilbert_space.laplacian()
V = hilbert_space.potential_energy(lambda x: 0.5 * x ** 2)
H = -0.5 * L + V

U = qx.Propagator(H, t0=0.0, t1=2 * jnp.pi, dt_max=0.01)

y0 = hilbert_space.from_function(lambda x: jnp.exp(-0.5 * (x - 1) ** 2))
y0 = y0 / y0.norm()

result = U.propagate(y0)
y1 = result.y1
```

## Road map

This project is in the early development stage, and the library 
is still unstable, but there are a lot interesting extensions I have planned 
including

- Density matrices
- Open quantum systems:
    - Stochastic unraveling schemes of Lindbladians
    - Dynamical low-rank approximations of quantum master equations
- Nonlinear and quasi-linear Schrodinger equations
- Support for arbitrary / user-generated spatial meshes

If you have specific need or you have an idea for an interesting feature, 
please contact me!
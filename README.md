qmax
=====================

## Overview
qmax is a JAX-based library for simulating quantum mechanical systems. 

The library aims to provide a convenient interface to define complicated Hamiltonians 
and provides a comprehensive suite of geometric numerical integration schemes for 
fast and efficient simulation of the Schrödinger equation. 

Features include:

- Suite of methods to approximate exponentials of operators, including
    - Splitting methods
    - High-order composition methods 
    - Krylov subspace approximations
    - Chebyshev polynomial approximations
    - Scaled and truncated Taylor expansions
- Exponential timesteppers based on the commutator-free quasi-Magnus expansion
- Adjoint methods for solving optimal control problems
- Memory efficient matrix-free eigensolvers
- Tools for static analysis of numerical methods
- Full compatibility with JAX transformations (JIT, vmap, autodiff)


<!--
## Documentation
The documentation is available at https://docs.qmax.io
-->

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
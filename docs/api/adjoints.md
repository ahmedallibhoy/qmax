# Adjoints

Since qmax is built on top of JAX, we may use autodiff to differentiate through the solver. This 
makes the library ideal for optimal control applications. For example, consider the Schrödinger equation 

$$
i\hbar\dot{\psi}(t) = H(t, u(t))\psi(t)
$$

where $H(t, u(t))$ is a time-varying controlled Hamiltonian and $u(t)$ is a control input. Suppose 
that we want to optimize the following cost function with respect to the control or initial condition:

$$
J\big(\psi_0;  u(\cdot)\big) = V(t_1, \psi(t_1)) + \int_{t_0}^{t_1}\ell(t, u(t), \psi(t))dt.
$$

We can solve the problem with any gradient-based optimization algorithm, using JAX to automatically differentiate 
through the solver to compute $\frac{\partial J}{\partial \psi_0}$ and $\frac{\partial J}{\partial u(t)}$. 
Mathematically, this is equivalent to obtaining the derivatives by numerically solving a differential equation 
called the *adjoint*.

qmax provides several adjoint methods that determine various details about how JAX differentiates through 
the internals of the solver, such as checkpointing (saving intermediate values of the computation called *residuals*), and 
the method to reconstruct unsaved residuals, allowing the user to control the tradeoff between memory and computational speed.

Unlike most standard ODE integrators (e.g. those provided by libraries like [diffrax](https://docs.kidger.site/diffrax/)), 
the CFETs provided by qmax are *symmetric*, meaning that the exact trajectory can be reconstructed by stepping backward 
through the solver. This allows qmax to offer a fast and memory-efficient [qmax.adjoint.ReversibleAdjoint][] method, which 
is the default used by [qmax.Propagator][]. 

::: qmax.adjoint.DirectAdjoint 
    options:
        members:
            -

::: qmax.adjoint.ReversibleAdjoint
    options:
        members:
            -

::: qmax.adjoint.CheckpointedAdjoint 
    options:
        members:
            -
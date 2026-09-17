# Adjoints

Consider the Schrödinger equation 

$$
i\hbar\dot{\psi}(t) = H(t, u(t))\psi(t)
$$

where $H(t, u(t))$ is a time-varying controlled Hamiltonian and $u(t)$ is a control input. Suppose 
that we want to optimize the following cost function with respect to the control or initial condition:

$$
J\big(\psi_0;  u(\cdot)\big) = V(t_1, \psi(t_1)) + \int_{t_0}^{t_1}\ell(t, u(t), \psi(t))dt.
$$

Using JAX, one may automatically differentiate through the solver to compute $\frac{\partial J}{\partial \psi_0}$ and $\frac{\partial J}{\partial u(t)}$. 

qmax provides several adjoint methods that determine various details about how JAX differentiates 
through the solver such as checkpointing, i.e. the number of intermediate values to save, 
and the method to reconstruct unsaved residuals, allowing the user to control the tradeoff between computational speed and memory when differentiating. 

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
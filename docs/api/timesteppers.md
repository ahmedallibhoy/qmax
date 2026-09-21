# Timesteppers
Consider the Schrödinger equation 

$$
i\hbar\dot{\psi} = H(t)\psi
$$

with time-varying Hamiltonian $H(t)$. 
Solutions are $\psi(t) = U\big(t_0, t\big)\psi_0$ where $U(t_0, t)$ is the *propagator* of the system.
We describe here the timestepping method -- numerical methods to estimate $U\big(t, t + dt\big)\psi$ -- 
used by qmax,
which are all based on the *Magnus expansion* (see the following reference for details).

??? cite "References"
    1. S. Blanes, F. Casas, J. A. Oteo, and J. Ros, "The Magnus expansion and some of its applications," 
    *Phys. Rep.*, vol. 470, no. 5-6, pp. 151-238, 2009.

**The Magnus Expansion**: The propagator can be expressed analytically in terms of the *Magnus expansion*. Using the 
*ansatz* that $U(t_0, t_1) = \exp(\Omega(t_0, t_1))$ for some $\Omega(t_0, t_1)$,
we can show that $\Omega$ is given by the infinite series $\Omega(t_0, t_1) = \sum_{k=1}^{\infty}\Omega_k$, 
where the first three terms are

$$
\begin{aligned}
\Omega_1 &= -\frac{i}{\hbar}\int_{t_0}^{t_1}H(s)ds, \\
\Omega_2 &= -\frac{i}{2\hbar}\int_{t_0}^{t_1}\int_{t_0}^{s_1}[H(s_1), H(s_2)]ds_2ds_1, \\
\Omega_3 &= -\frac{i}{6\hbar}\int_{t_0}^{t_1}\int_{t_0}^{s_1}\int_{t_0}^{s_2}\big([H(s_1), [H(s_2), H(s_3)]] + [H(s_3), [H(s_2), H(s_1)]]\big)ds_3ds_2ds_1.
\end{aligned}
$$

In terms of the series, the propagator is $U(t_0, t_1) = \Pi_{k=0}^{\infty}\exp(\Omega_k)$. 

**Commutator-free Exponential Timesteppers**: The series expansion can be efficiently computed using 
a class of numerical methods called *commutator-free exponential timesteppers* (CFETs). As the name
implies, they accomplish this without directly forming the nested commutators. A CFET with $m$ stages 
and $n$ exponential evaluations is an approximation of the form

$$
\tilde{\Omega}_k = -\frac{i}{\hbar}\sum_{j=1}^{m}w_{kj}H(t + c_{j}dt)dt \qquad k=1, 2, \dots n,
$$

which yields the approximate propagator $\tilde{U}(t, t + dt) = \Pi_{k=0}^{n}\exp(\tilde{\Omega}_k)$. 
The CFET has order $r$ if the approximation error scales like $O(dt^{r + 1})$. 
Identifying the weights $w_{kj}$ so that the approximation achieves a given order involves 
solving a set of algebraic equations called *order conditions*. Modern schemes are derived using 
numerical optimization techniques, by minimizing the approximation error over the set of weights such that the 
order conditions hold.

**Geometric Properties**: Unlike conventional ODE integrators, the CFETs provided by qmax 
are *symmetric* (the method is reversible by taking a step of size $-dt$, i.e. $\tilde{U}(t, t + dt)^{-1} = \tilde{U}(t + dt, t)$), which enables efficient computation of adjoints to differentiate through 
the solver (see [Adjoints](adjoints.md) for details). Additionally, the CFETs *preserve the geometric structure of the exact solution*. In the context of quantum mechanics, the approximate propagator is guaranteed to be
unitary even in the presence of truncation errors, making these methods ideal for accurate long-time 
simulation of quantum systems. 


::: qmax.timestepper.Midpoint 
    options:
        members:
            -

::: qmax.timestepper.CFET_r4_e2 
    options:
        members:
            -

::: qmax.timestepper.CFET_r4_e3opt 
    options:
        members:
            -

::: qmax.timestepper.CFET_r4_e4opt 
    options:
        members:
            -

::: qmax.timestepper.CFET_r6_e4opt_cplx 
    options:
        members:
            -
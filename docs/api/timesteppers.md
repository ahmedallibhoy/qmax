# Timesteppers

Consider the Schrödinger equation $i\hbar\dot{\psi} = H(t)\psi$ with time-varying Hamiltonian $H(t)$. 
Solutions to the equation take the form $\psi(t) = U\big(t_0, t\big)\psi_0$ where $U(t_0, t)$ is 
called the *propagator* of the system.

The propagator can be expressed in closed form using the *Magnus expansion*
$U(t_0, t_1) = \exp(\Omega(t_0, t_1))$ where $\Omega(t_0, t_1)$ admits the series expansion

$$
\Omega(t_0, t_1) = -\frac{i}{\hbar}\int_{t_0}^{t_1}H(s)ds - \frac{i}{2\hbar}\int_{t_0}^{t_1}\int_{t_0}^{s_1}[H(s_1), H(s_2)]ds_2ds_1 + \cdots
$$

The series expansion can be efficiently computed using a class of numerical methods 
called *commutator-free exponential timesteppers* (CFETs). As the name implies, the method approximates 
the individual terms of the series without directly forming the nested commutators.

A CFET with $m$ stages and $n$ 
exponential evaluations is an approximation of the form

$$
\exp(\Omega(t, t + dt)) \approx \exp(\tilde{B}_n)\exp(\tilde{B}_{n-1}) \cdots \exp(\tilde{B}_1)
$$

where $\tilde{B}_k = -\frac{i}{\hbar}dt\sum_{j=1}^{m}w_{kj}H(t + c_{j}dt)$. The CFET has order $r$
if $\psi(t + dt) = \psi_1 + O(dt^{r + 1})$ where $\psi_1$ is the state produced by the approximate 
propagator. 

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
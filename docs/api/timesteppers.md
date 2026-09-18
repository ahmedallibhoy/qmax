# Timesteppers

Consider the Schrödinger equation $i\hbar\dot{\psi} = H(t)\psi$ with time-varying Hamiltonian $H(t)$. 
Solutions are $\psi(t) = U\big(t_0, t\big)\psi_0$ where the operator $U(t_0, t)$ is 
the *propagator* of the system.

The propagator can be expressed in closed form using the *Magnus expansion*
$U(t_0, t_1) = \exp(\Omega(t_0, t_1))$ where $\Omega(t_0, t_1)$ is given by the infinite series

$$
\Omega(t_0, t_1) = -\frac{i}{\hbar}\int_{t_0}^{t_1}H(s)ds - \frac{i}{2\hbar}\int_{t_0}^{t_1}\int_{t_0}^{s_1}[H(s_1), H(s_2)]ds_2ds_1 - \frac{i}{6\hbar}\int_{t_0}^{t_1}\int_{t_0}^{s_1}\int_{t_0}^{s_2}\big([H(s_1), [H(s_2), H(s_3)]] + [H(s_3), [H(s_2), H(s_1)]]\big)ds_3ds_2ds_1 + \cdots
$$

The series expansion can be efficiently computed using a class of numerical methods 
called *commutator-free exponential timesteppers* (CFETs). As the name implies, they accomplish
this without directly forming the nested commutators.

A CFET with $m$ stages and $n$ 
exponential evaluations is an approximation of the form

$$
\exp(\Omega(t, t + dt)) \approx \exp(\tilde{\Omega}_n)\exp(\tilde{\Omega}_{n-1}) \cdots \exp(\tilde{\Omega}_1)
$$

where $\tilde{\Omega}_k = -\frac{i}{\hbar}\sum_{j=1}^{m}w_{kj}H(t + c_{j}dt)dt$. The CFET has order $r$
if the approximation error scales like $O(dt^{r + 1})$. 
Identifying the weights $w_{kj}$ so that the approximation achieves a given order involves 
solving a set of algebraic equations called *order conditions*. Modern schemes are derived using 
numerical optimization techniques, by minimizing the approximation error over the set of weights such that the 
order conditions hold.

Unlike conventional ODE integrators, CFETs preserve the geometric structure of the 
exact solution. In the context of quantum mechanics, the approximate state is guaranteed to be
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
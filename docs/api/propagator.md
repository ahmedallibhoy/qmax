# Propagator 

Consider the Schrödinger equation

$$
i\hbar \dot{\psi}(t) = H(t, u(t))\psi(t).
$$

where $H(t, u)$ is a controlled Hamiltonian, and $u(t)$ is a control input. Given $\psi(t_0)=\psi_0$, 
the solution to the Schrödinger equation at time $t_1$ is $\psi(t_1) = U\big(t_0, t_1; u(\cdot)\big)\psi_0$, 
where $U\big(t_0, t_1; u(\cdot)\big)$ is the *propagator*. 


qmax numerically approximates $\psi(t_1)$ by subdividing the interval $[t_0, t_1]$ into $N$ equally sized
steps, then propagating the initial condition over the interval one step at a time,

$$
\begin{aligned}
&\tilde{\psi}_0 = \psi(t_0) \\
&\tilde{\psi}_{k + 1} = \tilde{U}(\tau_k, \tau_k + dt)\tilde{\psi}_k, \qquad \tau_k = t_0 + kdt \\
&\psi(t_1) \approx \tilde{\psi}_{N - 1}
\end{aligned}
$$

where for $k=1, \dots, N$, the step $\tilde{U}(\tau_k, \tau_k + dt)$ approximates the true propagator 
$U(\tau_k, \tau_k + dt)$ using a commutator-free exponential timestepper (CFET). 
See [Timesteppers](timesteppers.md) for details. 

::: qmax.Propagator 
    options: 
        members: 
            - __init__
            - propagate
            - count_stage
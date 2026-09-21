# Philosophy of qmax

Given the number of software packages available for simulating quantum mechanical systems, 
one might ask why the world needs another? While this is a fair question, my knee-jerk 
response would be that qmax started out mainly as an educational project for me to 
learn more about quantum mechanics and geometric numerical integrators, and is not intended to 
displace any of the industry standard solutions. However, I also think that qmax
earns its place within the larger scientific computing / comptuational 
physics ecosystem with a unique design philosophy that I will elaborate on here.
Though other libraries may incorporate certain parts of these concepts, 
to the best of my knowledge, qmax is the only to do all of these at once. 

### Compositionality of numerical methods

qmax provides numerical methods to solve the controlled and time-varying Schrödinger equation, 

$$
i\hbar \dot{\psi} = H(t, u(t))\psi
$$ 

Rather than relying on standard ODE integrators, qmax uses exponential timestepping methods,
where the solution $\psi(t)$ is expressed in terms of the exponential action $\psi \mapsto \exp(\tau H(t, u(t)))\psi$
evaluated at various quadrature points. Unlike other libraries for solving differential equations, 
the solver does not view the operator $H$ as a blackbox. Instead, the 
numerical routines for approximating the exponential actions of operators are built up and composed hierarchically.

In qmax, every operator has an `exponentiator`, which is an object that 
approximates the exponential action $\psi \mapsto \exp(\tau A)\psi$ of the operator $A$. For some 
operators, the exponential action is known in closed form (e.g. potential energies, or Pauli operators acting on 
two-level systems), and can be accessed using `ExactExponentiator`.  If a closed-form expression is unavailable, 
qmax provides a suite of methods to efficiently approximate the exponential action, 
including Krylov subspace methods, Chebyshev polynomial approximations, and scaled and truncated Taylor expansions.

The exponentiation methods factor through certain algebraic compositions, 
e.g. $(A + B)$ or $cA$. In the codebase, this is implemented in terms 
of *delegating exponentiators*, each of which act on a specific type of composite operator by delegating 
to the exponentiators of its children. The following table shows the delegating exponentiators provided by qmax.

| Composition | Node type | Delegating exponentiator | Computed exponential action | Exact or approximate? |
|---|---|---|---|---|
| $sI + cA$ | `ShiftScaleOperator` | `ShiftScaleExponentiator` | $\exp(h(sI + cA)) = e^{hs}\exp((hc)A)$ | exact |
| $A + B$ | `AddOperator` | [Split methods](api/exponentiators/split.md) | see documentation for [split methods](api/exponentiators/split.md) | approximate |
| $I \otimes \cdots A \cdots \otimes I$ | `LiftOperator` | `LiftExp` | $\exp(h(I \otimes \cdots A \cdots \otimes I)) = I \otimes \cdots \exp(hA) \cdots \otimes I$ | exact |
| $A \oplus B$ | `KroneckerSum` | `KroneckerSumExp` | $\exp(h(A\oplus B)) =\exp(hA) \otimes \exp(hB)$ | exact |

Since the appropriate delegating exponentiator is automatically assigned to an algebraic composition, very 
little effort is required on the part of the user to configure the exponential approximation of a complicated 
operator. Consider the following example of a particle on $\mathbb{R}^2$ with potential energy $V(x) = \|x\|^2$.

```python
import qmax as qx 

hilbert_space = qx.spaces.FiniteDifference(x0=(-10, -10), x1=(10, 10), num_steps=(500, 500))
L = hilbert_space.laplacian()
V = hilbert_space.potential_energy(lambda x: 0.5 * (x @ x))
H = -0.5 * L + V
```

Using the tree inspection utilties provided by qmax, we can view the entire expression tree of $H$ and the 
exponentiator assigned to each node. We see that the exponential action of the operator automatically 
reduces to the exponential actions of the 1D Laplacian, and the closed-form 
exponential action of $V$. 

```python
print(H.tree(show_exp=True))
```

```
(-0.5 * Laplacian + FiniteDifferencePotentialEnergy)     exponentiator=Strang()
├─-0.5 * Laplacian                                       exponentiator=ShiftScaleExponentiator()
| └─Laplacian                                            exponentiator=KroneckerSumExp()
|   ├─Laplacian1D(axis=0)                                exponentiator=Cayley()
|   └─Laplacian1D(axis=1)                                exponentiator=Cayley()
└─FiniteDifferencePotentialEnergy                        exponentiator=ExactExponentiator()
```

### Inspectability of numerical routines

### Geometric structure preserving integrators

### Matrix-free linear algebra

### Full compatibility with JAX transformations
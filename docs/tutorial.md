
# Tutorial

## Architecture of qmax

1. Hilbert Spaces
2. Operators 
3. Exponentiators
4. Propagator

## Hilbert Spaces

##### Tensor Products

##### Spatial Discretizations

## Operators 

##### Time-varying Operators

##### Controlled Operators

## Exponentiators


In qmax, every operator has an `exponentiator`, which is an object that 
approximates the exponential action $\psi \mapsto \exp(\tau A)\psi$ of the operator $A$. For some 
operators, the exponential action is known in closed form (e.g. potential energies, or Pauli operators acting on 
two-level systems), and can be accessed using `ExactExponentiator`.  If a closed-form expression is unavailable, 
qmax provides a suite of methods to efficiently approximate the exponential action, 
including Krylov subspace methods, Chebyshev polynomial approximations, and scaled and truncated Taylor expansions.

##### Factorization through the operator algebra

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

##### Split Methods

##### Composed Exponentiators

##### Adapting an Exponentiator to an Operator

## Propagator

##### Constructing a Propagator

##### Timesteppers

##### Adjoints

## Introspection
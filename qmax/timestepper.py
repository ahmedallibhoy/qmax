from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from jaxtyping import Array


def _gl_rule(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return jnp.array((x + 1) / 2), jnp.array(w / 2)  


class AbstractTimeStepper(eqx.Module):
    num_nodes: eqx.AbstractClassVar[int]
    weights: eqx.AbstractClassVar[Array]
    order: eqx.AbstractClassVar[int]

    @property
    def quad_rule(self) -> Array:
        return _gl_rule(self.num_nodes)


class Midpoint(AbstractTimeStepper):
    num_nodes: ClassVar[int] = 1
    weights:   ClassVar[Array] = jnp.array([[1.]])
    order:     ClassVar[int] = 2


class CFET_r4_e2(AbstractTimeStepper):
    """
    Fourth-order commutator-free exponential time-propagator (CFET).

        A fourth-order commutator-free exponential integrator for nonautonomous differential equations.
        Thalhammer, Mechthild. 
        SIAM Journal on Numerical Analysis 44.2 (2006): 851-864.
    """

    num_nodes: ClassVar[int] = 2
    order:     ClassVar[int] = 4
    weights:   ClassVar[Array] = jnp.array(
        [[0.25 + (3.0 ** 0.5) / 6, 0.25 - (3.0 ** 0.5) / 6],
         [0.25 - (3.0 ** 0.5) / 6, 0.25 + (3.0 ** 0.5) / 6]])


class CFET_r4_e3opt(AbstractTimeStepper):
    """
    Optimized 4th-order CFET with 3 exponentials

        High-order commutator-free exponential time-propagation of driven quantum systems 
        Alvermann, Andreas, and Holger Fehske. 
        Journal of Computational Physics 230.15 (2011): 5930-5956.
    """

    num_nodes: ClassVar[int] = 3
    order:     ClassVar[int] = 4
    weights:   ClassVar[Array] = jnp.array(
        [[ 0.3025568331880237 , -0.03333333333333335,  0.00577650014530972],
         [-0.03055555555555558,  0.5111111111111112 , -0.03055555555555558],
         [ 0.00577650014530972, -0.03333333333333335,  0.3025568331880237 ]])


class CFET_r4_e4opt(AbstractTimeStepper):
    """
    Optimized 4th-order CFET with real coefficients

        High-order commutator-free quasi-Magnus exponential integrators for non-autonomous linear evolution equations.
        Blanes, Sergio, Fernando Casas, and Mechthild Thalhammer.  
        Computer Physics Communications 220 (2017): 243-262.
    """

    num_nodes: ClassVar[int] = 3
    order:     ClassVar[int] = 4
    weights:   ClassVar[Array] = jnp.array(
        [[ 0.2463347584748155, -0.0469610812011527,  0.0119511881315244],
         [ 0.0622500005170514,  0.2691833034233750, -0.0427581693456134],
         [-0.0427581693456134,  0.2691833034233750,  0.0622500005170514],
         [ 0.0119511881315244, -0.0469610812011527,  0.2463347584748155]])


class CFET_r6_e4opt(AbstractTimeStepper):
    """
    Optimized 6th-order CFET with complex coefficients

        High-order commutator-free quasi-Magnus exponential integrators for non-autonomous linear evolution equations.
        Blanes, Sergio, Fernando Casas, and Mechthild Thalhammer.  
        Computer Physics Communications 220 (2017): 243-262.
    """

    num_nodes: ClassVar[int] = 3
    order:     ClassVar[int] = 6
    weights:   ClassVar[Array] = jnp.array(
        [[ 0.245985577298764294+0.038734389227164527j, -0.046806149832548937+0.012442141491185027j,  0.010894359342569201-0.004575808769067271j],
         [ 0.062868370946917202-0.048761268117765233j,  0.269028372054771159-0.012442141491185027j, -0.041970529810472921+0.014602687659667977j],
         [-0.041970529810472921+0.014602687659667977j,  0.269028372054771159-0.012442141491185027j,  0.062868370946917202-0.048761268117765233j],
         [ 0.010894359342569201-0.004575808769067271j, -0.046806149832548937+0.012442141491185027j,  0.245985577298764294+0.038734389227164527j]])


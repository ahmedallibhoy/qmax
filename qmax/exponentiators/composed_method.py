from __future__ import annotations

from typing import ClassVar, Optional, TYPE_CHECKING
from abc import abstractmethod
from functools import reduce

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, ScalarLike

from .._introspect import CountDict, Path
from .._internal import _update_field
from ..hilbert_space import AbstractState
from .base import Order, AbstractExponentiator, NotExponentiableError
from .split import AbstractSplitMethod

if TYPE_CHECKING:
    from ..operator import Operator


# TODO: NoExponentiableError on order checks should be replaced by warnings

__all__ = [
    "compose",
    "AbstractCompositionMethod",
    "Yoshida",
    "Suzuki",
    "Symmetric_r6_s7",
    "Symmetric_r6_s9",
    "Symmetric_r8_s15",
    "Symmetric_r8_s17",
    "Symmetric_r10_s31",
    "Symmetric_r10_s33",
    "Symmetric_r10_s35",
    "AbstractComposedExponentiator",
    "ComposedExponentiator",
    "ComposedSplitExponentiator",
]


class AbstractCompositionMethod(eqx.Module):

    def __check_init__(self):
        if not jnp.allclose(jnp.sum(self.weights), 1):
            raise ValueError("Composition weights must sum to 1")
        if not jnp.allclose(self.weights, self.weights[::-1]):
            raise ValueError(f"Composition weights must be a palindromic sequence")

    @property
    @abstractmethod
    def weights(self) -> Array:
        pass

    @property
    @abstractmethod
    def composed_order(self) -> int:
        pass


class Yoshida(AbstractCompositionMethod):
    """
    Yoshida triple-jump exponential splitting, see e.g. [1], and [2, Example 4.2]

        1. Yoshida, Haruo. "Construction of higher order symplectic integrators."
           Physics letters A 150.5-7 (1990): 262-268.

        2. Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations
           Hairer, Ernst and Lubich, Christian and Wanner, Gerhard. Springer-Verlag, 2006
    """

    composed_order = 4
    
    @property
    def weights(self) -> Array:
        w1 = 1 / (2 - 2 ** (1 / 3))
        w2 = 1 - 2 * w1
        return jnp.array([w1, w2, w1])


class Suzuki(AbstractCompositionMethod):
    """
    Suzuki quintuple-jump exponential splitting, see e.g. [1], and [2, Example 4.3]

        1. Suzuki, Masuo. "Fractal decomposition of exponential operators with applications 
           to many-body theories and Monte Carlo simulations." 
           Physics Letters A 146.6 (1990): 319-323.

        2. Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations
           Hairer, Ernst and Lubich, Christian and Wanner, Gerhard. Springer-Verlag, 2006
    """
    composed_order = 4

    @property
    def weights(self) -> Array:
        w1 = 1 / (4 - 4 ** (1 / 3))
        w2 = 1 - 4 * w1
        return jnp.array([w1, w1, w2, w1, w1])


class Symmetric_r6_s7(AbstractCompositionMethod):
    """
    6th order symmetric composition with 7 stages, see e.g. [1, Table 2] and [2, 3.7.1]  

        1. On the numerical integration of ordinary differential equations by symmetric
        composition methods. McLachlan, Robert I.
        SIAM Journal on Scientific Computing 16.1 (1995): 151-168.

        2. A Concise Introduction to Geometric Numerical Integration (2nd ed.).
        Blanes, Sergio, and Fernando Casas. Chapman and Hall/CRC. 2025.
    """
    composed_order = 6
    weights: ClassVar[Array] = jnp.array(
        [0.7845136104775573, 0.23557321335935813, -1.177679984178871,
         1.3151863206839112, -1.177679984178871, 0.23557321335935813,
         0.7845136104775573])


class Symmetric_r6_s9(AbstractCompositionMethod):
    """
    6th order symmetric composition with 9 stages, see e.g. [1, Table 2] and [2, 3.7.1]  

        1. On the numerical integration of ordinary differential equations by symmetric
        composition methods. McLachlan, Robert I.
        SIAM Journal on Scientific Computing 16.1 (1995): 151-168.

        2. A Concise Introduction to Geometric Numerical Integration (2nd ed.).
        Blanes, Sergio, and Fernando Casas. Chapman and Hall/CRC. 2025.
    """
    composed_order = 6
    weights: ClassVar[Array] = jnp.array(
        [0.1867, 0.5554970237124784, 0.12946694891347535,
         -0.8432656233877346, 0.9432033015235616, -0.8432656233877346,
         0.12946694891347535, 0.5554970237124784, 0.1867])


class Symmetric_r8_s15(AbstractCompositionMethod):
    """
    8th order symmetric composition with 15 stages, see e.g. [1, Table 2] and [2, 3.7.1]  

        1. On the numerical integration of ordinary differential equations by symmetric
        composition methods. McLachlan, Robert I.
        SIAM Journal on Scientific Computing 16.1 (1995): 151-168.

        2. A Concise Introduction to Geometric Numerical Integration (2nd ed.).
        Blanes, Sergio, and Fernando Casas. Chapman and Hall/CRC. 2025.
    """
    composed_order = 8
    weights: ClassVar[Array] = jnp.array(
        [0.741670364350613, -0.4091008258000316, 0.1907547102962384,
         -0.5738624711160822, 0.2990641813036559, 0.33462491824529816,
         0.3152930923967666, -0.7968879393529165, 0.3152930923967666,
         0.33462491824529816, 0.2990641813036559, -0.5738624711160822,
         0.1907547102962384, -0.4091008258000316, 0.741670364350613])


class Symmetric_r8_s17(AbstractCompositionMethod):
    """
    8th order symmetric composition with 17 stages, see e.g. [1, Table 2] and [2, 3.7.1]  

        1. On the numerical integration of ordinary differential equations by symmetric
        composition methods. McLachlan, Robert I.
        SIAM Journal on Scientific Computing 16.1 (1995): 151-168.

        2. A Concise Introduction to Geometric Numerical Integration (2nd ed.).
        Blanes, Sergio, and Fernando Casas. Chapman and Hall/CRC. 2025.
    """
    composed_order = 8
    weights: ClassVar[Array] = jnp.array(
        [0.12886597938144329, 0.581514087105251, -0.4101753714698501,
         0.18514693571658775, -0.4095523434208514, 0.14440594108001203,
         0.27833550039367966, 0.31495668391629483, -0.6269948254051341,
         0.31495668391629483, 0.27833550039367966, 0.14440594108001203,
         -0.4095523434208514, 0.18514693571658775, -0.4101753714698501,
         0.581514087105251, 0.12886597938144329])


class Symmetric_r10_s31(AbstractCompositionMethod):
    """
    10th order symmetric composition with 31 stages, see Section 4.4:

    "Derivation of symmetric composition constants for symmetric integrators."
    Sofroniou, Mark, and Giulia Spaletta. Optimization Methods and Software 20.4-5 (2005): 597-613.
    """
    composed_order = 10
    weights: ClassVar[Array] = jnp.array(
        [0.14998070054317051502516939497857, 0.091208635101489291996105121514462, 0.50623124887796194535266557555255,
         0.094789715925889154094231454089204, -0.19520875735034504160990960439871, -0.38816256756251756192331854792644,
         -0.27450555650873276528931810649505, 0.14264675556451861069659069043321, 0.067102518966825349346877396037809,
         -0.19643186370792190448674783323248, 0.29602854892160888804740587728740, 0.18917810251470701571585847859316,
         0.19394700133244324371285167850479, 0.10120067580762238380456506324802, -0.58186926782264021140090352527182,
         0.60772821879184217383575377417062, -0.58186926782264021140090352527182, 0.10120067580762238380456506324802,
         0.19394700133244324371285167850479, 0.18917810251470701571585847859316, 0.29602854892160888804740587728740,
         -0.19643186370792190448674783323248, 0.067102518966825349346877396037809, 0.14264675556451861069659069043321,
         -0.27450555650873276528931810649505, -0.38816256756251756192331854792644, -0.19520875735034504160990960439871,
         0.094789715925889154094231454089204, 0.50623124887796194535266557555255, 0.091208635101489291996105121514462,
         0.14998070054317051502516939497857])


class Symmetric_r10_s33(AbstractCompositionMethod):
    """
    10th order symmetric composition with 33 stages, see Section 4.4:

    "Derivation of symmetric composition constants for symmetric integrators."
    Sofroniou, Mark, and Giulia Spaletta. Optimization Methods and Software 20.4-5 (2005): 597-613.
    """
    composed_order = 10
    weights: ClassVar[Array] = jnp.array(
        [0.070711261586085399079302771810203, 0.090342080937267568345577914389234, 0.14103133297152486103524322594476,
         0.40206004554029767537357060971803, -0.30239722849131075165735249848238, -0.22462355658241460137093154363351,
         0.061496988956063121940380707068411, 0.11346847775740802675296685287062, 0.23654672241381781124636015203490,
         0.27211409645898977643699556260890, 0.076129418470277234386530906651024, -0.18543093454238185309165565783301,
         -0.46495110925607623804616342747217, 0.10423014962104084592532590279051, 0.13621181452383232935841998116651,
         -0.27010275720513252644976102610064, 0.48632639368142264147037913293721, -0.27010275720513252644976102610064,
         0.13621181452383232935841998116651, 0.10423014962104084592532590279051, -0.46495110925607623804616342747217,
         -0.18543093454238185309165565783301, 0.076129418470277234386530906651024, 0.27211409645898977643699556260890,
         0.23654672241381781124636015203490, 0.11346847775740802675296685287062, 0.061496988956063121940380707068411,
         -0.22462355658241460137093154363351, -0.30239722849131075165735249848238, 0.40206004554029767537357060971803,
         0.14103133297152486103524322594476, 0.090342080937267568345577914389234, 0.070711261586085399079302771810203])


class Symmetric_r10_s35(AbstractCompositionMethod):
    """
    10th order symmetric composition with 35 stages, see Section 4.4:

    "Derivation of symmetric composition constants for symmetric integrators."
    Sofroniou, Mark, and Giulia Spaletta. Optimization Methods and Software 20.4-5 (2005): 597-613.
    """
    composed_order = 10
    weights: ClassVar[Array] = jnp.array(
        [0.078795722521686419263907679337684, 0.31309610341510852776481247192647, 0.027918383235078066109520273275299,
         -0.22959284159390709415121339679655, 0.13096206107716486317465685927961, -0.26973340565451071434460973222411,
         0.074973343155891435666137105641410, 0.11199342399981020488957508073640, 0.36613344954622675119314812353150,
         -0.39910563013603589787862981058340, 0.10308739852747107731580277001372, 0.41143087395589023782070411897608,
         -0.0048663605831352617621956593099771, -0.39203335370863990644808193642610, 0.051942502962449647037182904015976,
         0.050665090759924496335874344156866, 0.049674370639729879054568800279461, 0.049317735759594537917680008339338,
         0.049674370639729879054568800279461, 0.050665090759924496335874344156866, 0.051942502962449647037182904015976,
         -0.39203335370863990644808193642610, -0.0048663605831352617621956593099771, 0.41143087395589023782070411897608,
         0.10308739852747107731580277001372, -0.39910563013603589787862981058340, 0.36613344954622675119314812353150,
         0.11199342399981020488957508073640, 0.074973343155891435666137105641410, -0.26973340565451071434460973222411,
         0.13096206107716486317465685927961, -0.22959284159390709415121339679655, 0.027918383235078066109520273275299,
         0.31309610341510852776481247192647, 0.078795722521686419263907679337684])


class AbstractComposedExponentiator(AbstractExponentiator):
    base_exp: AbstractExponentiator
    method: AbstractCompositionMethod

    def check_exponentiable(
        self, 
        op: Operator, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None):

        self.base_exp.check_exponentiable(op, parent_path, child_idx)

        base_effective_order = self.base_exp.effective_order(op)

        if not base_effective_order == 2:
            raise NotExponentiableError(
                f"Composition methods are only compatible with exponentiators of order 2 "
                f"but {self.base_exp} has effective_order={base_effective_order} when applied "
                f"to the operator {op.label}")


    @property
    def weights(self) -> Array:
        return self.method.weights

    @property
    def order(self) -> Order:
        return self.method.composed_order

    def effective_order(self, op: Operator) -> Order:
        return self.method.composed_order


class ComposedExponentiator(AbstractComposedExponentiator):

    def adapt(
        self,
        op: Operator,
        dt_max: ScalarLike) -> AbstractExponentiator:
        
        base_exp = self.base_exp.adapt(op, jnp.max(jnp.abs(self.weights)) * dt_max)
        return _update_field(self, "base_exp", base_exp)

    @property
    def operator_type(self) -> type[Operator]:
        return self.base_exp.operator_type

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        def update(exp_y, w):
            return self.base_exp.exp(op, w * h, exp_y), None

        y1 = self.base_exp.exp(op, self.weights[0] * h, y)
        exp_y, _ = jax.lax.scan(update, y1, self.weights[1:])
        return exp_y

    def count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None) -> CountDict:

        return reduce(lambda a, b: a | b, 
            [self.base_exp.count(op, w * h, parent_path, child_idx) for w in self.weights])


class ComposedSplitExponentiator(AbstractComposedExponentiator, AbstractSplitMethod):
    base_exp: AbstractSplitMethod
    _a: tuple[float, ...] = eqx.field(static=True)
    _b: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        base_exp: AbstractSplitMethod,
        method: AbstractCompositionMethod,
        *,
        nest_left: bool=True):

        self.base_exp = base_exp
        self.method = method

        a_sub = self.base_exp.a
        b_sub = self.base_exp.b
        a = [self.weights[0] * a_sub[0]]
        b = list(self.weights[0] * b_sub)

        for w, w_next in zip(self.weights[:-1], self.weights[1:]):
            a += list(w * a_sub[1:-1]) + [w * a_sub[-1] + w_next * a_sub[0]]
            b += list(w_next * b_sub)

        a += list(self.weights[-1] * a_sub[1:])

        self._a = tuple(float(c) for c in a)
        self._b = tuple(float(c) for c in b)
        self.nest_left = nest_left

    @property
    def a(self) -> Array:
        return jnp.array(self._a)

    @property
    def b(self) -> Array:
        return jnp.array(self._b)


def compose(
    base_exp: AbstractExponentiator,
    composition: AbstractCompositionMethod,
    nest_left: Optional[bool]=None) -> AbstractComposedExponentiator:

    if not base_exp.order == 2:
        raise NotExponentiableError(
            f"Composition methods only compatible with exponentiators of order 2 "
            f"but received exponentiator of order={base_exp.order}")

    if isinstance(base_exp, AbstractSplitMethod):
        if nest_left is None:
            nest_left = base_exp.nest_left
        return ComposedSplitExponentiator(base_exp, composition, nest_left=nest_left)

    return ComposedExponentiator(base_exp, composition)

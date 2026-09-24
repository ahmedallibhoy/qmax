from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Array, ArrayLike, Complex, Real

if TYPE_CHECKING:
    RealArrayLike = Array | np.ndarray
    ComplexArrayLike = Array | np.ndarray
    RealScalarLike = int | float | Array | np.ndarray
    ComplexScalarLike = int | float | complex | Array | np.ndarray
else:
    RealArrayLike = Real[ArrayLike, "..."]
    ComplexArrayLike = Complex[ArrayLike, "..."]
    RealScalarLike = Real[ArrayLike, ""]
    ComplexScalarLike = Complex[ArrayLike, ""]

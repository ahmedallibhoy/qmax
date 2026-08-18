from . import hilbert_space
from . import operator
from . import spaces
from . import timestepper
from . import eig
from . import exponentiators

from . import tensor

from .propagator import *
from .utils import *

# TODO (Operator refactor):
#   1. ScaleSquareExponentiator should not assume the operator is self-adjoint DONE
#
#   2. All operators should default to ScaleSquareExponentiator if possible DONE
#
#   3. is_hermitian and is_unitary properties should be removed from Operator and subclasses
#       a. AdjOperator should only return self if called on an instance of AdjOperator DONE
#       b. ShiftOperator and ScalarMulOperator should guard spectral_bounds by checking dtype of self.c DONE
#       c. Ensure that no other consumers of these properties remain DONE
#
#   4. AddOperator should default to StrangSplitting only if operands are exponentiable DONE
#       - This somewhat conflicts with 2, need to think this through... DONE
#
#   5. adj_action should be an abstract method of Operator  DONE
#
#   6. Add an AbstractHermitianOperator which delegates adj_action to action DONE
#       a. Identity and ZeroOperator should inherit DONE
#       b. AbstractPotentialEnergy, PseudoSpectralLaplacian, FiniteDifferenceLaplacian, and AbstractPauliOperator
#           should inherit DONE
#
#   7. TensorOperators should delegate adj_operator as appropriate DONE
#
#   8. Operator should point to instance of AbstractHilbertSpace. 
#       a. Replace domain checks with instance equality checks. 
#       b. Ensure this carries over to TensorOperators correctly
#

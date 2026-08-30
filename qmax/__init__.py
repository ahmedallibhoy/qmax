from . import hilbert_space
from . import operator
from . import spaces
from . import timestepper
from . import eig
from . import exponentiators
from . import plot
from . import tensor
from . import control

from .propagator import *
from .utils import *


# TODO: ROADMAP -- short / medium term
#   1. Time Propagation
#       a. ControlledPropagator should take proper drift + control form ... DONE
#       b. Propagator should compute cost functions 
#       c. explicit adjoints + checkpointing
#       d. utilities / helpers for optimal control 
#       e. dense solutions?
#
#   2. Introspection
#       a. exponentiators should report backward error estimates?
#       b. adapt() should accept tunable tolerances
#       c. validating interface and exp counts (see 6)
#       d. fix path / field semantics ... DONE
#       e. extending count functionality to propagators
#
#   3. Eigensolvers
#       a. Polynomial / rational filtering
#       b. More sophisticated spectrum slicing
#
#   4. Spaces and Spatial Discretizations
#       a. extending current spaces
#           i. operators and state constructors
#       b. new spaces
#           i. Non-rectangular spatial discretizations (e.g. spherical, cylindrical)
#           ii. Arbitrary meshes
#
#   5. Exponentiation
#       a. Processing 
#
#   6. Misc Utilities 
#       a. Wigner functions
#       b. better plotting / visualization?
#
#   7. Testing / Validation
#       a. Rewrite test suite, benchmarking and diagnostic scripts
#
#   8. Documentation :/
#       a. docstrings / makedocs    
#       b. worked examples
#       c. website
#

# TODO: ROADMAP -- long term
#   1. Nonlinearities + exponential integrators
#   2. Density matrices
#   3. Open quantum systems / dissipation / Lindbladians       
#   4. Dynamic low-rank approximations / tensor trains
#
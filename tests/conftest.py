import jax 
jax.config.update("jax_enable_x64", True)

import pytest 

RTOL = 1e-5
ATOL = 1e-8
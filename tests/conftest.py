
import jax

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


RTOL = 1e-5
ATOL = 1e-8

KEY = jax.random.key(0)

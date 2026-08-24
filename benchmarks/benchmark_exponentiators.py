import sys, os
sys.path.append("../")

import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike, PRNGKeyArray

import qmax as qx
from qmax.operator import Operator
from qmax.exponentiators import AbstractExponentiator


def test_exp(
    op: Operator, 
    exp: AbstractExponentiator, 
    dt_range: ArrayLike, 
    num_per_dt: int=1,
    key: PRNGKeyArray=jax.random.key(0)) -> tuple[list[float], list[float], list[Array]]:

    op = op.with_exponentiator(exp)
    hs = op.domain

    compile_times = []
    exc_times = []
    errors = []

    for dt in dt_range:
        y0s = hs.random(key, shape=(num_per_dt,))
        expm = jax.scipy.linalg.expm(-1j * dt * op.to_matrix())

        do_test = jax.jit(lambda y: op.exp(-1j * dt, y))

        compile_start = time.perf_counter()
        jax.block_until_ready(do_test(y0s))
        compile_time = time.perf_counter() - compile_start

        exc_start = time.perf_counter()
        y1s = jax.block_until_ready(do_test(y0s))
        exc_time = time.perf_counter() - exc_start

        y1s_exact = jax.vmap(lambda y_coeff: hs.from_coeffs(expm @ y_coeff))(y0s.coeffs)
        error = jnp.linalg.norm(y1s_exact.coeffs - y1s.coeffs, axis=1)

        compile_times.append(compile_time)
        exc_times.append(exc_time)
        errors.append(error)

    return compile_times, exc_times, errors


def implied_order(dt_range: ArrayLike, num_per_dt: int, errors: list[Array]):
    dt_all = jnp.full((samples_per, dt_range.shape[0]), dt_range).ravel(order="F")
    error_all = jnp.concatenate(errors)

    log_dt = jnp.log(dt_all)
    log_error = jnp.log(error_all)
    log_dt_aux = jnp.stack((log_dt, jnp.ones_like(log_dt)), axis=1)

    (order, offset), _, _, _ = jnp.linalg.lstsq(log_dt_aux, log_error)
    return order



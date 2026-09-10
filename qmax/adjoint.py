from typing import Callable, ClassVar, Optional, TYPE_CHECKING
from functools import partial

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike, Scalar, ScalarLike

from .hilbert_space import AbstractState

if TYPE_CHECKING:
    from .propagator import Propagator, SaveFunction, CostFunction


def _step(
    carry: tuple[ArrayLike, ScalarLike, ScalarLike], 
    u_next: ArrayLike, 
    u_quad: ArrayLike, 
    t_pair: tuple[ScalarLike, ScalarLike], 
    U: Propagator, 
    running_cost_fn: CostFunction, 
    dt: ScalarLike) -> tuple[Array, Scalar, Scalar]:
    
    y, cost, total = carry
    t, t_next = t_pair
    y_next = U.propagate_stage(t, dt, y, u_quad)
    cost_next = running_cost_fn(t_next, y_next, u_next)
    total_next = total + 0.5 * (cost + cost_next) * dt
    return y_next, cost_next, total_next


@eqx.filter_custom_vjp
def _propagate(
    vjp_args: tuple[AbstractState, ArrayLike, ArrayLike],
    U: Propagator,
    running_cost_fn: CostFunction,
    save_every: SaveFunction,
    save_fn: int,
    callback: Callable,
    *,
    outer_scan_fn=jax.lax.scan,
    inner_scan_fn=jax.lax.scan) -> tuple[AbstractState, Scalar, PyTree]:

    y0, us, u_quads = vjp_args
    ts, dt = U.ts, U.dt

    def step(carry, args):
        t_pair, u_pair, u_quad = args
        _, u_next = u_pair
        carry_next = _step(carry, u_next, u_quad, t_pair, U, running_cost_fn, dt)
        if callback is not None:
            callback()
        return carry_next, None

    def loop(carry, args):
        y, _, _ = carry
        (t_starts, _), (u_starts, _), _ = args
        carry, _ = inner_scan_fn(step, carry, args)
        return carry, save_fn(t_starts[0], y, u_starts[0])

    t0, t1 = ts[0], ts[-1]
    cost0 = running_cost_fn(t0, y0, us[0])

    num_blocks = (ts.shape[0] - 1) // save_every
    args = (
        (ts[:-1].reshape(-1, save_every), ts[1:].reshape(-1, save_every)),
        (us[:-1].reshape(num_blocks, save_every, -1),
         us[1:].reshape(num_blocks, save_every, -1)),
        u_quads.reshape(num_blocks, save_every, *u_quads.shape[1:]))

    (y1, _, running_cost), ys = outer_scan_fn(loop, (y0, cost0, 0.0), args)

    ys = jax.tree.map(
        lambda a, b: jnp.concatenate([a, jnp.asarray(b)[None]]),
        ys, save_fn(t1, y1, us[-1]))

    return y1, running_cost, ys


@_propagate.def_fwd
def _propagate_fwd(_, vjp_args, *args, **kwargs):
    y1, running_cost, ys = _propagate(vjp_args, *args, **kwargs)
    return (y1, running_cost, ys), (y1, running_cost)


@_propagate.def_bwd
def _propagate_bwd(res, grad_out, _, vjp_args, U, running_cost_fn, *args, **kwargs):
    y0, us, u_quads = vjp_args
    y1, total = res
    ts, dt = U.ts, U.dt

    g_y1, g_total, g_ys = grad_out

    if g_y1 is None:
        g_y1 = y1.hilbert_space.zeros_like(y1)

    if g_total is None:
        g_total = 0.0

    if jax.tree.leaves(g_ys):
        raise NotImplementedError("Save values of propagate are not backward differentiable")

    def bwd_step(carry, args):
        carry_next, g_carry_next = carry
        t_pair, u_pair, u_quad_next = args
        u, u_next = u_pair

        carry = _step(
            carry_next, u, u_quad_next[:, ::-1], t_pair[::-1], U, running_cost_fn, -dt)

        _, vjp = eqx.filter_vjp(
            lambda cr, un, uq: _step(cr, un, uq, t_pair, U, running_cost_fn, dt),
            carry, u_next, u_quad_next)

        g_carry, g_u_next, g_u_quad_next = vjp(g_carry_next)

        return (carry, g_carry), (g_u_next, g_u_quad_next)

    t0, t1 = ts[0], ts[-1]
    cost1 = running_cost_fn(t1, y1, us[-1])

    ((y0, _, _), (g_y0, g_cost0, _)), (g_us, g_u_quads) = jax.lax.scan(
        bwd_step,
        ((y1, cost1, total), (g_y1, 0.0, g_total)),
        ((ts[:-1], ts[1:]), (us[:-1], us[1:]), u_quads),
        reverse=True)

    _, vjp = eqx.filter_vjp(running_cost_fn, t0, y0, us[0])
    _, g_y0_step, g_u0 = vjp(g_cost0)

    g_y0 = g_y0 + g_y0_step
    g_us = jnp.concatenate([g_u0[None], g_us])
    return g_y0, g_us, g_u_quads


class AbstractAdjoint(eqx.Module):
    outer_scan_fn: eqx.AbstractClassVar[Callable]
    inner_scan_fn: eqx.AbstractClassVar[Callable]
    use_custom_vjp: eqx.AbstractClassVar[bool]

    @property
    def propagate_fn(self) -> Callable:
        if self.use_custom_vjp:
            return partial(
                _propagate, 
                outer_scan_fn=self.outer_scan_fn, 
                inner_scan_fn=self.inner_scan_fn)

        return partial(
            _propagate.fn,
            outer_scan_fn=self.outer_scan_fn,
            inner_scan_fn=self.inner_scan_fn)


class DirectAdjoint(AbstractAdjoint):
    """
    Uses plain lax.scan for looping and differentiates directly through it. Supports 
    both forward and reverse mode differentiation. Stores every residual so not suitable 
    for high-dimensional systems due to memory use. 
    """

    # Callables stored on a class are bound as methods and called 
    # with self as the first argument unless marked as static
    outer_scan_fn: ClassVar[Callable] = staticmethod(jax.lax.scan) 
    inner_scan_fn: ClassVar[Callable] = staticmethod(jax.lax.scan)
    use_custom_vjp: ClassVar[bool] = False


class ReversibleAdjoint(AbstractAdjoint):
    """
    Uses plain lax.scan for looping but implements a custom vjp rule which reconstructs 
    the trajectory by stepping backward through the solver. Similar speed to DirectAdjoint
    but memory of the adjoint scales like O(1). Only supports reverse-mode differentiation 
    and saved values cannot be differentiated. 
    """

    outer_scan_fn: ClassVar[Callable] = staticmethod(jax.lax.scan)
    inner_scan_fn: ClassVar[Callable] = staticmethod(jax.lax.scan)
    use_custom_vjp: ClassVar[bool] = True


class CheckpointedAdjoint(AbstractAdjoint):
    """
    Uses a checkpointed scan for looping. Memory scales as O(√num_steps) by default, 
    though the number of checkpoints saved by the outer and inner scans can be adjusted
    by setting outer_checkpoints and inner_checkpoints respectively. This method only 
    supports reverse-mode differentiation.
    """

    outer_checkpoints: Optional[int] = None
    inner_checkpoints: Optional[int] = None
    use_custom_vjp: ClassVar[bool] = False

    @property
    def outer_scan_fn(self) -> Callable:
        return partial(eqxi.scan, kind="checkpointed", checkpoints=self.outer_checkpoints)

    @property
    def inner_scan_fn(self) -> Callable:
        return partial(eqxi.scan, kind="checkpointed", checkpoints=self.inner_checkpoints)

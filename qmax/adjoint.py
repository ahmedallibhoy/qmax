from typing import Callable, ClassVar, Optional, TYPE_CHECKING
from functools import partial

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .propagator import Propagator


__all__ = [
    "AbstractAdjoint",
    "DirectAdjoint",
    "ReversibleAdjoint",
    "CheckpointedAdjoint",
]


def _zeros_like(tree):
    return jax.tree.map(jnp.zeros_like, eqx.filter(tree, eqx.is_inexact_array))


def _step(U, cost_fn, y, t_pair):
    t, t_next = t_pair
    y_next = U.propagate_stage(t, U.dt, y)
    cost_next = 0.5 * (cost_fn(t, y) + cost_fn(t_next, y_next)) * U.dt
    return y_next, jnp.asarray(cost_next)


@eqx.filter_custom_vjp
def _propagate(
    vjp_args, save_every, save_fn, callback, *,
    outer_scan_fn=jax.lax.scan, inner_scan_fn=jax.lax.scan):
    U, cost_fn, y0 = vjp_args

    def step(carry, t_pair):
        y, total = carry
        y_next, cost_next = _step(U, cost_fn, y, t_pair)
        total_next = total + cost_next
        if callback is not None:
            callback()
        return (y_next, total_next), None

    def loop(carry, t_pairs):
        y, _ = carry
        t_starts, _ = t_pairs
        carry, _ = inner_scan_fn(step, carry, t_pairs)
        return carry, save_fn(t_starts[0], y)

    ts = U.ts
    (y1, total_cost), ys = outer_scan_fn(loop, (y0, 0.0),
        (ts[:-1].reshape(-1, save_every), ts[1:].reshape(-1, save_every)))

    ys = jax.tree.map(
        lambda a, b: jnp.concatenate([a, jnp.asarray(b)[None]]),
        ys, save_fn(U.t1, y1))

    return y1, total_cost, ys


@_propagate.def_fwd
def _propagate_fwd(_, vjp_args, *args, **kwargs):
    y1, total_cost, ys = _propagate(vjp_args, *args, **kwargs)
    return (y1, total_cost, ys), y1


@_propagate.def_bwd
def _propagate_bwd(y1, grad_out, _, vjp_args, *args, **kwargs):
    U, cost_fn, y0 = vjp_args
    g_y1, g_total_cost, g_ys = grad_out

    if g_y1 is None:
        g_y1 = y1.hilbert_space.zeros_like(y1)

    if g_total_cost is None:
        g_total_cost = 0.0

    if jax.tree.leaves(g_ys):
        raise NotImplementedError("Save values of propagate are not backward differentiable")

    def bwd_step(carry, t_pair):
        y_next, g_y_next, g_U_next, g_cost_fn_next = carry
        _, t_next = t_pair

        # Reconstruct y from y_next by stepping backward through the method
        # If the integration method is symmetric, the reconstruction is exact
        y = U.propagate_stage(t_next, -U.dt, y_next)

        _, vjp = eqx.filter_vjp(lambda *args: _step(*args, t_pair), U, cost_fn, y)

        # The function step returns the values
        #
        #   y_next = Φ(U, consts, y)
        #   step_cost = ½·dt·(ℓ(t, y, consts) + ℓ(t_next, y_next, consts))
        #
        # where consts are constants closed over by cost_fn. If we define
        #
        #   λ̃ := g_y_next + ½·dt·g_total_cost·∇_yℓ(t_next, y_next),
        #
        # then tracing through one step of vjp, we have
        #
        #   g_U_step = (∂Φ/∂U)† λ̃
        #   g_cost_fn_step = ½·dt·[ ∂ℓ(t, y, consts)/∂consts + ∂ℓ(t_next, y_next, consts)/∂consts ]
        #   g_y = (∂Φ/∂y)†λ̃ + ½·dt·g_total_cost·∇_yℓ(t,y)
        #
        # The last expression is exactly the discrete adjoint dynamics.
        #
        # In the case where U is a propagator of the time-varying operator H = H0 + Σₖu_k(t)H_k,
        # and cost_fn closes over the inputs via ℓ(t, y) = ℓ(t, u(t), y), then the component
        # of g_U_step holding the control uₖ will approximately compute
        #
        #   dt·Σⱼvⱼ·(1/ħ)Im⟨λ(sⱼ), H_k y(sⱼ)⟩,
        #
        # where (sⱼ, vⱼ) are Gauss-Legendre quadrature corresponding to the timestepper. The
        # component of g_cost_fn_step holding the control uₖ will compute:
        #
        #   ½·dt·g_total_cost·[∂ℓ(t, uₖ(t), y(t))/∂uₖ + ∂ℓ(t_next, uₖ(t_next), y(t_next))/∂uₖ].
        #
        # Summing the previous quantities over all the iterations will approximate the cotangent of the cost
        # with respect to the inputs.

        g_U_step, g_cost_fn_step, g_y = vjp((g_y_next, g_total_cost))

        return (y, g_y,
            jax.tree.map(jnp.add, g_U_next, g_U_step),
            jax.tree.map(jnp.add, g_cost_fn_next, g_cost_fn_step)), None

    ts = U.ts
    (_, g_y0, g_U, g_cost_fn), _ = jax.lax.scan(
        bwd_step,
        (y1, g_y1, _zeros_like(U), _zeros_like(cost_fn)),
        (ts[:-1], ts[1:]), reverse=True)

    return (g_U, g_cost_fn, g_y0)


class AbstractAdjoint(eqx.Module):
    outer_scan_fn: eqx.AbstractClassVar[Callable]
    inner_scan_fn: eqx.AbstractClassVar[Callable]
    use_custom_vjp: eqx.AbstractClassVar[bool]

    @property
    def propagate_fn(self) -> Callable:
        if self.use_custom_vjp:
            return partial(_propagate,
                outer_scan_fn=self.outer_scan_fn,
                inner_scan_fn=self.inner_scan_fn)

        return partial(_propagate.fn,
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

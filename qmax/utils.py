from typing import Callable

import jax
import jax.numpy as jnp

from .hilbert_space import AbstractHilbertSpace, AbstractState


__all__ = ["zeros_like", "stack", "unstack", "over_batch"]

# TODO: consolidate with hilbert_space.py 

def zeros_like(y: AbstractState) -> AbstractState:
    return y.hilbert_space.from_coeffs(jnp.zeros_like(y.coeffs))


def stack(ys: list[AbstractState]) -> AbstractState:
    return ys[0].hilbert_space.from_coeffs(
        jnp.stack([y.coeffs for y in ys], axis=0))


def unstack(y: AbstractState) -> tuple[AbstractState]:
    if len(y.coeffs.shape) == 1:
        return (y,)
        
    return tuple(
        y.hilbert_space.from_coeffs(y.coeffs[idx]) 
        for idx in range(y.coeffs.shape[0]))


def over_batch(
    fn: Callable[[AbstractState], AbstractState], 
    y: AbstractState) -> AbstractState:

    space = y.hilbert_space
    flat = y.coeffs.reshape(-1, y.coeffs.shape[-1])
    out = jax.vmap(lambda c: fn(space.from_coeffs(c)).coeffs)(flat)
    return space.from_coeffs(out.reshape(*y.coeffs.shape[:-1], out.shape[-1]))

   
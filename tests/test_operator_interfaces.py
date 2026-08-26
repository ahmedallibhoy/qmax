
import pytest

from typing import Callable, TypeVar
from dataclasses import dataclass

import jax 
import jax.numpy as jnp

from jaxtyping import ScalarLike

import qmax as qx 
from qmax.operator import Operator
from qmax.spaces import FiniteDifference, PseudoSpectral, Qubits, TwoLevel, NLevel
from qmax.hilbert_space import AbstractState

from conftest import RTOL, ATOL

T = TypeVar("T")


def action_agrees(op: Operator, y: AbstractState) -> bool:
    X = op.to_matrix()
    return jnp.allclose(
        op.action(y).coeffs, X @ y.coeffs, 
        rtol=RTOL, atol=ATOL)


def adj_action_agrees(op: Operator, y: AbstractState) -> bool:
    X = op.to_matrix()
    return jnp.allclose(
        op.adj_action(y).coeffs, X.conj().T @ y.coeffs, 
        rtol=RTOL, atol=ATOL)


def solve_agrees(
    op: Operator, 
    b: AbstractState, 
    scale: ScalarLike, 
    shift: ScalarLike) -> bool:

    X = op.to_matrix()
    A = scale * X + shift * jnp.eye(op.domain.dim)

    return jnp.allclose(
        (op.solve(y, scale, shift)).coeffs, jnp.linalg.solve(A, y.coeffs), 
        rtol=RTOL, atol=ATOL)


def exp_action_agrees(op: Operator, h: ScalarLike, y: AbstractState) -> bool:
    X = op.to_matrix()
    expm = jax.scipy.linalg.expm(h * X)
    return jnp.allclose(
        op.exp_action(h, y).coeffs, expm @ y.coeffs, 
        rtol=RTOL, atol=ATOL)


def binary_composition_matrix_agrees(
    fns: tuple[Callable[[T, T], T]], A: Operator, B: Operator) -> bool:

    fn_mat, fn_op = fns

    return jnp.allclose(
        fn_mat(A.to_matrix(), B.to_matrix()), 
        fn_op(A, B).to_matrix(), rtol=RTOL, atol=ATOL)


def binary_composition_action_agrees(
    fns: tuple[Callable[[T, T], T]], A: Operator, B: Operator, y: AbstractState) -> bool:

    fn_mat, fn_op = fns

    return jnp.allclose(
        fn_mat(A.to_matrix(), B.to_matrix()) @ y.coeffs,
        fn_op(A, B).action(y).coeffs,
        rtol=RTOL, atol=ATOL)


def binary_composition_adj_action_agrees(
    fns: tuple[Callable[[T, T], T]], A: Operator, B: Operator, y: AbstractState) -> bool:

    fn_mat, fn_op = fns

    return jnp.allclose(
        fn_mat(A.to_matrix(), B.to_matrix()).conj().T @ y.coeffs,
        fn_op(A, B).adj_action(y).coeffs,
        rtol=RTOL, atol=ATOL)


def unary_composition_matrix_agrees(
    fns: tuple[Callable[[T], T]], A: Operator) -> bool:
    return jnp.allclose(
        fn(A.to_matrix()), fn(A).to_matrix(), rtol=RTOL, atol=ATOL)


def unary_composition_action_agrees(
    fns: tuple[Callable[[T], T]], A: Operator, y: AbstractState) -> bool:

    fn_mat, fn_op = fns

    return jnp.allclose(
        fn_mat(A.to_matrix()) @ y.coeffs, fn_op(A).action(y.coeffs), 
        rtol=RTOL, atol=ATOL)


def unary_composition_adj_action_agrees(
    fns: tuple[Callable[[T], T]], A: Operator, y: AbstractState) -> bool:

    fn_mat, fn_op = fns

    return jnp.allclose(
        fn_mat(A.to_matrix()).conj().T @ y.coeffs, 
        fn_op(A).adj_action(y.coeffs), rtol=RTOL, atol=ATOL)


def action_preserves_batch(A: Operator, y_batch: AbstractState, rank: int) -> bool:
    y1_batch = A.action(y_batch)
    if rank == 1:
        y1_vmap = jax.vmap(A.action)(y_batch)
    elif rank == 2:
        y1_vmap = jax.vmap(lambda yy: jax.vmap(A.action)(yy))(y_batch)
    
    return jnp.allclose(y1_batch, y1_vmap, rtol=RTOL, atol=ATOL)


def adj_action_preserves_batch(
    A: Operator, h: ScalarLike, y_batch: AbstractState, rank: int) -> bool:
    
    y1_batch = A.action(y_batch)
    if rank == 1:
        y1_vmap = jax.vmap(A.adj_action)(y_batch)
    elif rank == 2:
        y1_vmap = jax.vmap(lambda y: jax.vmap(A.adj_action)(y))(y_batch)
    
    return jnp.allclose(y1_batch, y1_vmap, rtol=RTOL, atol=ATOL)


def exp_action_preserves_batch(
    A: Operator, h: ScalarLike, y_batch: AbstractState, rank: int) -> bool:

    y1_batch = A.action(y_batch)
    if rank == 1:
        y1_vmap = jax.vmap(partial(A.exp_action, h))(y_batch)
    elif rank == 2:
        y1_vmap = jax.vmap(lambda y: jax.vmap(partial(A.exp_action, dt))(y))(y_batch)
    
    return jnp.allclose(y1_batch, y1_vmap, rtol=RTOL, atol=ATOL)



BINARY_COMPOSITIONS = [
    (lambda A, B: A + B, lambda A, B: A + B),
    (lambda A, B: A - B, lambda A, B: A - B),
    (lambda A, B: A @ B, lambda A, B: A @ B),
    (lambda A, B: 2 * A + 4 * B, lambda A, B: 2 * A + 4 * B), 
    (lambda A, B: A @ B @ A + jnp.eye(A.shape[0]), lambda A, B: A @ B @ A + 3)  
]


UNARY_COMPOSITIONS = [
    (lambda A: 2 * A, lambda A: 2 * A), 
    (lambda A: 1.5 * A + jnp.eye(A.shape[0]), lambda A: 1.5 * A + 1), 
    (lambda A: jnp.eye(A.shape[0]) + 1.5 * A, lambda A: 1 + 1.5 * A),
    (lambda A: A.conj().T, lambda A: A.H), 
    (lambda A: A + A.conj().T, lambda A: A + A.H), 
    (lambda A: A.conj().T @ A, lambda A: A.H @ A), 
    (lambda A: 1j * (A - A.conj().T), lambda A: 1j * (A - A.H))
]

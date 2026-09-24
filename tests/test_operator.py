
import jax
import jax.numpy as jnp
import pytest
from conftest import ATOL, KEY, RTOL
from helpers import OPERATOR_PAIRS, OPERATORS, SPACES

import qmax as qx


def action_agrees(op, y):
    X = op.to_matrix()
    return jnp.allclose(
        op.action(y).coeffs, X @ y.coeffs, 
        rtol=RTOL, atol=ATOL)


def adj_action_agrees(op, y):
    X = op.to_matrix()
    return jnp.allclose(
        op.adj_action(y).coeffs, X.conj().T @ y.coeffs, 
        rtol=RTOL, atol=ATOL)


def solve_agrees(op, b, scale, shift):
    X = op.to_matrix()
    A = scale * X + shift * jnp.eye(op.domain.dim)

    return jnp.allclose(
        (op.solve(b, scale, shift)).coeffs, jnp.linalg.solve(A, b.coeffs), 
        rtol=RTOL, atol=ATOL)


def exp_action_agrees(op, h, y):
    X = op.to_matrix()
    expm = jax.scipy.linalg.expm(h * X)
    return jnp.allclose(
        op.exp_action(h, y).coeffs, expm @ y.coeffs, 
        rtol=RTOL, atol=ATOL)


def unary_composition_matrix_agrees(fns, A):
    fn_mat, fn_op = fns

    return jnp.allclose(
        fn_mat(A.to_matrix()), fn_op(A).to_matrix(), rtol=RTOL, atol=ATOL)


def unary_composition_action_agrees(fns, A, y):
    fn_mat, fn_op = fns

    return jnp.allclose(
        fn_mat(A.to_matrix()) @ y.coeffs, fn_op(A).action(y).coeffs, 
        rtol=RTOL, atol=ATOL)


def unary_composition_adj_action_agrees(fns, A, y):
    fn_mat, fn_op = fns

    return jnp.allclose(
        fn_mat(A.to_matrix()).conj().T @ y.coeffs, 
        fn_op(A).adj_action(y).coeffs, rtol=RTOL, atol=ATOL)


def binary_composition_matrix_agrees(fns, A, B):
    fn_mat, fn_op = fns

    return jnp.allclose(
        fn_mat(A.to_matrix(), B.to_matrix()), 
        fn_op(A, B).to_matrix(), rtol=RTOL, atol=ATOL)


def binary_composition_action_agrees(fns, A, B, y):
    fn_mat, fn_op = fns

    return jnp.allclose(
        fn_mat(A.to_matrix(), B.to_matrix()) @ y.coeffs,
        fn_op(A, B).action(y).coeffs,
        rtol=RTOL, atol=ATOL)


def binary_composition_adj_action_agrees(fns, A, B, y):
    fn_mat, fn_op = fns

    return jnp.allclose(
        fn_mat(A.to_matrix(), B.to_matrix()).conj().T @ y.coeffs,
        fn_op(A, B).adj_action(y).coeffs,
        rtol=RTOL, atol=ATOL)


BINARY_COMPOSITIONS = {
    "sum": (lambda A, B: A + B, lambda A, B: A + B),
    "difference": (lambda A, B: A - B, lambda A, B: A - B),
    "matmul": (lambda A, B: A @ B, lambda A, B: A @ B),
    "linear_comb": (lambda A, B: 2 * A + 4 * B, lambda A, B: 2 * A + 4 * B), 
    "complex": (lambda A, B: A @ B @ A + jnp.eye(A.shape[0]), lambda A, B: A @ B @ A + 1)  
}


UNARY_COMPOSITIONS = {
    "scalar_mul": (lambda A: 2 * A, lambda A: 2 * A), 
    "shift_scale": (lambda A: 1.5 * A + jnp.eye(A.shape[0]), lambda A: 1.5 * A + 1), 
    "identity_plus_scale": (lambda A: jnp.eye(A.shape[0]) + 1.5 * A, lambda A: 1 + 1.5 * A),
    "conjugate_transpose": (lambda A: A.conj().T, lambda A: A.H), 
    "symmetrization": (lambda A: A + A.conj().T, lambda A: A + A.H), 
    "grammian": (lambda A: A.conj().T @ A, lambda A: A.H @ A), 
    "quadrature": (lambda A: 1j * (A - A.conj().T), lambda A: 1j * (A - A.H))
}


@pytest.mark.parametrize("op", OPERATORS.values(), ids=OPERATORS.keys())
def test_operator_consistent(op):
    y = op.domain.random(KEY)

    assert action_agrees(op, y)
    assert adj_action_agrees(op, y)

    if op.overrides_solve:
        assert solve_agrees(op, y, 1, 0.01)
        assert solve_agrees(op, y, -1, 0.01)
        assert solve_agrees(op, y, 1j, 0.01)
        assert solve_agrees(op, y, -1j, 0.01)

    if op.overrides_exp_action and op.tree_order is None:
        assert exp_action_agrees(op, 0.01, y)
        assert exp_action_agrees(op, -0.01, y)
        assert exp_action_agrees(op, 0.01j, y)
        assert exp_action_agrees(op, -0.01j, y)


@pytest.mark.parametrize("fns", UNARY_COMPOSITIONS.values(), ids=UNARY_COMPOSITIONS.keys())
@pytest.mark.parametrize("op", OPERATORS.values(), ids=OPERATORS.keys())
def test_operator_unary_composition(fns, op):
    y = op.domain.random(KEY)

    assert unary_composition_matrix_agrees(fns, op)
    assert unary_composition_action_agrees(fns, op, y)
    assert unary_composition_adj_action_agrees(fns, op, y)


@pytest.mark.parametrize("fns", BINARY_COMPOSITIONS.values(), ids=BINARY_COMPOSITIONS.keys())
@pytest.mark.parametrize("op_pair", OPERATOR_PAIRS.values(), ids=OPERATOR_PAIRS.keys())
def test_operator_binary_composition(fns, op_pair):
    A, B = op_pair
    y = A.domain.random(KEY)

    assert binary_composition_matrix_agrees(fns, A, B)
    assert binary_composition_action_agrees(fns, A, B, y)
    assert binary_composition_adj_action_agrees(fns, A, B, y)


fd = SPACES["finite_difference_1d"]
twolevel = SPACES["twolevel"]
nlevel = SPACES["nlevel"]

@pytest.mark.parametrize("A,B", [(fd.laplacian(), twolevel.pauli("x"))])
def test_reject_incomaptible_composition(A, B):
    with pytest.raises(qx.expression_tree.IncompatibleDomainError):
        A + B


@pytest.mark.parametrize("A,y", [(twolevel.pauli("x"), nlevel.random(KEY))])
def test_reject_incompatible_domain_on_action(A, y):
    with pytest.raises(qx.expression_tree.IncompatibleDomainError):
        A(y)


@pytest.mark.parametrize("A,y", [(twolevel.pauli("x"), nlevel.random(KEY))])
def test_reject_incompatible_domain_on_exp(A, y):
    with pytest.raises(qx.expression_tree.IncompatibleDomainError):
        A.exp(-1j, y)


@pytest.mark.parametrize("A,y", [(twolevel.pauli("x"), nlevel.random(KEY))])
def test_reject_incompatible_domain_on_solve(A, y):
    with pytest.raises(qx.expression_tree.IncompatibleDomainError):
        A.solve(y, shift=-0.1)

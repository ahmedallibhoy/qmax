import pytest 
from functools import reduce

import jax 
import jax.numpy as jnp 

import qmax as qx

from conftest import RTOL, ATOL, KEY
from helpers import SPACES
from test_operator import action_agrees, adj_action_agrees


fd = SPACES["finite_difference_1d"]
ps = SPACES["pseudospectral_1d"]
qubits = SPACES["qubits"]
nlevel = SPACES["nlevel"]

fd_times_ps = qx.tensor.TensorProduct([fd, ps])
qubits_times_nlevel = qx.tensor.TensorProduct([qubits, nlevel])
nlevel5 = qx.tensor.TensorPower(nlevel, 5)


TENSOR_PROD_CASES = {
    "fd times ps": (fd, ps), 
    "qubits times nlevel": (qubits, nlevel)
}

@pytest.mark.parametrize("space1,space2", TENSOR_PROD_CASES.values(), ids=TENSOR_PROD_CASES.keys())
def test_tensor_prod(space1, space2):
    tensor_space = qx.tensor.TensorProduct((space1, space2))

    y1 = space1.random(KEY)
    y2 = space2.random(KEY)

    y_prod = tensor_space.product_state((y1, y2))
    y_prod_coeffs = jnp.kron(y1.coeffs, y2.coeffs)
    y_prod_tensor = y_prod_coeffs.reshape((space1.dim, space2.dim))
    
    assert jnp.allclose(y_prod.coeffs, y_prod_coeffs, rtol=RTOL, atol=ATOL)

    assert jnp.allclose(
        tensor_space.from_tensor(y_prod_tensor).tensor, y_prod_tensor, 
        rtol=RTOL, atol=ATOL)


TENSOR_POWER_CASES = {
    "ps^2": (ps, 2), 
    "nlevel^5": (nlevel, 5)
}

@pytest.mark.parametrize("space,power", TENSOR_POWER_CASES.values(), ids=TENSOR_POWER_CASES.keys())
def test_tensor_power(space, power):
    tensor_space = qx.tensor.TensorPower(space, power)
    
    y = space.random(KEY)
    y_zero = space.zeros_like(y)
    z = jnp.zeros_like(y.coeffs)

    y_prod = tensor_space.product_state([y] + [y_zero] * power)
    y_prod_coeffs = reduce(lambda a, b: jnp.kron(a, b), [y.coeffs] + [z] * (power - 1))
    y_prod_tensor = y_prod_coeffs.reshape((space.dim for _ in range(power)))
    
    assert jnp.allclose(y_prod.coeffs, y_prod_coeffs, rtol=RTOL, atol=ATOL)

    assert jnp.allclose(
        tensor_space.from_tensor(y_prod_tensor).tensor, y_prod_tensor, 
        rtol=RTOL, atol=ATOL)


LIFT_CASES = {
    "Lift(ps_lapl, 1)": 
        (fd_times_ps, ps.laplacian(), 1), 
    "Lift(S_xS_yS_z, 0)": 
        (qubits_times_nlevel, qubits.pauli_product(["x", "y", "z"]), 0), 
    "Lift(annihilator, -1)":
        (nlevel5, nlevel.annihilator(), -1)
}

@pytest.mark.parametrize("tensor_space,A,lift_idx", LIFT_CASES.values(), ids=LIFT_CASES.keys())
def test_lift(tensor_space, A, lift_idx):
    Alift = tensor_space.lift(A, lift_idx)

    Is = [tensor_space[idx].identity() for idx in range(tensor_space.num_factors)]
    mat_list = [
        A.to_matrix() if idx == lift_idx % tensor_space.num_factors else I.to_matrix() 
        for (idx, I) in enumerate(Is)
    ]

    Amat = reduce(lambda a, b: jnp.kron(a, b), mat_list)
    assert jnp.allclose(Alift.to_matrix(), Amat, rtol=RTOL, atol=ATOL)

    y = tensor_space.random(KEY)
    assert action_agrees(Alift, y)
    assert adj_action_agrees(Alift, y)



A = nlevel.annihilator()
C = nlevel.creator()

KRON_SUM_CASES = {
    "KroneckerSum(fd_lapl, ps_lapl)": (fd_times_ps, [fd.laplacian(), ps.laplacian()]), 
    "KroneckerSum(a, c, a, c, a)": (nlevel5, [A, C, A, C, A])
}

@pytest.mark.parametrize("tensor_space,op_list", KRON_SUM_CASES.values(), ids=KRON_SUM_CASES.keys())
def test_kronecker_sum_on_prod(tensor_space, op_list):
    kron_op = tensor_space.kron_sum(op_list)
    
    Is = [tensor_space[idx].identity() for idx in range(tensor_space.num_factors)]
    matrix = jnp.zeros((tensor_space.dim, tensor_space.dim))

    for factor_idx, A in enumerate(op_list):
        mat_list = [
            A.to_matrix() if idx == factor_idx else I.to_matrix() 
            for (idx, I) in enumerate(Is)
        ]
        matrix += reduce(lambda a, b: jnp.kron(a, b), mat_list)

    assert jnp.allclose(kron_op.to_matrix(), matrix, rtol=RTOL, atol=ATOL)

    y = tensor_space.random(KEY)
    assert action_agrees(kron_op, y)
    assert adj_action_agrees(kron_op, y)


KRON_PROD_CASES = {
    "KroneckerProduct(fd_lapl, ps_lapl)": (fd_times_ps, [fd.laplacian(), ps.laplacian()]), 
    "KroneckerProduct(a, c, a, c, a)": (nlevel5, [A, C, A, C, A])
}

@pytest.mark.parametrize("tensor_space,op_list", KRON_PROD_CASES.values(), ids=KRON_PROD_CASES.keys())
def test_kronecker_prod(tensor_space, op_list):
    kron_prod = tensor_space.kron_prod(op_list)

    matrix = reduce(lambda a, b: jnp.kron(a, b), [op.to_matrix() for op in op_list])
    assert jnp.allclose(kron_prod.to_matrix(), matrix)

    y = tensor_space.random(KEY)
    assert action_agrees(kron_prod, y)
    assert adj_action_agrees(kron_prod, y)


@pytest.mark.parametrize("tensor_space,A,lift_idx", [(fd_times_ps, fd.laplacian(), 1)])
def test_lift_rejects_incompatible_domain(tensor_space, A, lift_idx):
    with pytest.raises(qx.expression_tree.IncompatibleDomainError):
        Alift = tensor_space.lift(A, lift_idx) 


@pytest.mark.parametrize("tensor_space,op_list", [(nlevel5, [A, A, A, A, fd.laplacian()])]) 
def test_kron_sum_rejects_incompatible_domains(tensor_space, op_list):
    with pytest.raises(qx.expression_tree.IncompatibleDomainError):
        op = tensor_space.kron_sum(op_list)


@pytest.mark.parametrize("tensor_space,op_list", [(nlevel5, [A, A, A, A])]) 
def test_kron_prod_rejects_incompatible_num_factors(tensor_space, op_list):
    with pytest.raises(ValueError):
        op = tensor_space.kron_sum(op_list)

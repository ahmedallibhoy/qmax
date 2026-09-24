import jax.numpy as jnp

import qmax as qx

SPACES = {
    "finite_difference_1d":  qx.FiniteDifference(
        -10, 10, num_steps=100),
    "finite_difference_2d":  qx.FiniteDifference(
        jnp.array([-10, -10]), jnp.array([10, 10]), num_steps=(50, 50)),
    "pseudospectral_1d": qx.PseudoSpectral(-10, 10, 100, 50),
    "pseudospectral_2d": qx.PseudoSpectral(
        jnp.array([-10, -10]), jnp.array([10, 10]), (50, 50), (25, 25)),
    "twolevel": qx.TwoLevel(),
    "qubits": qx.Qubits(3),
    "nlevel": qx.NLevel(5)
}


OPERATORS = {}
OPERATOR_PAIRS = {}

def append_operators(fn):
    global OPERATORS
    OPERATORS = OPERATORS | fn()
    return fn

def append_operator_pairs(fn):
    global OPERATOR_PAIRS
    OPERATOR_PAIRS = OPERATOR_PAIRS | fn()
    return fn


@append_operators
def finite_difference_operators():
    space_1d = SPACES["finite_difference_1d"]
    space_2d = SPACES["finite_difference_2d"]

    return {
        "fd_lapl_1d": space_1d.laplacian(),
        "fd_potential_1d": space_1d.potential_energy(lambda x: x ** 2),
        "fd_lapl_2d": space_2d.laplacian(),
        "fd_potential_2d": space_2d.potential_energy(lambda x: x @ x)
    }

@append_operator_pairs
def finite_difference_pairs():
    space_1d = SPACES["finite_difference_1d"]
    space_2d = SPACES["finite_difference_2d"]

    return {
        "fd_lapl_1d, fd_potential_1d": (
            -space_1d.laplacian(), space_1d.potential_energy(lambda x: x ** 2)),
        "fd_lapl_2d, fd_potential_2d": (
            -space_2d.laplacian(), space_2d.potential_energy(lambda x: x @ x))
    }

@append_operators
def pseudospectral_operators():
    space_1d = SPACES["pseudospectral_1d"]
    space_2d = SPACES["pseudospectral_2d"]

    return {
        "ps_lapl_1d": space_1d.laplacian(),
        "ps_potential_1d": space_1d.potential_energy(lambda x: x ** 2),
        "ps_lapl_2d": space_2d.laplacian(),
        "ps_potential_2d": space_2d.potential_energy(lambda x: x @ x)
    }

@append_operator_pairs
def pseudospectral_pairs():
    space_1d = SPACES["pseudospectral_1d"]
    space_2d = SPACES["pseudospectral_2d"]

    return {
        "ps_lapl_1d, ps_potential_1d":
            (-space_1d.laplacian(), space_1d.potential_energy(lambda x: x ** 2)),
        "ps_lapl_2d, ps_potential_2d":
            (-space_2d.laplacian(), space_2d.potential_energy(lambda x: x @ x))
    }


@append_operators
def qubit_operators():
    twolevel = SPACES["twolevel"]
    qubits = SPACES["qubits"]

    return {
        "I": twolevel.pauli("i"),
        "S_x": twolevel.pauli("x"),
        "S_y": twolevel.pauli("y"),
        "S_z": twolevel.pauli("z"),
        "TensorProd(S_x, S_y, S_z)": qubits.pauli_product(["x", "y", "z"]),
        "TensorProd(S_x, I, I)": qubits.pauli_product(["x", "i", "i"]),
        "TensorProd(I, S_x, I)": qubits.pauli_product(["i", "x", "i"]),
        "TensorProd(I, I, S_x)": qubits.pauli_product(["i", "i", "x"]),
    }


@append_operator_pairs
def qubit_pairs():
    twolevel = SPACES["twolevel"]
    qubits = SPACES["qubits"]

    return {
        "S_x, S_y": (twolevel.pauli("x"), twolevel.pauli("y")),
        "TensorProd(S_x, S_y, S_z), TensorProd(I, I, S_x)":
            (qubits.pauli_product(["x", "y", "z"]), qubits.pauli_product(["i", "i", "x"]))
    }


@append_operators
def nlevel_operators():
    space = SPACES["nlevel"]

    return {
        "annihilator": space.annihilator(),
        "creator": space.creator()
    }


@append_operator_pairs
def nlevel_pairs():
    space = SPACES["nlevel"]

    return {
        "annihilator, annihilator": (space.annihilator(), space.annihilator()),
        "ahnihilator, creator": (space.annihilator(), space.creator())
    }

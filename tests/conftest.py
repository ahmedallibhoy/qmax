"""Shared configuration, spaces, and numeric helpers for the qmax test suite.

This module runs before any test imports jax, so it is the single place where
the global jax configuration is pinned:

  * CPU backend, for reproducibility and to avoid requiring a GPU.
  * 64-bit precision, which parts of qmax assume (the Al-Mohy & Higham theta
    tables select the double-precision branch, and the Miller recurrence behind
    ChebyshevExponentiator overflows in single precision).

Everything here is layer-agnostic: the Hilbert spaces to test against, and the
handful of helpers for building states and comparing arrays. Anything specific
to one layer belongs in that layer's test module.

Currently excluded, by request: FiniteDifference and ScaleSquareExponentiator
(being refactored) and qmax.tensor_prod (under construction).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from qmax.spaces.nlevel import TwoLevel
from qmax.spaces.pseudospectral import PseudoSpectral

TWO_PI = 2 * float(jnp.pi)


# --------------------------------------------------------------------------
# Spaces
# --------------------------------------------------------------------------

PS_1D_FULL = PseudoSpectral(0.0, TWO_PI, 16, 16)

# num_modes < mesh_size: the mode grid is a strict subspace of the mesh grid, so
# pad/truncate are non-trivial and operators become compressions onto it.
PS_1D_DEALIASED = PseudoSpectral(0.0, TWO_PI, 32, 9)

# Domain length != 2*pi and an odd mode count (so there is no Nyquist mode).
# Keeps tests from conflating the mode index m with the wavenumber 2*pi*m/L,
# which coincide only on a 2*pi domain.
PS_1D_ODD = PseudoSpectral(-1.0, 1.0, 11, 11)

PS_2D_FULL = PseudoSpectral((0.0, 0.0), (TWO_PI, TWO_PI), (6, 6), (6, 6))
PS_2D_DEALIASED = PseudoSpectral((0.0, 0.0), (TWO_PI, TWO_PI), (8, 8), (5, 5))

QUBIT = TwoLevel()

SPACES = {
    "ps_1d_full": PS_1D_FULL,
    "ps_1d_dealiased": PS_1D_DEALIASED,
    "ps_1d_odd": PS_1D_ODD,
    "ps_2d_full": PS_2D_FULL,
    "ps_2d_dealiased": PS_2D_DEALIASED,
    "qubit": QUBIT,
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

# Tolerances for float64. Comparisons are between different algorithms, not the
# same arithmetic, so exact equality is not expected.
RTOL = 1e-9
ATOL = 1e-11


def random_state(space, key):
    """A random complex state on ``space``, normalized to unit 2-norm."""
    coeffs = jax.random.normal(key, (space.dim,), dtype=jnp.complex128)
    return space.from_coeffs(coeffs / jnp.linalg.norm(coeffs))


def dense(op, space):
    """An operator's own dense representation, as a numpy array."""
    return np.asarray(op.to_matrix(space), dtype=complex)


def assert_close(got, want, what, rtol=RTOL, atol=ATOL):
    __tracebackhide__ = True
    npt.assert_allclose(
        np.asarray(got), np.asarray(want), rtol=rtol, atol=atol, err_msg=what
    )


def true_bounds(op, space):
    """The tightest interval possible: [min, max] eigenvalue of the dense matrix.

    Uses eigvalsh, so it presumes a Hermitian operator -- which is the same
    assumption spectral_bounds itself makes by returning a real interval.
    """
    eigvals = np.linalg.eigvalsh(dense(op, space))
    return float(eigvals[0]), float(eigvals[-1])


@pytest.fixture
def key():
    return jax.random.key(0)

"""Batching transparency: leading batch axes must ride through untouched.

``AbstractState.coeffs`` has shape ``(*batch, dim)``. Every operation is required
to treat the leading axes as batch, so that

    op(space.from_coeffs(stacked)).coeffs  ==  stack([op(y_i).coeffs for y_i])

exactly -- not approximately. Both sides run the same arithmetic in the same
order; the only difference is how many states travel through at once. So a
failure here is a shape or axis bug, never a numerical one.

Two batch ranks are swept. Rank 1 alone would miss anything that indexes from the
front and happens to land correctly with a single leading axis, and rank 2 is
what a ``vmap`` over initial conditions stacked on a timestepper's time axis
actually produces.

Batch sizes are deliberately chosen to collide with factor dimensions on the
tensor spaces (batch 2 or 3 against factor dim 2), so a misplaced axis stays
shape-valid and only the values catch it.

Scope note: this file asks whether batching is transparent, not whether the
operators are mathematically right -- the unbatched result is the reference, so a
uniformly wrong operator passes. Correctness against dense matrices lives in
test_operator.py.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import pytest

from conftest import (
    PS_1D_DEALIASED,
    PS_1D_FULL,
    PS_2D_FULL,
    QUBIT,
    TWO_PI,
    assert_close,
)
from qmax.operator import Identity
from qmax.spaces.finite_difference import (
    FiniteDifference,
    FiniteDifference1D,
    FiniteDifference1DLaplacian,
    FiniteDifferenceLaplacian,
    FiniteDifferencePotentialEnergy,
)
from qmax.spaces.nlevel import PauliOperator
from qmax.spaces.pseudospectral import (
    PseudoSpectralLaplacian,
    PseudoSpectralPotentialEnergy,
)
from qmax.system import TimeInvariantSystem
from qmax.tensor import (
    BatchKroneckerProduct,
    BatchKroneckerSum,
    KroneckerProduct,
    KroneckerSum,
    LiftOperator,
    TensorPower,
    TensorProduct,
)
from qmax.timestepper import Midpoint
from qmax.utils import over_batch, stack, unstack, zeros_like

BATCH_SHAPES = [(3,), (2, 3)]

DT = 0.05

# solve() is asked for (shift * I + scale * A) y = b, so the shift has to sit
# outside the spectrum of every case or the system is singular and the iterative
# fallback stagnates rather than failing cleanly. The binding constraint is the
# three-qubit Kronecker sums, whose eigenvalues reach exactly +3.
SOLVE_SCALE, SOLVE_SHIFT = -1.0, 5.0


# --------------------------------------------------------------------------
# Spaces beyond the shared ones: the tensor and finite-difference layers
# --------------------------------------------------------------------------

FD_1D = FiniteDifference1D(0.0, 1.0, 8)
FD_2D = FiniteDifference((0.0, 0.0), (1.0, 1.0), (4, 6))

# Heterogeneous factors with distinct dims, so a factor index used as a raw
# array axis lands somewhere visibly wrong.
TENSOR_PRODUCT = TensorProduct((QUBIT, PS_1D_FULL))
TENSOR_POWER = TensorPower(QUBIT, 3)

PAULI_X = PauliOperator(axis="x")
# Y is the one Pauli with M != M.T, so it catches a transpose slip that X and Z
# would hide.
PAULI_Y = PauliOperator(axis="y")

LAPLACIAN = PseudoSpectralLaplacian()
V_1D = PseudoSpectralPotentialEnergy(lambda x: 2.0 + jnp.cos(x))
V_2D = PseudoSpectralPotentialEnergy(lambda x: 2.0 + jnp.cos(x[0]) + jnp.cos(x[1]))
V_FD = FiniteDifferencePotentialEnergy(lambda x: jnp.sum(x**2))


@dataclass(frozen=True, eq=False)
class BatchCase:
    """An operator, a space it is valid on, and which optional interfaces it has."""

    space: object
    op: object
    has_exp_action: bool = True


CASES = {
    # spaces layer
    "ps_1d-laplacian":        BatchCase(PS_1D_FULL, LAPLACIAN),
    "ps_1d-potential":        BatchCase(PS_1D_FULL, V_1D, has_exp_action=False),
    "ps_1d_dealiased-potent": BatchCase(PS_1D_DEALIASED, V_1D, has_exp_action=False),
    "ps_2d-laplacian":        BatchCase(PS_2D_FULL, LAPLACIAN),
    "ps_2d-potential":        BatchCase(PS_2D_FULL, V_2D, has_exp_action=False),
    "qubit-pauli_x":          BatchCase(QUBIT, PAULI_X),
    "qubit-pauli_y":          BatchCase(QUBIT, PAULI_Y),
    "fd_1d-laplacian":        BatchCase(FD_1D, FiniteDifference1DLaplacian(), has_exp_action=False),
    "fd_2d-laplacian":        BatchCase(FD_2D, FiniteDifferenceLaplacian(2), has_exp_action=False),
    "fd_2d-potential":        BatchCase(FD_2D, V_FD),

    # operator algebra: these only delegate, so they inherit their children
    "algebra-identity":       BatchCase(PS_1D_FULL, Identity()),
    "algebra-shift":          BatchCase(PS_1D_FULL, LAPLACIAN + 2.0),
    "algebra-scalar_mul":     BatchCase(PS_1D_FULL, 3.0 * LAPLACIAN),
    "algebra-add_strang":     BatchCase(PS_1D_FULL, LAPLACIAN + V_1D, has_exp_action=False),

    # tensor layer on a heterogeneous product
    "tp-lift_0":              BatchCase(TENSOR_PRODUCT, LiftOperator(PAULI_X, 0, 2)),
    "tp-lift_1":              BatchCase(TENSOR_PRODUCT, LiftOperator(LAPLACIAN, 1, 2)),
    "tp-kron_sum":            BatchCase(TENSOR_PRODUCT, KroneckerSum((PAULI_Y, LAPLACIAN))),
    "tp-kron_product":        BatchCase(TENSOR_PRODUCT, KroneckerProduct((PAULI_Y, LAPLACIAN)),
                                        has_exp_action=False),

    # tensor layer on a power, including the scanned dispatch
    "tpow-lift_2":            BatchCase(TENSOR_POWER, LiftOperator(PAULI_Y, 2, 3)),
    "tpow-kron_sum":          BatchCase(TENSOR_POWER, KroneckerSum((PAULI_X, PAULI_Y, PAULI_X))),
    "tpow-kron_product":      BatchCase(TENSOR_POWER, KroneckerProduct((PAULI_X, PAULI_Y, PAULI_X)),
                                        has_exp_action=False),
    "tpow-batch_kron_sum":    BatchCase(TENSOR_POWER, BatchKroneckerSum((PAULI_X, PAULI_Y, PAULI_X))),
    "tpow-batch_kron_prod":   BatchCase(TENSOR_POWER, BatchKroneckerProduct((PAULI_X, PAULI_Y, PAULI_X)),
                                        has_exp_action=False),
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def batched_coeffs(space, key, batch_shape):
    """Random complex coefficients of shape ``(*batch_shape, dim)``."""
    k1, k2 = jax.random.split(key)
    shape = (*batch_shape, space.dim)
    return jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)


def stacked_reference(state_fn, space, coeffs):
    """Apply ``state_fn`` to each batch element separately and restack."""
    flat = coeffs.reshape(-1, coeffs.shape[-1])
    out = jnp.stack([state_fn(space.from_coeffs(c)).coeffs for c in flat])
    return out.reshape(*coeffs.shape[:-1], out.shape[-1])


def assert_batch_transparent(state_fn, space, coeffs, what):
    __tracebackhide__ = True
    got = state_fn(space.from_coeffs(coeffs)).coeffs
    want = stacked_reference(state_fn, space, coeffs)
    assert got.shape == want.shape, f"{what}: shape {got.shape} != {want.shape}"
    assert_close(got, want, what)


@pytest.fixture(params=list(CASES), ids=list(CASES))
def case(request):
    return CASES[request.param]


@pytest.fixture(params=BATCH_SHAPES, ids=[str(s) for s in BATCH_SHAPES])
def batch_shape(request):
    return request.param


# --------------------------------------------------------------------------
# The operator interface
# --------------------------------------------------------------------------


class TestOperatorInterface:
    """Each entry point must be transparent to leading batch axes."""

    def test_action(self, case, batch_shape, key):
        coeffs = batched_coeffs(case.space, key, batch_shape)
        assert_batch_transparent(case.op, case.space, coeffs, "action")

    def test_exp_action(self, case, batch_shape, key):
        if not case.has_exp_action:
            pytest.skip("operator does not implement exp_action")
        coeffs = batched_coeffs(case.space, key, batch_shape)
        assert_batch_transparent(
            lambda y: case.op.exp_action(DT, y), case.space, coeffs, "exp_action"
        )

    def test_exp(self, case, batch_shape, key):
        coeffs = batched_coeffs(case.space, key, batch_shape)
        assert_batch_transparent(
            lambda y: case.op.exp(DT, y), case.space, coeffs, "exp"
        )

    def test_solve(self, case, batch_shape, key):
        coeffs = batched_coeffs(case.space, key, batch_shape)
        assert_batch_transparent(
            lambda b: case.op.solve(b, SOLVE_SCALE, SOLVE_SHIFT),
            case.space,
            coeffs,
            "solve",
        )


# --------------------------------------------------------------------------
# State and space plumbing
# --------------------------------------------------------------------------


class TestStatePlumbing:
    """Arithmetic, indexing, and the values/coeffs round trips."""

    def test_arithmetic(self, case, batch_shape, key):
        coeffs = batched_coeffs(case.space, key, batch_shape)
        for what, fn in (
            ("add", lambda y: y + y),
            ("scalar_mul", lambda y: 2.0 * y),
            ("neg", lambda y: -y),
            ("zeros_like", zeros_like),
        ):
            assert_batch_transparent(fn, case.space, coeffs, what)

    def test_getitem_indexes_batch_axes_only(self, key):
        """The coefficient axis must survive every form of index.

        A tuple index is the trap: ``coeffs[(0, 1), :]`` is read by numpy as
        fancy indexing over the first axis, which returns a plausibly shaped
        array rather than raising.
        """
        space = PS_1D_FULL
        coeffs = batched_coeffs(space, key, (2, 3))
        y = space.from_coeffs(coeffs)

        assert y[0].coeffs.shape == (3, space.dim)
        assert y[1:].coeffs.shape == (1, 3, space.dim)
        assert y[0, 1].coeffs.shape == (space.dim,)
        assert y[:, 1].coeffs.shape == (2, space.dim)

        assert_close(y[0, 1].coeffs, coeffs[0, 1], "getitem values")
        assert_close(y[:, 1].coeffs, coeffs[:, 1], "getitem values")

    @pytest.mark.parametrize(
        "space", [PS_1D_FULL, PS_2D_FULL, PS_1D_DEALIASED, FD_2D],
        ids=["ps_1d", "ps_2d", "ps_1d_dealiased", "fd_2d"],
    )
    def test_values_round_trip(self, space, batch_shape, key):
        coeffs = batched_coeffs(space, key, batch_shape)
        assert_batch_transparent(
            lambda y: y.hilbert_space.from_values(y.values), space, coeffs, "values"
        )

    @pytest.mark.parametrize(
        "space", [TENSOR_PRODUCT, TENSOR_POWER], ids=["product", "power"]
    )
    def test_coeff_tensor_round_trip(self, space, batch_shape, key):
        coeffs = batched_coeffs(space, key, batch_shape)
        assert_batch_transparent(
            lambda y: y.hilbert_space.from_tensor(y.coeff_tensor),
            space,
            coeffs,
            "coeff_tensor",
        )

    def test_product_state_broadcasts_mixed_ranks(self, key):
        """A batched factor and an unbatched one give a batch of product states."""
        k1, k2 = jax.random.split(key)
        qubit_batched = QUBIT.from_coeffs(
            jax.random.normal(k1, (2, QUBIT.dim), dtype=jnp.complex128)
        )
        ps_single = PS_1D_FULL.from_coeffs(
            jax.random.normal(k2, (PS_1D_FULL.dim,), dtype=jnp.complex128)
        )

        got = TENSOR_PRODUCT.product_state((qubit_batched, ps_single))
        want = jnp.stack([
            jnp.kron(qubit_batched.coeffs[i], ps_single.coeffs) for i in range(2)
        ])

        assert got.coeffs.shape == (2, TENSOR_PRODUCT.dim)
        assert_close(got.coeffs, want, "product_state")


# --------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------


class TestUtils:

    def test_stack_unstack_round_trip(self, key):
        keys = jax.random.split(key, 3)
        states = [
            PS_1D_FULL.from_coeffs(
                jax.random.normal(k, (PS_1D_FULL.dim,), dtype=jnp.complex128)
            )
            for k in keys
        ]

        stacked = stack(states)
        assert stacked.coeffs.shape == (3, PS_1D_FULL.dim)

        recovered = unstack(stacked)
        assert isinstance(recovered, tuple) and len(recovered) == 3
        for got, want in zip(recovered, states):
            assert_close(got.coeffs, want.coeffs, "stack/unstack")

    def test_stack_nests(self, key):
        states = [
            PS_1D_FULL.from_coeffs(
                jax.random.normal(k, (PS_1D_FULL.dim,), dtype=jnp.complex128)
            )
            for k in jax.random.split(key, 3)
        ]
        inner = stack(states)
        outer = stack([inner, inner])

        assert outer.coeffs.shape == (2, 3, PS_1D_FULL.dim)
        assert outer[1, 2].coeffs.shape == (PS_1D_FULL.dim,)

    def test_unstack_of_unbatched_is_a_singleton(self, key):
        y = PS_1D_FULL.from_coeffs(
            jax.random.normal(key, (PS_1D_FULL.dim,), dtype=jnp.complex128)
        )
        assert unstack(y) == (y,)

    def test_over_batch_matches_direct_application(self, batch_shape, key):
        """over_batch is the escape hatch for ops that are not rank-polymorphic."""
        space, op = FD_1D, FiniteDifference1DLaplacian()
        coeffs = batched_coeffs(space, key, batch_shape)
        fn = lambda b: op.solve(b, SOLVE_SCALE, SOLVE_SHIFT)

        got = over_batch(fn, space.from_coeffs(coeffs)).coeffs
        want = stacked_reference(fn, space, coeffs)

        assert got.shape == want.shape
        assert_close(got, want, "over_batch")


# --------------------------------------------------------------------------
# Time propagation
# --------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Orders do not match")
class TestPropagate:
    """The trajectory a timestepper returns is itself a batched state.

    The order-mismatch warning is expected here and irrelevant to batching: the
    potential's exponentiator is first order against Midpoint's second. Order
    consistency is checked in the system layer's own tests.
    """

    SPACE = PS_1D_FULL
    STEPS = 4

    def system(self):
        return TimeInvariantSystem(-0.5 * LAPLACIAN + V_1D)

    def propagate(self, coeffs):
        ys, _ = self.system().propagate(
            0.0, self.STEPS * DT, DT, self.SPACE.from_coeffs(coeffs), Midpoint()
        )
        return ys.coeffs

    def test_batched_y0_matches_per_element(self, key):
        """Time is prepended, so a pre-batched y0 gives (T+1, B, dim)."""
        coeffs = batched_coeffs(self.SPACE, key, (3,))
        got = self.propagate(coeffs)
        want = jnp.stack([self.propagate(c) for c in coeffs], axis=1)

        assert got.shape == (self.STEPS + 1, 3, self.SPACE.dim)
        assert_close(got, want, "propagate with batched y0")

    def test_vmap_over_initial_conditions(self, key):
        """vmap puts its axis in front instead, giving (B, T+1, dim)."""
        coeffs = batched_coeffs(self.SPACE, key, (3,))
        got = jax.vmap(self.propagate)(coeffs)
        want = jnp.stack([self.propagate(c) for c in coeffs])

        assert got.shape == (3, self.STEPS + 1, self.SPACE.dim)
        assert_close(got, want, "vmap(propagate)")

    def test_propagate_is_jittable(self, key):
        """check_consistency must not branch on a traced value."""
        coeffs = batched_coeffs(self.SPACE, key, ())
        jitted = jax.jit(self.propagate)
        assert_close(jitted(coeffs), self.propagate(coeffs), "jit(propagate)")

    def test_trajectory_is_a_usable_batched_state(self, key):
        """Observables over a trajectory go through the same batching path."""
        coeffs = batched_coeffs(self.SPACE, key, ())
        trajectory = self.SPACE.from_coeffs(self.propagate(coeffs))
        assert_batch_transparent(V_1D, self.SPACE, trajectory.coeffs, "op on trajectory")

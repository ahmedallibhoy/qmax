"""The operator layer: the Operator interface, the operator algebra, and the
error behaviour of incompatible states and incompatible compositions.

Every operator is required to expose ``to_matrix``, and that dense matrix is the
ground truth. The matrix-free interfaces must agree with the corresponding
numpy/scipy operation on it:

    action(y)               <->  M @ y
    exp_action(dt, y)       <->  scipy.linalg.expm(dt * M) @ y
    solve(b, scale, shift)  <->  numpy.linalg.solve(shift * I + scale * M, b)

The same checks are applied to composite operators, so ``2 * (a + b)`` is held
to exactly the same contract as a primitive.

CASES and PAIRS below are literal tables. Adding a row extends every sweep and
every algebra test with no change to a test body. The error tables are
deliberately not exhaustive: the domain check and the compatibility check are
each a single branch, so one representative case per mechanism exercises them.

Scope note: this file tests the operator *interface*, not whether a given
operator is the mathematically right one. Nothing here anchors a to_matrix
against an external reference -- anchoring the matrices belongs with the spaces 
layer.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from conftest import (
    PS_1D_DEALIASED,
    PS_1D_FULL,
    PS_1D_ODD,
    PS_2D_DEALIASED,
    PS_2D_FULL,
    QUBIT,
    SPACES,
    assert_close,
    dense,
    random_state,
    true_bounds,
)
from qmax.hilbert_space import AbstractHilbertSpace
from qmax.operator import AddOperator, Identity, Operator, ShiftOperator
from qmax.spaces.nlevel import PauliOperator
from qmax.spaces.pseudospectral import (
    PseudoSpectralLaplacian,
    PseudoSpectralPotentialEnergy,
)
from qmax.exponentiators import AbstractExponentiator
from qmax.split import AbstractSplitMethod, Strang


@dataclass(frozen=True, eq=False)
class OpCase:
    """An operator together with a space it is valid on."""

    space: AbstractHilbertSpace
    op: Operator
    # Whether the operator implements exp_action at all. PseudoSpectralPotential
    # Energy dropped its own, because on a dealiased space it was not the
    # exponential of to_matrix; it relies on PseudoSpectralExponentiator instead,
    # which carries the honest order of 1.
    has_exp_action: bool = True
    solve_cases: tuple = (
        (-1.0, 5.0),
        (-1.0, 5.0j),
        (-1.0, 5.0 + 5.0j),
    )


@dataclass(frozen=True, eq=False)
class PairCase:
    """Two operators on a common space, for the algebra tests."""

    space: AbstractHilbertSpace
    a: Operator
    b: Operator


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------

# An operator instance is not bound to a space, so one instance is shared by
# every space it is valid on.

IDENTITY = Identity()
LAPLACIAN = PseudoSpectralLaplacian()

# Potentials are kept bounded away from zero so that shift * I + scale * V stays
# well conditioned. "even" gives a real symmetric matrix; "skew" has an odd part,
# so the matrix is Hermitian but not symmetric; "broadband" is not bandlimited,
# so the mesh-grid product aliases and action must alias the same way to_matrix
# does.
V_1D_EVEN = PseudoSpectralPotentialEnergy(lambda x: 2.0 + jnp.cos(x))
V_1D_SKEW = PseudoSpectralPotentialEnergy(lambda x: 2.0 + jnp.cos(x) + 0.5 * jnp.sin(x))
V_1D_BROADBAND = PseudoSpectralPotentialEnergy(lambda x: 2.0 + jnp.exp(jnp.sin(x)))

V_2D_EVEN = PseudoSpectralPotentialEnergy(lambda x: 2.0 + jnp.cos(x[0]) + jnp.cos(x[1]))
V_2D_SKEW = PseudoSpectralPotentialEnergy(
    lambda x: 2.0 + jnp.cos(x[0]) + 0.5 * jnp.sin(x[1])
)
V_2D_BROADBAND = PseudoSpectralPotentialEnergy(
    lambda x: 2.0 + jnp.exp(jnp.sin(x[0]) * jnp.cos(x[1]))
)

PAULI_X = PauliOperator(axis="x")
PAULI_Y = PauliOperator(axis="y")
PAULI_Z = PauliOperator(axis="z")


# --------------------------------------------------------------------------
# Cases: every operator on every space it is valid on
# --------------------------------------------------------------------------

CASES = {
    "ps_1d_full-identity":            OpCase(PS_1D_FULL, IDENTITY),
    "ps_1d_full-laplacian":           OpCase(PS_1D_FULL, LAPLACIAN),
    "ps_1d_full-potential_even":      OpCase(PS_1D_FULL, V_1D_EVEN, has_exp_action=False),
    "ps_1d_full-potential_skew":      OpCase(PS_1D_FULL, V_1D_SKEW, has_exp_action=False),
    "ps_1d_full-potential_broadband": OpCase(PS_1D_FULL, V_1D_BROADBAND, has_exp_action=False),

    "ps_1d_odd-identity":             OpCase(PS_1D_ODD, IDENTITY),
    "ps_1d_odd-laplacian":            OpCase(PS_1D_ODD, LAPLACIAN),
    "ps_1d_odd-potential_even":       OpCase(PS_1D_ODD, V_1D_EVEN, has_exp_action=False),
    "ps_1d_odd-potential_skew":       OpCase(PS_1D_ODD, V_1D_SKEW, has_exp_action=False),
    "ps_1d_odd-potential_broadband":  OpCase(PS_1D_ODD, V_1D_BROADBAND, has_exp_action=False),

    "ps_2d_full-identity":            OpCase(PS_2D_FULL, IDENTITY),
    "ps_2d_full-laplacian":           OpCase(PS_2D_FULL, LAPLACIAN),
    "ps_2d_full-potential_even":      OpCase(PS_2D_FULL, V_2D_EVEN, has_exp_action=False),
    "ps_2d_full-potential_skew":      OpCase(PS_2D_FULL, V_2D_SKEW, has_exp_action=False),
    "ps_2d_full-potential_broadband": OpCase(PS_2D_FULL, V_2D_BROADBAND, has_exp_action=False),

    # On the dealiased spaces the mode grid is a strict subspace of the mesh
    # grid. The Laplacian is unaffected, being diagonal in the mode basis. The
    # potentials are compressions onto that subspace, which is why they no longer
    # claim an exp_action at all and why their solve falls through to lineax.
    "ps_1d_dealiased-identity":            OpCase(PS_1D_DEALIASED, IDENTITY),
    "ps_1d_dealiased-laplacian":           OpCase(PS_1D_DEALIASED, LAPLACIAN),
    "ps_1d_dealiased-potential_even":      OpCase(PS_1D_DEALIASED, V_1D_EVEN, has_exp_action=False),
    "ps_1d_dealiased-potential_skew":      OpCase(PS_1D_DEALIASED, V_1D_SKEW, has_exp_action=False),
    "ps_1d_dealiased-potential_broadband": OpCase(PS_1D_DEALIASED, V_1D_BROADBAND, has_exp_action=False),

    "ps_2d_dealiased-identity":            OpCase(PS_2D_DEALIASED, IDENTITY),
    "ps_2d_dealiased-laplacian":           OpCase(PS_2D_DEALIASED, LAPLACIAN),
    "ps_2d_dealiased-potential_even":      OpCase(PS_2D_DEALIASED, V_2D_EVEN, has_exp_action=False),
    "ps_2d_dealiased-potential_skew":      OpCase(PS_2D_DEALIASED, V_2D_SKEW, has_exp_action=False),
    "ps_2d_dealiased-potential_broadband": OpCase(PS_2D_DEALIASED, V_2D_BROADBAND, has_exp_action=False),

    "qubit-identity": OpCase(QUBIT, IDENTITY),
    "qubit-pauli_x":  OpCase(QUBIT, PAULI_X),
    "qubit-pauli_y":  OpCase(QUBIT, PAULI_Y),
    "qubit-pauli_z":  OpCase(QUBIT, PAULI_Z),

    # Shifted operators, op + c*I. Registered as ordinary cases so they get the
    # same sweep as anything else -- a ShiftOperator is just an operator. The
    # default solve_cases still apply: shifting by these amounts keeps every
    # spectrum below the real shift of 5.0. exp_action is inherited from the
    # wrapped operator, so a shifted potential has none either.
    "ps_1d_full-laplacian_shifted":         OpCase(PS_1D_FULL, LAPLACIAN + 2.0),
    "ps_1d_full-potential_shifted":         OpCase(PS_1D_FULL, V_1D_EVEN - 1.0, has_exp_action=False),
    "ps_1d_odd-laplacian_shifted":          OpCase(PS_1D_ODD, LAPLACIAN + 2.0),
    "ps_2d_full-potential_shifted":         OpCase(PS_2D_FULL, V_2D_EVEN + 0.5, has_exp_action=False),
    "ps_1d_dealiased-potential_shifted":    OpCase(PS_1D_DEALIASED, V_1D_EVEN + 2.0, has_exp_action=False),
    "qubit-pauli_x_shifted":                OpCase(QUBIT, PAULI_X + 3.0),
}

EXP_EXACT = {k: c for k, c in CASES.items() if c.has_exp_action}
NO_EXP_ACTION = {k: c for k, c in CASES.items() if not c.has_exp_action}
IDENTITY_CASES = {k: c for k, c in CASES.items() if isinstance(c.op, Identity)}


# The algebra tests are combinatorial -- pairs x expressions x interfaces -- so
# they use one representative pair per combination on spaces that differ in a way
# the algebra can see. The richer variants (every potential, every Pauli axis,
# the 2d spaces) are covered by the per-operator sweep in TestContracts.
# No identity operands here: a sum with a multiple of the identity collapses to a
# ShiftOperator, so it never reaches AddOperator. See TestShift.
PAIRS = {
    "ps_1d_full: laplacian + potential":      PairCase(PS_1D_FULL, LAPLACIAN, V_1D_EVEN),
    "ps_1d_full: potential + potential":      PairCase(PS_1D_FULL, V_1D_EVEN, V_1D_SKEW),

    "ps_1d_dealiased: laplacian + potential": PairCase(PS_1D_DEALIASED, LAPLACIAN, V_1D_EVEN),
    "ps_1d_dealiased: potential + potential": PairCase(PS_1D_DEALIASED, V_1D_EVEN, V_1D_SKEW),

    "qubit: pauli_x + pauli_z":               PairCase(QUBIT, PAULI_X, PAULI_Z),
    "qubit: pauli_x + pauli_y":               PairCase(QUBIT, PAULI_X, PAULI_Y),
}

# Every spelling that must collapse to the same ShiftOperator: op + 2*I. Covers a
# bare scalar and an explicit identity term, on either side, via + and -.
SHIFT_SPELLINGS = {
    "op + 2.0": lambda op: op + 2.0,
    "2.0 + op": lambda op: 2.0 + op,
    "op + 2.0 * Identity()": lambda op: op + 2.0 * Identity(),
    "2.0 * Identity() + op": lambda op: 2.0 * Identity() + op,
    "op - (-2.0)": lambda op: op - (-2.0),
    "op - (-2.0) * Identity()": lambda op: op - (-2.0) * Identity(),
    "(op + 0.5) + 1.5": lambda op: (op + 0.5) + 1.5,
}

# Shifts whose base operator, sign, or nesting differs from the table above.
SHIFT_FORMS = {
    "op + Identity()": (lambda op: op + Identity(), 1.0),
    "Identity() + op": (lambda op: Identity() + op, 1.0),
    "op - Identity()": (lambda op: op - Identity(), -1.0),
    "op - 2.0": (lambda op: op - 2.0, -2.0),
}

# (operator, a space outside its domain)
FOREIGN_STATE_CASES = {
    "laplacian <- qubit": (LAPLACIAN, QUBIT),
    "potential <- qubit": (V_1D_EVEN, QUBIT),
    "pauli_x <- ps_1d_full": (PAULI_X, PS_1D_FULL),
}

# operator pairs whose domains are disjoint, so composing them must raise
INCOMPATIBLE_PAIRS = {
    "laplacian + pauli_x": (LAPLACIAN, PAULI_X),
    "potential + pauli_z": (V_1D_EVEN, PAULI_Z),
}


# --------------------------------------------------------------------------
# Contract checks
# --------------------------------------------------------------------------

SCALARS = [2.5, -1.0, 0.5j]

# dt values for exponentiation. Real dt exercises the decaying/growing case;
# imaginary dt is how propagation actually uses these operators
# (dt -> -1j * dt / hbar), where the exponential is unitary.

# Solve cases for derived operators -- scalar multiples and composites -- whose
# spectrum is not the one the OpCase was written against. Only the shifts with a
# non-zero imaginary part are kept, since those are non-singular whatever the
# spectrum turns out to be. A real shift needs a known spectrum to dodge, which
# is exactly what OpCase.solve_cases is for.


def check_action(op, space, key):
    """action(y) == to_matrix(space) @ y."""
    __tracebackhide__ = True
    y = random_state(space, key)
    assert_close(op(y).coeffs, dense(op, space) @ np.asarray(y.coeffs), "action")


DTS = (0.25, -0.4j)

def check_exp_action(op, space, key):
    """exp_action(dt, y) == expm(dt * to_matrix(space)) @ y."""
    __tracebackhide__ = True
    M = dense(op, space)
    y = random_state(space, key)
    coeffs = np.asarray(y.coeffs)

    for dt in DTS:
        want = scipy.linalg.expm(dt * M) @ coeffs
        assert_close(op.exp_action(dt, y).coeffs, want, f"exp_action(dt={dt})")


DERIVED_SOLVE_CASES = ((-1.0, 5.0j), (-1.0, 5.0 + 5.0j))

def check_solve(op, space, key, solve_cases=DERIVED_SOLVE_CASES):
    """solve(b, scale, shift) == numpy.linalg.solve(shift * I + scale * M, b).

    Cases come from the OpCase for a primitive operator. Composites keep the
    default, since their spectrum is not predictable from the operands'.
    """
    __tracebackhide__ = True
    M = dense(op, space)
    eye = np.eye(space.dim)
    b = random_state(space, key)
    coeffs = np.asarray(b.coeffs)

    for scale, shift in solve_cases:
        want = np.linalg.solve(shift * eye + scale * M, coeffs)
        got = op.solve(b, scale, shift).coeffs
        assert_close(got, want, f"solve(scale={scale}, shift={shift})")

def check_all(op, space, key, *, exp_action=True, solve_cases=DERIVED_SOLVE_CASES):
    """Every applicable interface against the dense ground truth."""
    __tracebackhide__ = True
    check_action(op, space, key)
    if exp_action:
        check_exp_action(op, space, key)
    check_solve(op, space, key, solve_cases)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class TestContracts:
    """Every registered operator against its own dense matrix."""

    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_action(self, case, key):
        check_action(case.op, case.space, key)

    @pytest.mark.parametrize("case", EXP_EXACT.values(), ids=list(EXP_EXACT))
    def test_exp_action(self, case, key):
        check_exp_action(case.op, case.space, key)

    @pytest.mark.parametrize("case", NO_EXP_ACTION.values(), ids=list(NO_EXP_ACTION))
    def test_exp_action_not_implemented(self, case, key):
        """An operator with no closed form must raise rather than return a wrong
        answer -- its exponentiator supplies the approximation, and carries the
        order that goes with it."""
        with pytest.raises(NotImplementedError):
            case.op.exp_action(-0.1j, random_state(case.space, key))

    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_solve(self, case, key):
        """Covers the overridden diagonal solves, the Pauli matrix solve, and
        the generic lineax fallback in Operator.solve, which is what the
        potential operators use."""
        check_solve(case.op, case.space, key, case.solve_cases)


class TestIdentity:

    @pytest.mark.parametrize("case", IDENTITY_CASES.values(), ids=list(IDENTITY_CASES))
    def test_to_matrix(self, case):
        assert_close(case.op.to_matrix(case.space), np.eye(case.space.dim), "identity")

    @pytest.mark.parametrize("case", IDENTITY_CASES.values(), ids=list(IDENTITY_CASES))
    def test_action_is_the_same_state(self, case, key):
        y = random_state(case.space, key)
        assert case.op(y) is y

    @pytest.mark.parametrize("space", SPACES.values(), ids=list(SPACES))
    def test_accepts_every_space(self, space, key):
        """Identity's domain is AbstractHilbertSpace, so it is unrestricted and
        has no foreign space to reject."""
        Identity()(random_state(space, key))


class TestScalarMul:

    @pytest.mark.parametrize("c", SCALARS)
    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_to_matrix(self, case, c):
        assert_close(
            (c * case.op).to_matrix(case.space),
            c * dense(case.op, case.space),
            "scaled matrix",
        )

    @pytest.mark.parametrize("c", SCALARS)
    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_interfaces(self, case, c, key):
        check_all(c * case.op, case.space, key, exp_action=case.has_exp_action)

    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_scalar_folds_into_dt(self, case, key):
        """ScalarMulOperator.exp delegates as op.exp(c * dt, y) rather than
        exponentiating a scaled operator. Holds whether or not the underlying
        exp_action is exact, since both sides take the same path."""
        y = random_state(case.space, key)
        c, dt = 2.5, -0.3j
        assert_close(
            (c * case.op).exp(dt, y).coeffs,
            case.op.exp(c * dt, y).coeffs,
            "scalar folded into dt",
        )

    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_nested_scaling(self, case, key):
        assert_close(
            (2.0 * (3.0 * case.op)).to_matrix(case.space),
            6.0 * dense(case.op, case.space),
            "nested scaling",
        )
        check_action(2.0 * (3.0 * case.op), case.space, key)

    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_negation_and_division(self, case, key):
        M = dense(case.op, case.space)
        assert_close((-case.op).to_matrix(case.space), -M, "negation")
        assert_close((case.op / 4.0).to_matrix(case.space), M / 4.0, "division")
        check_action(-case.op, case.space, key)

    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_preserves_domain(self, case):
        assert (2.0 * case.op).domain is case.op.domain


class TestAdd:

    @pytest.mark.parametrize("pair", PAIRS.values(), ids=list(PAIRS))
    def test_to_matrix(self, pair):
        assert_close(
            (pair.a + pair.b).to_matrix(pair.space),
            dense(pair.a, pair.space) + dense(pair.b, pair.space),
            "sum matrix",
        )

    @pytest.mark.parametrize("pair", PAIRS.values(), ids=list(PAIRS))
    def test_interfaces(self, pair, key):
        """exp_action is excluded: AddOperator does not implement it, and its
        exp goes through the splitting method instead. How well that split
        approximates the exponential of the sum belongs to the split layer."""
        check_all(pair.a + pair.b, pair.space, key, exp_action=False)

    @pytest.mark.parametrize("pair", PAIRS.values(), ids=list(PAIRS))
    def test_exp_action_not_implemented(self, pair, key):
        y = random_state(pair.space, key)
        with pytest.raises(NotImplementedError):
            (pair.a + pair.b).exp_action(-0.1j, y)

    @pytest.mark.parametrize("pair", PAIRS.values(), ids=list(PAIRS))
    def test_subtraction(self, pair, key):
        assert_close(
            (pair.a - pair.b).to_matrix(pair.space),
            dense(pair.a, pair.space) - dense(pair.b, pair.space),
            "difference matrix",
        )
        check_action(pair.a - pair.b, pair.space, key)


class TestShift:
    """op + c*I collapses to a ShiftOperator, which factors the shift out of the
    exponential: exp(dt*(A + cI)) == exp(dt*c) * exp(dt*A). That keeps a shifted
    operator exponentiable exactly when its base is, rather than making a sum
    conditionally exponentiable."""

    BASES = {
        "ps_1d_full-laplacian": (PS_1D_FULL, LAPLACIAN),
        #"ps_1d_full-potential": (PS_1D_FULL, V_1D_EVEN),
        "qubit-pauli_x": (QUBIT, PAULI_X),
    }

    @pytest.mark.parametrize("spelling", SHIFT_SPELLINGS.values(), ids=list(SHIFT_SPELLINGS))
    @pytest.mark.parametrize("base", BASES.values(), ids=list(BASES))
    def test_spellings_collapse_to_the_same_shift(self, base, spelling):
        """Every way of writing op + 2*I gives one ShiftOperator with c == 2,
        never a nested one and never an AddOperator holding an identity term."""
        space, op = base
        shifted = spelling(op)

        assert isinstance(shifted, ShiftOperator)
        assert not isinstance(shifted.op, ShiftOperator), "shift did not collapse"
        assert_close(shifted.c, 2.0, "collapsed coefficient")
        assert_close(
            shifted.to_matrix(space),
            dense(op, space) + 2.0 * np.eye(space.dim),
            "shifted matrix",
        )

    @pytest.mark.parametrize("form,c", SHIFT_FORMS.values(), ids=list(SHIFT_FORMS))
    @pytest.mark.parametrize("base", BASES.values(), ids=list(BASES))
    def test_other_forms(self, base, form, c):
        space, op = base
        shifted = form(op)

        assert isinstance(shifted, ShiftOperator)
        assert_close(shifted.c, c, "coefficient")
        assert_close(
            shifted.to_matrix(space),
            dense(op, space) + c * np.eye(space.dim),
            "shifted matrix",
        )

    @pytest.mark.parametrize("base", BASES.values(), ids=list(BASES))
    def test_scalar_minus_operator_negates_the_base(self, base, key):
        """2.0 - A is (-A) + 2I, so the base is negated and the shift is not."""
        space, op = base
        shifted = 2.0 - op

        assert isinstance(shifted, ShiftOperator)
        assert_close(shifted.c, 2.0, "coefficient")
        want = 2.0 * np.eye(space.dim) - dense(op, space)
        assert_close(shifted.to_matrix(space), want, "2 - A")
        check_action(shifted, space, key)

    @pytest.mark.parametrize("base", BASES.values(), ids=list(BASES))
    def test_shift_never_produces_an_add_operator(self, base):
        """The collapse is what lets AddOperator stay unconditionally
        non-exponentiable, so guard it directly."""
        space, op = base
        for build in list(SHIFT_SPELLINGS.values()) + [f for f, _ in SHIFT_FORMS.values()]:
            assert not isinstance(build(op), AddOperator)

    @pytest.mark.parametrize("base", BASES.values(), ids=list(BASES))
    def test_inherits_exponentiator_and_order(self, base):
        """A shift is exact, so it must not degrade the base operator's order the
        way a Strang split would."""
        space, op = base
        shifted = op + 2.0
        assert shifted.exponentiator is op.exponentiator
        assert shifted.exp_order == op.exp_order

    @pytest.mark.parametrize("base", BASES.values(), ids=list(BASES))
    def test_solve_folds_the_shift_into_the_base(self, base, key):
        """solve delegates to the base operator with an adjusted shift, so it
        reuses whatever specialised solve the base has rather than the generic
        lineax fallback."""
        space, op = base
        shifted = op + 2.0
        b = random_state(space, key)

        scale, shift = -1.0, 5.0
        got = shifted.solve(b, scale, shift).coeffs
        want = op.solve(b, scale, shift + scale * 2.0).coeffs
        assert_close(got, want, "shift folded into base solve")

    @pytest.mark.parametrize("base", BASES.values(), ids=list(BASES))
    def test_preserves_domain(self, base):
        space, op = base
        assert (op + 2.0).domain is op.domain

    @pytest.mark.parametrize("pair", PAIRS.values(), ids=list(PAIRS))
    def test_shifted_sum_is_exponentiable(self, pair, key):
        """A shift over a *sum* must still exponentiate. The shift factors out and
        the base keeps its own exponentiator, so ShiftOperator.exp has to delegate
        rather than inherit Operator.exp -- otherwise the split method receives
        the ShiftOperator and looks for op1/op2 on it."""
        space = pair.space
        base = pair.a + pair.b
        dt = -0.3j

        for shifted in (base + 2.0, 2.0 + base):
            assert isinstance(shifted, ShiftOperator)
            y = random_state(space, key)
            got = shifted.exp(dt, y).coeffs
            want = (np.exp(dt * 2.0) * base.exp(dt, y)).coeffs
            assert_close(got, want, "shift factored out of a split")


# --------------------------------------------------------------------------
# Exponentiator delegation
# --------------------------------------------------------------------------

# What matters about a base is the kind of exponentiator it carries.
# ExactExponentiator only needs exp_action, so a missing delegation is invisible;
# a split method reads op1/op2 off whatever it is handed, so it is not.
DELEGATION_BASES = {
    "laplacian (exact)":  (PS_1D_FULL, LAPLACIAN),
    "ps sum (strang)":    (PS_1D_FULL, LAPLACIAN + V_1D_EVEN),
    "qubit sum (strang)": (QUBIT, PAULI_X + PAULI_Z),
}

# Sums, kept as their operands so a test can rebuild them with a different
# exponentiator installed.
DELEGATION_SUMS = {
    "ps sum":    (PS_1D_FULL, LAPLACIAN, V_1D_EVEN),
    "qubit sum": (QUBIT, PAULI_X, PAULI_Z),
}

# wrap(base).exp(dt, y) must equal expected(base, dt, y): the wrapper folds its
# own scalar into dt or out as a prefactor, and hands the *base* to the base's
# exponentiator.
DELEGATIONS = {
    "2.5 * base": (lambda b: 2.5 * b, lambda b, dt, y: b.exp(2.5 * dt, y)),
    "base * 2.5": (lambda b: b * 2.5, lambda b, dt, y: b.exp(2.5 * dt, y)),
    "-base":      (lambda b: -b,      lambda b, dt, y: b.exp(-dt, y)),
    "base / 2.0": (lambda b: b / 2.0, lambda b, dt, y: b.exp(dt / 2.0, y)),
    "base + 2.0": (lambda b: b + 2.0, lambda b, dt, y: np.exp(dt * 2.0) * b.exp(dt, y)),
    "2.0 + base": (lambda b: 2.0 + b, lambda b, dt, y: np.exp(dt * 2.0) * b.exp(dt, y)),
    "base - 2.0": (lambda b: b - 2.0, lambda b, dt, y: np.exp(-dt * 2.0) * b.exp(dt, y)),
    "2.0 - base": (lambda b: 2.0 - b, lambda b, dt, y: np.exp(dt * 2.0) * b.exp(-dt, y)),
}


class _RequiresAddOperator(AbstractSplitMethod):
    """Stands in for a real split method by reading op1/op2 off the operator it
    is handed, so it fails loudly if a wrapper passes itself instead of the sum
    its exponentiator belongs to."""

    def exp(self, op, dt, y):
        assert isinstance(op, AddOperator), (
            f"exponentiator was handed a {type(op).__name__} instead of the "
            f"AddOperator it belongs to; the wrapper exposed .exponentiator "
            f"without delegating .exp"
        )
        return op.op1.exp(dt / 2, op.op2.exp(dt, op.op1.exp(dt / 2, y)))

    @property
    def order(self):
        return 2
        
    @property
    def dt_scales(self):
        return 0.5, 1.0


class TestExponentiatorDelegation:
    """A wrapper that exposes the wrapped operator's exponentiator must also
    delegate exp to it.

    Inheriting Operator.exp calls self.exponentiator.exp(self, ...), which hands
    the *wrapper* to an exponentiator belonging to the wrapped operator. With
    ExactExponentiator that happens to work, since it only calls exp_action. With
    a split method it does not: it reads op1/op2 and finds neither. ShiftOperator
    had exactly this bug, so (A + B) + 2.0 raised AttributeError on exp.
    """

    @pytest.mark.parametrize("wrap,expected", DELEGATIONS.values(), ids=list(DELEGATIONS))
    @pytest.mark.parametrize("base", DELEGATION_BASES.values(), ids=list(DELEGATION_BASES))
    def test_exp_matches_the_delegated_form(self, base, wrap, expected, key):
        space, op = base
        y = random_state(space, key)
        dt = -0.3j
        assert_close(
            wrap(op).exp(dt, y).coeffs, expected(op, dt, y).coeffs, "delegated exp"
        )

    @pytest.mark.parametrize("wrap,expected", DELEGATIONS.values(), ids=list(DELEGATIONS))
    @pytest.mark.parametrize("base", DELEGATION_BASES.values(), ids=list(DELEGATION_BASES))
    def test_wrapper_exposes_the_base_exponentiator(self, base, wrap, expected):
        """The premise of the whole issue: the wrapper reports an exponentiator
        that is not its own."""
        space, op = base
        assert wrap(op).exponentiator is op.exponentiator

    @pytest.mark.parametrize("wrap,expected", DELEGATIONS.values(), ids=list(DELEGATIONS))
    @pytest.mark.parametrize("base", DELEGATION_SUMS.values(), ids=list(DELEGATION_SUMS))
    def test_split_exponentiator_receives_the_sum(self, base, wrap, expected, key):
        """Directly pins the invariant rather than inferring it from a value:
        install an exponentiator that requires the AddOperator interface and
        confirm the wrapper hands it the sum, not itself."""
        space, a, b = base
        op = (a + b).with_exponentiator(_RequiresAddOperator())
        wrap(op).exp(-0.3j, random_state(space, key))

    @pytest.mark.parametrize("base", DELEGATION_SUMS.values(), ids=list(DELEGATION_SUMS))
    def test_nested_wrappers(self, base, key):
        """Scaling a shifted sum and shifting a scaled sum are different
        operators, and both have to reach the sum's exponentiator."""
        space, a, b = base
        op = a + b
        y = random_state(space, key)
        dt = -0.3j

        # 2.5 * (op + 2I) -> the shift is scaled too
        assert_close(
            (2.5 * (op + 2.0)).exp(dt, y).coeffs,
            (np.exp(2.5 * dt * 2.0) * op.exp(2.5 * dt, y)).coeffs,
            "scaled shifted sum",
        )
        # (2.5 * op) + 2I -> the shift is not
        assert_close(
            ((2.5 * op) + 2.0).exp(dt, y).coeffs,
            (np.exp(dt * 2.0) * op.exp(2.5 * dt, y)).coeffs,
            "shifted scaled sum",
        )


# spectral_bounds returns a real interval, so it only applies under real scaling.
REAL_SCALARS = [2.5, -1.0, 0.5]

# Real-scalar composites, for checking that a bound stays sound as it propagates.
BOUND_COMPOSITES = {
    "a + b": lambda a, b: a + b,
    "a - b": lambda a, b: a - b,
    "-(a + b)": lambda a, b: -(a + b),
    "2.5 * (a + b)": lambda a, b: 2.5 * (a + b),
    "(a + b) + 2.0": lambda a, b: (a + b) + 2.0,
    "2.5 * a + 0.5 * b": lambda a, b: 2.5 * a + 0.5 * b,
}


def assert_contains_spectrum(op, space, what):
    """spectral_bounds must be a superset of the true spectrum, never a subset.

    Only the direction is checked here. A subset silently breaks its consumers:
    ScaleSquareExponentiator picks (m, s) assuming every eigenvalue lies within
    the interval, and Chebyshev's expansion diverges outside [-1, 1] after
    scaling. A superset only costs work. How *loose* the bound is is a separate,
    quantitative question -- see tests/diagnose_spectral_bounds.py.
    """
    __tracebackhide__ = True
    lo, hi = np.asarray(op.spectral_bounds(space), dtype=float)
    lo_true, hi_true = true_bounds(op, space)

    tol = 1e-9 * max(1.0, abs(lo_true), abs(hi_true))
    assert lo <= lo_true + tol, (
        f"{what}: lower bound {lo} is above the true minimum {lo_true}"
    )
    assert hi >= hi_true - tol, (
        f"{what}: upper bound {hi} is below the true maximum {hi_true}"
    )


class TestSpectralBounds:
    """spectral_bounds must overestimate, and must keep overestimating as it
    propagates through the algebra."""

    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_contains_spectrum(self, case):
        assert_contains_spectrum(case.op, case.space, "primitive")

    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_is_ordered(self, case):
        lo, hi = np.asarray(case.op.spectral_bounds(case.space), dtype=float)
        assert lo <= hi, f"bounds are inverted: [{lo}, {hi}]"

    @pytest.mark.parametrize("c", REAL_SCALARS)
    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_scalar_multiple_stays_sound(self, case, c):
        """Negative c flips the interval, which is why ScalarMulOperator sorts
        rather than assuming an order."""
        assert_contains_spectrum(c * case.op, case.space, f"{c} * op")

    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_shift_stays_sound(self, case):
        assert_contains_spectrum(case.op + 2.0, case.space, "op + 2.0")
        assert_contains_spectrum(case.op - 2.0, case.space, "op - 2.0")

    @pytest.mark.parametrize(
        "build", BOUND_COMPOSITES.values(), ids=list(BOUND_COMPOSITES)
    )
    @pytest.mark.parametrize("pair", PAIRS.values(), ids=list(PAIRS))
    def test_composite_stays_sound(self, pair, build, key):
        """AddOperator adds the two intervals, which is conservative by Weyl's
        inequalities but can be loose. Soundness is what matters here."""
        assert_contains_spectrum(build(pair.a, pair.b), pair.space, "composite")

    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_rejects_complex_scaling(self, case):
        """c * A and A + cI are not Hermitian for complex c, so a real interval
        is meaningless and must be refused rather than silently returned."""
        with pytest.raises(NotImplementedError, match="real spectrum"):
            (0.5j * case.op).spectral_bounds(case.space)
        with pytest.raises(NotImplementedError, match="real spectrum"):
            (case.op + 0.5j).spectral_bounds(case.space)

    @pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
    def test_traced_scalar_is_jit_safe(self, case):
        """The sign of c is not known at trace time, so ScalarMulOperator must
        order the interval without branching on it."""
        bounds = jax.jit(lambda c: (c * case.op).spectral_bounds(case.space))(-2.0)
        lo, hi = np.asarray(bounds, dtype=float)
        assert lo <= hi, f"bounds are inverted under jit: [{lo}, {hi}]"


class TestComposition:
    """Interfaces must survive arbitrary nesting of the allowed operations."""

    EXPRESSIONS = [
        ("a + b", lambda a, b: a + b, lambda A, B: A + B),
        ("a - b", lambda a, b: a - b, lambda A, B: A - B),
        ("-(a + b)", lambda a, b: -(a + b), lambda A, B: -(A + B)),
        ("2 * (a + b)", lambda a, b: 2.0 * (a + b), lambda A, B: 2.0 * (A + B)),
        ("(a + b) / 2", lambda a, b: (a + b) / 2.0, lambda A, B: (A + B) / 2.0),
        (
            "3 * a + 2 * b",
            lambda a, b: 3.0 * a + 2.0 * b,
            lambda A, B: 3.0 * A + 2.0 * B,
        ),
        ("a + b + a", lambda a, b: a + b + a, lambda A, B: A + B + A),
        ("a - 2 * b", lambda a, b: a - 2.0 * b, lambda A, B: A - 2.0 * B),
        ("-0.5 * a + b", lambda a, b: -0.5 * a + b, lambda A, B: -0.5 * A + B),
    ]

    EXPR_IDS = [e[0] for e in EXPRESSIONS]

    @pytest.mark.parametrize("label,build,build_dense", EXPRESSIONS, ids=EXPR_IDS)
    @pytest.mark.parametrize("pair", PAIRS.values(), ids=list(PAIRS))
    def test_to_matrix_composes(self, pair, label, build, build_dense):
        op = build(pair.a, pair.b)
        want = build_dense(dense(pair.a, pair.space), dense(pair.b, pair.space))
        assert_close(op.to_matrix(pair.space), want, label)

    @pytest.mark.parametrize("label,build,build_dense", EXPRESSIONS, ids=EXPR_IDS)
    @pytest.mark.parametrize("pair", PAIRS.values(), ids=list(PAIRS))
    def test_action_matches_composed_matrix(self, pair, label, build, build_dense, key):
        check_action(build(pair.a, pair.b), pair.space, key)

    @pytest.mark.parametrize("label,build,build_dense", EXPRESSIONS, ids=EXPR_IDS)
    @pytest.mark.parametrize("pair", PAIRS.values(), ids=list(PAIRS))
    def test_solve_matches_composed_matrix(self, pair, label, build, build_dense, key):
        check_solve(build(pair.a, pair.b), pair.space, key)


class TestIncompatibleStates:
    """Operators must reject states from spaces outside their domain.

    One representative case per operator type: the check is a single isinstance
    test in Operator._check_domain, so every (operator, foreign space)
    combination exercises the same branch.
    """

    @pytest.mark.parametrize(
        "op,foreign", FOREIGN_STATE_CASES.values(), ids=list(FOREIGN_STATE_CASES)
    )
    def test_call_rejects_foreign_state(self, op, foreign, key):
        y = random_state(foreign, key)
        with pytest.raises(TypeError, match=f"acts on {op.domain.__name__}"):
            op(y)

    @pytest.mark.parametrize(
        "op,foreign", FOREIGN_STATE_CASES.values(), ids=list(FOREIGN_STATE_CASES)
    )
    def test_exp_rejects_foreign_state(self, op, foreign, key):
        y = random_state(foreign, key)
        with pytest.raises(TypeError, match=f"acts on {op.domain.__name__}"):
            op.exp(-0.1j, y)

    @pytest.mark.parametrize(
        "op,foreign", FOREIGN_STATE_CASES.values(), ids=list(FOREIGN_STATE_CASES)
    )
    def test_scalar_multiple_rejects_foreign_state(self, op, foreign, key):
        """The scalar wrapper must not lose the domain check."""
        y = random_state(foreign, key)
        with pytest.raises(TypeError):
            (2.0 * op)(y)

    @pytest.mark.parametrize(
        "op,foreign", FOREIGN_STATE_CASES.values(), ids=list(FOREIGN_STATE_CASES)
    )
    def test_narrowed_sum_rejects_foreign_state(self, op, foreign, key):
        """Adding Identity narrows the sum to the operand's domain, so the sum
        must still reject what the operand rejects."""
        with pytest.raises(TypeError):
            (Identity() + op)(random_state(foreign, key))


class TestIncompatibleOperators:
    """Operators on disjoint domains must not compose."""

    @pytest.mark.parametrize(
        "a,b", INCOMPATIBLE_PAIRS.values(), ids=list(INCOMPATIBLE_PAIRS)
    )
    def test_addition_rejected_at_construction(self, a, b):
        with pytest.raises(TypeError, match="incompatible domains"):
            a + b

    @pytest.mark.parametrize(
        "a,b", INCOMPATIBLE_PAIRS.values(), ids=list(INCOMPATIBLE_PAIRS)
    )
    def test_addition_rejected_either_way_round(self, a, b):
        with pytest.raises(TypeError, match="incompatible domains"):
            b + a

    @pytest.mark.parametrize(
        "a,b", INCOMPATIBLE_PAIRS.values(), ids=list(INCOMPATIBLE_PAIRS)
    )
    def test_subtraction_also_rejected(self, a, b):
        with pytest.raises(TypeError, match="incompatible domains"):
            a - b

    @pytest.mark.parametrize("pair", PAIRS.values(), ids=list(PAIRS))
    def test_identity_composes_with_anything(self, pair, key):
        """Identity's domain is the base class, so it composes with every
        operator, and the sum narrows to the more specific domain."""
        narrowed = Identity() + pair.a
        assert narrowed.domain is pair.a.domain
        check_action(narrowed, pair.space, key)

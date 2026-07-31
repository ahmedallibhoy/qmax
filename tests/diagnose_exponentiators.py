"""Do the exponentiators achieve the order they advertise?

Each AbstractExponentiator reports an `order`. This measures it: apply the
exponentiator over a sweep of dt, compare against scipy.linalg.expm of the dense
matrix, and fit the slope of log(error) against log(dt).

Convention: order p means a single application has error O(dt^(p+1)). Forward
Euler approximates e^z by 1 + z, which first differs at z^2, so order 1 gives
slope 2. The implied order reported below is therefore slope - 1, directly
comparable to the advertised value.

An order of inf claims the exponentiation is exact up to roundoff. That shows up
as an error that never rises above ~1e-15 no matter how large dt gets, in which
case no slope can be fitted and the run is marked "roundoff". An inf-order method
that *does* produce a slope is not exact, and the implied order says how
inexact -- which is the interesting case.

dt is imaginary, as in propagation (dt -> -1j * t / hbar), and is normalized by
the operator's spectral radius so the sweep is comparable across operators.

Operators are kept small on purpose: the reference is scipy.linalg.expm of the
dense matrix, which is O(n^3) in time and O(n^2) in memory, so this cannot scale.
Cost is measured separately, at realistic sizes, by benchmark_exponentiators.py.

Run it:

    JAX_PLATFORMS=cpu python tests/diagnose_exponentiators.py

Not named test_* so pytest does not collect it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy.linalg

from conftest import PS_1D_DEALIASED, PS_1D_FULL, QUBIT, dense, random_state
from qmax.exponentiators import (
    ChebyshevExponentiator,
    CrankNicolson,
    ExactExponentiator,
    ForwardEuler,
    ImplicitEuler,
    KrylovExponentiator,
    LaguerreExponentiator,
    ScaleSquareExponentiator,
)
from test_operator import LAPLACIAN, PAULI_X, V_1D_EVEN

# Normalized step sizes. dt = -1j * z / spectral_radius, so z is roughly
# ||dt * A|| and the sweep means the same thing for every operator.
Z_VALUES = np.geomspace(1e-3, 0.5, 14)

# Window for the slope fit: above roundoff, below where the expansion stops
# being asymptotic.
FIT_FLOOR = 1e-12
FIT_CEILING = 1e-2
MIN_FIT_POINTS = 3

# Below this, an exponentiator is doing as well as the dense reference and no
# order can be measured.
ROUNDOFF = 1e-13


# --------------------------------------------------------------------------
# Representative operators
# --------------------------------------------------------------------------

# Chosen for different structure rather than coverage: diagonal in its own basis,
# dense in the mode basis, a non-commuting sum with no closed form, a tiny dense
# space, and a compression whose exp_action is known not to match expm.
OPERATORS = {
    "laplacian (diagonal)": (PS_1D_FULL, LAPLACIAN),
    "potential (dense)": (PS_1D_FULL, V_1D_EVEN),
    "T + V (non-commuting sum)": (PS_1D_FULL, -0.5 * LAPLACIAN + V_1D_EVEN),
    "pauli_x (2x2)": (QUBIT, PAULI_X),
    "potential (dealiased compression)": (PS_1D_DEALIASED, V_1D_EVEN),
}


def exponentiators(space, op):
    """Every exponentiator that can be applied to this operator.

    Krylov is capped at the space dimension: the Lanczos basis cannot exceed it,
    and iterating past breakdown produces a degenerate tridiagonal matrix.
    """
    dt_ref = -1j * 0.5 / spectral_radius(op, space)
    cheby_adapt = ChebyshevExponentiator().adapt(op, space, dt_ref)
    adapt_eps = ScaleSquareExponentiator().adapt(op, space, dt_ref)
    adapt_tol = ScaleSquareExponentiator(max_tol=1e-8).adapt(op, space, dt_ref)
    return {
        "ForwardEuler": ForwardEuler(),
        "ImplicitEuler": ImplicitEuler(),
        "CrankNicolson": CrankNicolson(),
        "Exact": ExactExponentiator(),
        f"Krylov({min(20, space.dim)})": KrylovExponentiator(min(20, space.dim)),
        # Degree sweep plus adapt. The Chebyshev truncation error is 2*|J_n(w)|
        # with w = |dt| * (lmax - lmin) / 2, so adapt can pick n from a closed
        # form -- no generated tables, unlike ScaleSquare.
        "Chebyshev(5)": ChebyshevExponentiator(5),
        "Chebyshev(10)": ChebyshevExponentiator(10),
        f"Cheby adapt(n={cheby_adapt.num_iterations})": cheby_adapt,
        "Laguerre(20)": LaguerreExponentiator(20),
        # Two independent levers, crossed. adapt picks m from the operator and dt_ref;
        # max_tol picks the backward-error table. With m fixed at 55, max_tol can
        # only act through s, so (55, 1e-8) is the control: it should show no gain
        # wherever s == 1.
        "ScaleSq(55, eps)": ScaleSquareExponentiator(),
        f"ScaleSq(m={adapt_eps.m}, eps)": adapt_eps,
        "ScaleSq(55, 1e-8)": ScaleSquareExponentiator(max_tol=1e-8),
        f"ScaleSq(m={adapt_tol.m}, 1e-8)": adapt_tol,
    }


def spectral_radius(op, space):
    return float(np.max(np.abs(np.linalg.eigvalsh(dense(op, space)))))


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def reference_solutions(op, space, y):
    """expm(dt * M) @ y over the sweep. Computed once per operator: it is the
    same reference for every exponentiator and is the slowest part of the run."""
    M = dense(op, space)
    coeffs = np.asarray(y.coeffs)
    radius = spectral_radius(op, space)
    return [scipy.linalg.expm(-1j * z / radius * M) @ coeffs for z in Z_VALUES]


def relative_errors(op, space, exponentiator, y, references):
    """Relative error of the exponentiator against expm, over the dt sweep."""
    radius = spectral_radius(op, space)

    errors = []
    for z, want in zip(Z_VALUES, references):
        dt = -1j * z / radius
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = np.asarray(exponentiator.exp(op, dt, y).coeffs)
        except Exception as exc:
            return None, f"{type(exc).__name__}"

        errors.append(np.linalg.norm(got - want) / np.linalg.norm(want))

    return np.array(errors), None



def fit_order(errors):
    """Implied order from the slope of log(error) vs log(dt), or None."""
    mask = np.isfinite(errors) & (errors > FIT_FLOOR) & (errors < FIT_CEILING)
    if mask.sum() < MIN_FIT_POINTS:
        return None, int(mask.sum()), None

    logs, loge = np.log(Z_VALUES[mask]), np.log(errors[mask])
    slope, intercept = np.polyfit(logs, loge, 1)
    residual = float(np.max(np.abs(slope * logs + intercept - loge)))
    return float(slope) - 1.0, int(mask.sum()), residual


def format_order(value):
    if value is None:
        return "--"
    if np.isinf(value):
        return "inf"
    return f"{value:g}"


def main():
    print(__doc__.split("Run it:")[0].rstrip())

    key = jax.random.key(0)
    any_mismatch = False

    for op_label, (space, op) in OPERATORS.items():
        y = random_state(space, key)
        radius = spectral_radius(op, space)
        references = reference_solutions(op, space, y)

        print()
        print("=" * 100)
        print(f"{op_label}    dim={space.dim}  spectral radius={radius:.4g}")
        print("=" * 100)
        print(
            f"{'exponentiator':<21}{'claimed':>9}{'implied':>9}{'pts':>6}"
            f"{'resid':>8}{'min err':>11}{'max err':>11}   note"
        )
        print("-" * 100)

        for exp_label, exponentiator in exponentiators(space, op).items():
            claimed = exponentiator.order
            errors, failure = relative_errors(op, space, exponentiator, y, references)

            if errors is None:
                print(
                    f"{exp_label:<21}{format_order(claimed):>9}{'--':>9}{'--':>6}"
                    f"{'--':>8}{'--':>11}{'--':>11}   raised {failure}"
                )
                continue

            implied, points, residual = fit_order(errors)
            lo, hi = float(np.min(errors)), float(np.max(errors))

            flat = hi < 10 * lo  # error barely moves across three decades of dt

            if hi < ROUNDOFF:
                # Exact to roundoff. For a finite claimed order this is not a
                # failure: a backward-error scheme like scaling-and-squaring
                # targets machine precision, so its Taylor degree m says nothing
                # about achievable accuracy.
                note = "exact to roundoff"
                if not np.isinf(claimed):
                    note += f"  (order reports m={claimed:g})"
            elif flat:
                note = f"plateau at {hi:.1e}, no dt dependence"
            elif implied is None:
                note = f"no fit window ({points} pts in range)"
            elif np.isinf(claimed):
                note = f"MISMATCH: claims exact, behaves as order {implied:.1f}"
            elif abs(implied - claimed) <= 0.3:
                note = "ok"
            elif implied > claimed:
                note = f"better than claimed ({claimed:g})"
            else:
                note = f"MISMATCH: claimed {claimed:g}"

            if note.startswith("MISMATCH"):
                any_mismatch = True

            print(
                f"{exp_label:<21}{format_order(claimed):>9}"
                f"{format_order(implied):>9}{points:>6}"
                f"{(f'{residual:.2f}' if residual is not None else '--'):>8}"
                f"{lo:>11.2e}{hi:>11.2e}   {note}"
            )

    print()
    print("=" * 100)
    print("MISMATCHES FOUND" if any_mismatch else "all measured orders agree with the claimed ones")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Do the splitting methods achieve the order they advertise?

exp(dt*(T + V)) is not exp(dt*T) exp(dt*V) unless T and V commute. A splitting
method approximates it by a product of sub-exponentials with chosen coefficients,
and its order is how fast the residual commutator error vanishes with dt.

This measures that: apply the split over a sweep of dt, compare against
scipy.linalg.expm of the dense sum, and fit the slope of log(error) against
log(dt). As in diagnose_exponentiators.py, order p means a single application has
error O(dt^(p+1)), so the implied order reported is slope - 1.

Each operand is paired with an *exact* exponentiator, which is the whole point of
the setup. AddOperator's effective order is

    min(split order, op1 exponentiator order, op2 exponentiator order)

so a second-order sub-exponentiator would silently cap a sixth-order splitting at
two, and the measurement would be of the wrong thing. Three of the four operands
are already exact by default on a full (non-dealiased) grid; only
FiniteDifferenceLaplacian defaults to CrankNicolson and has to be replaced.

Operators are kept small: the reference is a dense expm. Cost is measured
separately, at realistic sizes, by benchmark_splitting.py.

Run it:

    JAX_PLATFORMS=cpu python tests/diagnose_splitting.py

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

from conftest import dense, random_state
from qmax.exponentiators import ScaleSquareExponentiator
from qmax.spaces.finite_difference import (
    FiniteDifference,
    FiniteDifferenceLaplacian,
    FiniteDifferencePotentialEnergy,
)
from qmax.spaces.pseudospectral import (
    PseudoSpectral,
    PseudoSpectralLaplacian,
    PseudoSpectralPotentialEnergy,
)
from qmax.split import BlanesMoan6, Strang, Yoshida
from qmax.system import adapt_operator

TWO_PI = 2 * float(jnp.pi)

# A wider sweep than the exponentiator diagnostic needs. Splitting error goes
# like z^(p+1), so at order 6 it crosses from roundoff to visible over less than
# a decade of z -- the sweep has to reach high enough for a window to exist at
# all, while staying where the expansion is still asymptotic.
Z_VALUES = np.geomspace(1e-2, 3.0, 22)

FIT_FLOOR = 1e-12
FIT_CEILING = 1e-1
MIN_FIT_POINTS = 3
ROUNDOFF = 1e-13


SPLITS = {
    "Strang": Strang(),
    "Yoshida(1)": Yoshida(1),
    "Yoshida(2)": Yoshida(2),
    "BlanesMoan6": BlanesMoan6(),
}


def build(dim=24):
    """(label, space, T, V) with every operand exactly exponentiable.

    Full grids only -- num_modes == mesh_size -- so PseudoSpectralExponentiator
    is exact rather than a compression, and the splitting error is the only thing
    left in the measurement.
    """
    ps = PseudoSpectral(0.0, TWO_PI, dim, dim)
    fd = FiniteDifference(-1.0, 1.0, dim)

    return [
        (
            "pseudospectral T + V",
            ps,
            -0.5 * PseudoSpectralLaplacian(),                    # ExactExponentiator
            PseudoSpectralPotentialEnergy(lambda x: 2.0 + jnp.cos(x)),
        ),
        (
            "finite difference T + V",
            fd,
            # CrankNicolson by default, which is order 2 and would cap everything.
            # ScaleSquare rather than Chebyshev: it recomputes s from the actual
            # dt every call, so it cannot go stale, and an over-sized m is merely
            # wasteful. Chebyshev bakes the degree in and returns NaN when it
            # exceeds what w needs -- its Miller recurrence overflows.
            (-0.5 * FiniteDifferenceLaplacian()).with_exponentiator(
                ScaleSquareExponentiator()
            ),
            FiniteDifferencePotentialEnergy(lambda x: 2.0 + jnp.cos(x)),
        ),
    ]


def spectral_radius(op, space):
    return float(np.max(np.abs(np.linalg.eigvalsh(dense(op, space)))))


def fit_order(errors):
    """Implied order from the slope of log(error) vs log(dt), or None."""
    mask = np.isfinite(errors) & (errors > FIT_FLOOR) & (errors < FIT_CEILING)
    if mask.sum() < MIN_FIT_POINTS:
        return None, int(mask.sum()), None

    logs, loge = np.log(Z_VALUES[mask]), np.log(errors[mask])
    slope, intercept = np.polyfit(logs, loge, 1)
    residual = float(np.max(np.abs(slope * logs + intercept - loge)))
    return float(slope) - 1.0, int(mask.sum()), residual


def measure(op, space, y, radius, references):
    """(errors, drifts) over the dt sweep, or (None, message) on failure."""
    norm_y = float(np.linalg.norm(np.asarray(y.coeffs)))
    errors, drifts = [], []

    for z, want in zip(Z_VALUES, references):
        dt = -1j * z / radius
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = np.asarray(op.exp(dt, y).coeffs)
        except Exception as exc:
            return None, f"raised {type(exc).__name__}"

        errors.append(np.linalg.norm(got - want) / np.linalg.norm(want))
        drifts.append(float(np.linalg.norm(got) / norm_y - 1.0))

    return np.array(errors), np.array(drifts)


def main():
    print(__doc__.split("Run it:")[0].rstrip())

    key = jax.random.key(0)
    mismatches = 0

    for label, space, kinetic, potential in build():
        total = kinetic + potential
        y = random_state(space, key)
        radius = spectral_radius(total, space)

        M = dense(total, space)
        coeffs = np.asarray(y.coeffs)
        references = [
            scipy.linalg.expm(-1j * z / radius * M) @ coeffs for z in Z_VALUES
        ]

        print()
        print("=" * 96)
        print(f"{label}    dim={space.dim}  spectral radius={radius:.4g}")
        print("=" * 96)
        print(
            f"{'split method':<16}{'claimed':>9}{'implied':>9}{'pts':>6}{'resid':>8}"
            f"{'min err':>11}{'max err':>11}{'norm drift':>13}   note"
        )
        print("-" * 96)

        for name, split in SPLITS.items():
            op = (kinetic + potential).with_exponentiator(split)
            # adapt walks the tree, scaling dt by the split's own coefficients
            op = adapt_operator(op, space, abs(-1j * Z_VALUES[-1] / radius))

            errors, drifts = measure(op, space, y, radius, references)
            if errors is None:
                print(f"{name:<16}{split.order:>9}{'--':>9}{'--':>6}{'--':>8}"
                      f"{'--':>11}{'--':>11}{'--':>13}   {drifts}")
                continue

            implied, points, residual = fit_order(errors)
            lo, hi = float(np.min(errors)), float(np.max(errors))
            drift = float(drifts[np.argmax(np.abs(drifts))])

            if hi < ROUNDOFF:
                note = "exact to roundoff (operands commute?)"
            elif implied is None:
                note = f"no fit window ({points} pts in range)"
            elif abs(implied - split.order) <= 0.35:
                note = "ok"
            else:
                note = f"MISMATCH: claimed {split.order}"
                mismatches += 1

            print(
                f"{name:<16}{split.order:>9}{('--' if implied is None else f'{implied:.4g}'):>9}"
                f"{points:>6}{(f'{residual:.2f}' if residual is not None else '--'):>8}"
                f"{lo:>11.2e}{hi:>11.2e}{drift:>+13.2e}   {note}"
            )

    print()
    print("=" * 96)
    print("MISMATCHES FOUND" if mismatches else "all measured orders agree with the claimed ones")
    print("norm drift should be ~1e-16 for every method: a splitting is a product of")
    print("unitaries whenever its coefficients are real and the operands are exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""What does each exponentiator cost at a realistic problem size?

Companion to diagnose_exponentiators.py, which measures accuracy. That one is
pinned to small operators because its reference is a dense expm. This one needs
no reference, so it runs at sizes where the differences are worth caring about:
at dim 256 everything finishes in single-digit microseconds and the ranking is
dispatch noise, not work.

Operators are benchmarked one at a time rather than as T + V, because that is
what a splitting method actually exponentiates. T and V also isolate the cost of
solve, which the implicit methods depend on entirely: PseudoSpectralLaplacian
overloads it with a diagonal inversion, PseudoSpectralPotentialEnergy does not
and falls through to the generic lineax path in Operator.solve.

Every call is blocked before the clock stops, since jax dispatch is asynchronous,
and the repeat count adapts to the measured cost so a 20 us method and a 300 ms
method are both timed to the same precision. Compile time is reported separately.

Run it:

    JAX_PLATFORMS=cpu python tests/benchmark_exponentiators.py

Not named test_* so pytest does not collect it.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from conftest import random_state
from diagnose_exponentiators import FIT_CEILING, FIT_FLOOR, fit_order, format_order
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

# Sizes spanning the regime where cost starts to matter. 1d, since the operators
# are separable and the dimension is what drives the FFT cost.
SIZES = (4096, 65536)

# Normalized step sizes for the order fit, as in diagnose_exponentiators.py.
# dt = -1j * z / spectral_radius.
Z_VALUES = np.geomspace(1e-3, 0.5, 12)

TARGET_SECONDS = 0.1  # per timed batch
MAX_REPEATS = 2000
ROUNDS = 3


def build(dim):
    """(label, space, operator) for each discretization, at one size.

    The four differ mainly in how solve is implemented, which is what the
    implicit methods live or die by:

        pseudospectral T   diagonal in the mode basis, one divide
        pseudospectral V   no override at all, falls through to matrix-free GMRES
        finite difference T  tridiagonal, O(n) but not a simple divide
        finite difference V  diagonal in position space, one divide

    Finite differences are 1d only: FiniteDifferencePotentialEnergy assumes the
    grid and the coefficient vector have the same shape.
    """
    ps = PseudoSpectral(0.0, 2 * float(jnp.pi), dim, dim)
    fd = FiniteDifference(-1.0, 1.0, dim)

    return [
        ("ps T", "pseudospectral  T = -0.5 * laplacian   (diagonal solve)",
         ps, -0.5 * PseudoSpectralLaplacian()),
        ("ps V", "pseudospectral  V = potential          (no solve override -> GMRES)",
         ps, PseudoSpectralPotentialEnergy(lambda x: 2.0 + jnp.cos(x))),
        ("fd T", "finite diff     T = -0.5 * laplacian   (tridiagonal solve, no exp_action)",
         fd, -0.5 * FiniteDifferenceLaplacian()),
        ("fd V", "finite diff     V = potential          (diagonal solve)",
         fd, FiniteDifferencePotentialEnergy(lambda x: 2.0 + jnp.cos(x))),
    ]


def reference_for(op, space, dt, y):
    """The most trustworthy exponentiator available for this operator.

    ExactExponentiator where there is a closed form; FiniteDifferenceLaplacian
    has none, so fall back to scaling-and-squaring, which the accuracy diagnostic
    shows is exact to roundoff at and below its construction dt.
    """
    try:
        ExactExponentiator().exp(op, dt, y)
        return ExactExponentiator()
    except NotImplementedError:
        return ScaleSquareExponentiator().adapt(op, space, dt)


def spectral_radius(op, space):
    """From the operator's own bounds -- a dense eigendecomposition is exactly
    what this script is trying to avoid."""
    lo, hi = np.asarray(op.spectral_bounds(space), dtype=float)
    return max(abs(lo), abs(hi))


def exponentiators(space, op, dt):
    cheby_adapt = ChebyshevExponentiator().adapt(op, space, dt)
    adapt_eps = ScaleSquareExponentiator().adapt(op, space, dt)
    adapt_tol = ScaleSquareExponentiator(max_tol=1e-8).adapt(op, space, dt)
    return {
        "ForwardEuler": ForwardEuler(),
        "ImplicitEuler": ImplicitEuler(),
        "CrankNicolson": CrankNicolson(),
        "Exact": ExactExponentiator(),
        "Krylov(5) plain": KrylovExponentiator(5, orthogonalize=False),
        "Krylov(10) plain": KrylovExponentiator(10, orthogonalize=False),
        "Krylov(5) orth": KrylovExponentiator(5, orthogonalize=True),
        "Krylov(10) orth": KrylovExponentiator(10, orthogonalize=True),
        # Degree sweep plus adapt. The Chebyshev truncation error is 2*|J_n(w)|
        # with w = |dt| * (lmax - lmin) / 2, so adapt can pick n from a closed
        # form -- no generated tables, unlike ScaleSquare.
        "Chebyshev(5)": ChebyshevExponentiator(5),
        "Chebyshev(10)": ChebyshevExponentiator(10),
        f"Cheby adapt(n={cheby_adapt.num_iterations})": cheby_adapt,
        "Laguerre(20)": LaguerreExponentiator(20),
        # Two independent levers, crossed. adapt picks m from the operator and dt;
        # max_tol picks the backward-error table. With m fixed at 55, max_tol can
        # only act through s, so (55, 1e-8) is the control: it should show no gain
        # wherever s == 1.
        "ScaleSq(55, eps)": ScaleSquareExponentiator(),
        f"ScaleSq(m={adapt_eps.m}, eps)": adapt_eps,
        "ScaleSq(55, 1e-8)": ScaleSquareExponentiator(max_tol=1e-8),
        f"ScaleSq(m={adapt_tol.m}, 1e-8)": adapt_tol,
    }


def measure_accuracy(op, exponentiator, y, radius, reference):
    """(implied order, max relative error, worst norm drift) over the dt sweep.

    Order is fitted against a matrix-free reference rather than a dense expm,
    which is what makes it measurable at these sizes.

    Norm drift measures departure from unitarity. dt is imaginary and every
    operator here is Hermitian, so the true propagator is unitary and
    ||U y|| == ||y||; the signed drift says whether a method amplifies (+) or
    damps (-). This matters more than single-step error for long propagation,
    because it compounds over steps rather than averaging out.
    """
    norm_y = float(np.linalg.norm(np.asarray(y.coeffs)))
    errors, drifts = [], []

    for z in Z_VALUES:
        dt = -1j * z / radius
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                want = np.asarray(reference.exp(op, dt, y).coeffs)
                got = np.asarray(exponentiator.exp(op, dt, y).coeffs)
        except Exception:
            return None, None, None

        errors.append(np.linalg.norm(got - want) / np.linalg.norm(want))
        drifts.append(float(np.linalg.norm(got) / norm_y - 1.0))

    errors, drifts = np.array(errors), np.array(drifts)
    implied, _, _ = fit_order_over(errors)
    worst_drift = float(drifts[np.argmax(np.abs(drifts))])
    return implied, float(np.max(errors)), worst_drift


def fit_order_over(errors):
    """fit_order from the companion script, against this script's z grid."""
    mask = np.isfinite(errors) & (errors > FIT_FLOOR) & (errors < FIT_CEILING)
    if mask.sum() < 3:
        return None, int(mask.sum()), None
    logs, loge = np.log(Z_VALUES[mask]), np.log(errors[mask])
    slope, intercept = np.polyfit(logs, loge, 1)
    return float(slope) - 1.0, int(mask.sum()), None


def measure(op, exponentiator, y, dt):
    """(compile seconds, per-call seconds), or (None, message) on failure."""
    apply = jax.jit(lambda dt: exponentiator.exp(op, dt, y).coeffs)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            jax.block_until_ready(apply(dt))
            compile_seconds = time.perf_counter() - start

            # one untimed call to size the batch
            start = time.perf_counter()
            jax.block_until_ready(apply(dt))
            single = time.perf_counter() - start

            repeats = int(np.clip(TARGET_SECONDS / max(single, 1e-9), 1, MAX_REPEATS))
            best = np.inf
            for _ in range(ROUNDS):
                start = time.perf_counter()
                for _ in range(repeats):
                    jax.block_until_ready(apply(dt))
                best = min(best, (time.perf_counter() - start) / repeats)
    except Exception as exc:
        message = str(exc).split("\n")[0]
        if "RESOURCE_EXHAUSTED" in message or "Out of memory" in message:
            wanted = message.split("allocating")[-1].strip().rstrip(".")
            return None, f"OUT OF MEMORY, wanted {wanted}"
        return None, f"raised {type(exc).__name__}"

    return compile_seconds, best


# Reorthogonalization is the dominant cost in Lanczos -- double Gram-Schmidt
# against the whole basis every step, O(m^2 n) -- but dropping it lets the
# three-term recurrence lose orthogonality in floating point. These four
# permutations price that tradeoff.
KRYLOV_VARIANTS = {
    "Krylov(5), orth": KrylovExponentiator(5, orthogonalize=True),
    "Krylov(5), plain": KrylovExponentiator(5, orthogonalize=False),
    "Krylov(10), orth": KrylovExponentiator(10, orthogonalize=True),
    "Krylov(10), plain": KrylovExponentiator(10, orthogonalize=False),
}


def print_cross_table(title, results, shorts, extract):
    print()
    print(title)
    header = f"{'variant':<20}" + "".join(f"{s:>14}" for s in shorts)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for variant, per_operator in results.items():
        line = f"{variant:<20}"
        for short in shorts:
            line += f"{extract(per_operator[short]):>14}"
        print(line)


def krylov_comparison(dim, key):
    """Every Krylov permutation against every operator, side by side."""
    operators = build(dim)
    shorts = [short for short, _, _, _ in operators]
    results = {variant: {} for variant in KRYLOV_VARIANTS}

    for short, _, space, op in operators:
        y = random_state(space, key)
        radius = spectral_radius(op, space)
        dt = -1j * 0.5 / radius
        reference = reference_for(op, space, dt, y)

        for variant, exponentiator in KRYLOV_VARIANTS.items():
            _, error, drift = measure_accuracy(op, exponentiator, y, radius, reference)
            _, seconds = measure(op, exponentiator, y, dt)
            results[variant][short] = (error, drift, seconds)

    print()
    print("=" * 96)
    print(f"KRYLOV PERMUTATIONS, dim {dim}")
    print("=" * 96)

    print_cross_table(
        "max relative error over the dt sweep",
        results, shorts,
        lambda r: "--" if r[0] is None else f"{r[0]:.2e}",
    )
    print_cross_table(
        "norm drift (0 means unitary; the exact propagator is)",
        results, shorts,
        lambda r: "--" if r[1] is None else f"{r[1]:+.1e}",
    )
    print_cross_table(
        "per call",
        results, shorts,
        lambda r: "--" if r[2] is None else format_time(r[2]).strip(),
    )


def format_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:8.2f} ms"
    return f"{seconds:8.2f} s "


def main():
    print(__doc__.split("Run it:")[0].rstrip())

    key = jax.random.key(0)

    for dim in SIZES:
        for _, op_label, space, op in build(dim):
            y = random_state(space, key)
            radius = spectral_radius(op, space)
            dt = -1j * 0.5 / radius

            print()
            print("=" * 96)
            print(f"dim {dim}    {op_label}")
            print("=" * 96)
            print(
                f"{'exponentiator':<21}{'claimed':>9}{'implied':>9}{'max err':>11}"
                f"{'norm drift':>13}{'compile':>10}{'per call':>13}{'relative':>10}   note"
            )
            print("-" * 110)

            reference = reference_for(op, space, dt, y)
            baseline = None
            for label, exponentiator in exponentiators(space, op, dt).items():
                claimed = exponentiator.order
                compile_seconds, result = measure(op, exponentiator, y, dt)

                if compile_seconds is None:
                    print(
                        f"{label:<21}{format_order(claimed):>9}{'--':>9}{'--':>11}"
                        f"{'--':>13}{'--':>10}{'--':>13}{'--':>10}   {result}"
                    )
                    continue

                if baseline is None:
                    baseline = result

                implied, max_error, drift = measure_accuracy(
                    op, exponentiator, y, radius, reference
                )

                notes = []
                if max_error is not None and max_error < 1e-13:
                    notes.append("exact to roundoff")
                if drift is not None and abs(drift) > 1e-10:
                    notes.append("AMPLIFIES" if drift > 0 else "DAMPS")

                print(
                    f"{label:<21}{format_order(claimed):>9}{format_order(implied):>9}"
                    f"{(f'{max_error:.2e}' if max_error is not None else '--'):>11}"
                    f"{(f'{drift:+.2e}' if drift is not None else '--'):>13}"
                    f"{compile_seconds:>9.2f}s{format_time(result):>13}"
                    f"{result / baseline:>9.1f}x   {', '.join(notes)}"
                )

    krylov_comparison(SIZES[-1], key)

    print()
    print("=" * 96)
    print("relative   per-call cost against ForwardEuler on the same operator")
    print("implied    order fitted against a matrix-free reference, not a dense expm,")
    print("           so it is measurable at these sizes (-- means no fit window)")
    print("norm drift ||U y||/||y|| - 1 at its worst over the sweep. The exact")
    print("           propagator is unitary here, so nonzero means the method is not")
    return 0


if __name__ == "__main__":
    sys.exit(main())

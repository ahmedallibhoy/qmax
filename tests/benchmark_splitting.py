"""What does each splitting method cost at a realistic problem size?

Cost, anchored by an accuracy floor. The *order* belongs to
diagnose_splitting.py and cannot be fitted here: splitting error at fixed dt*rho
falls as 1/dim^2, so by dim 4096 every method sits at roundoff and there is no
slope to find. But the error itself is still worth reporting -- it is what
grounds the timings. At these sizes it should be at roundoff, and a method that
is not is one to investigate, whatever its cost.

The reference is ScaleSquare adapted to the sum at the largest dt in the sweep.
It stays valid across the whole sweep because ScaleSquare recomputes s from the
actual dt on every call, with only m fixed. Chebyshev does not -- its degree is
baked in, so one sized at the timing dt is unconverged at the top of the sweep,
and an earlier version of this script consequently reported the reference's own
truncation error as if it were the splitting error, identically for all four
methods.

To rank methods by cost per unit accuracy, combine the two scripts. With per-step
error e ~ C z^(p+1) and W = T * rho worth of dimensionless propagation time,

    global error ~ (W / z) * C * z^(p+1) = W * C * z^p

so reaching eps needs z* = (eps / (W C))^(1/p) and N* = W / z* steps. Take p and
C from the diagnostic, which measures them at a size where they are visible, and
the per-call cost from here. At eps = 1e-8 and W = 1000 that gives, in
pseudospectral work at dim 65536:

    BlanesMoan6  2.4 s  <  Yoshida(2)  3.6 s  <  Yoshida(1)  6.8 s  <<  Strang  520 s

Norm drift is still reported, because it needs no reference: a splitting with
real coefficients applied to exactly-exponentiated Hermitian operands is a
product of unitaries, so it should sit at roundoff whatever the order.

Each operand is paired with an exact exponentiator, for the reason spelled out in
diagnose_splitting.py: AddOperator's effective order is the minimum over the
split and both operands, so an inexact operand would cap the whole thing.

Run it:

    JAX_PLATFORMS=cpu python tests/benchmark_splitting.py

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

import numpy as np

from conftest import random_state
from diagnose_splitting import SPLITS, Z_VALUES, build
from qmax.exponentiators import ScaleSquareExponentiator
from qmax.system import adapt_operator

SIZES = (4096, 65536)

TARGET_SECONDS = 0.1
MAX_REPEATS = 500
ROUNDS = 3


def spectral_radius(op, space):
    lo, hi = np.asarray(op.spectral_bounds(space), dtype=float)
    return max(abs(lo), abs(hi))


def accuracy(op, y, radius, reference, total):
    """(max relative error, worst norm drift) over the dt sweep, or (None, None).

    Norm drift needs no reference -- the exact propagator is unitary here, so any
    departure from 1 is the method's own. The error does, and expecting it at
    roundoff is the point: this is a floor check, not an order measurement.
    """
    norm_y = float(np.linalg.norm(np.asarray(y.coeffs)))
    worst_error, worst_drift = 0.0, 0.0

    for z in Z_VALUES:
        dt = -1j * z / radius
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                want = np.asarray(reference.exp(total, dt, y).coeffs)
                got = np.asarray(op.exp(dt, y).coeffs)
        except Exception:
            return None, None

        worst_error = max(
            worst_error, np.linalg.norm(got - want) / np.linalg.norm(want)
        )
        drift = float(np.linalg.norm(got) / norm_y - 1.0)
        if abs(drift) > abs(worst_drift):
            worst_drift = drift

    return float(worst_error), worst_drift


def measure_cost(op, y, dt):
    """(compile seconds, per-call seconds) or (None, message)."""
    apply = jax.jit(lambda dt: op.exp(dt, y).coeffs)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            jax.block_until_ready(apply(dt))
            compile_seconds = time.perf_counter() - start

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
        return None, f"raised {type(exc).__name__}"
    return compile_seconds, best


def format_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:7.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:7.2f} ms"
    return f"{seconds:7.2f} s "


def main():
    print(__doc__.split("Run it:")[0].rstrip())
    key = jax.random.key(0)

    for dim in SIZES:
        for label, space, kinetic, potential in build(dim):
            total = kinetic + potential
            y = random_state(space, key)
            radius = spectral_radius(total, space)
            dt = -1j * 0.5 / radius

            reference = ScaleSquareExponentiator().adapt(
                total, space, -1j * Z_VALUES[-1] / radius
            )

            print()
            print("=" * 96)
            print(f"dim {dim}    {label}    spectral radius={radius:.4g}")
            print("=" * 96)
            print(
                f"{'split method':<16}{'order':>7}{'max err':>12}{'norm drift':>13}"
                f"{'compile':>10}{'per call':>12}{'relative':>10}"
            )
            print("-" * 96)

            baseline = None
            for name, split in SPLITS.items():
                op = (kinetic + potential).with_exponentiator(split)
                op = adapt_operator(op, space, abs(-1j * Z_VALUES[-1] / radius))

                compile_seconds, per_call = measure_cost(op, y, dt)
                if compile_seconds is None:
                    print(f"{name:<16}{split.order:>7}   {per_call}")
                    continue
                if baseline is None:
                    baseline = per_call

                error, drift = accuracy(op, y, radius, reference, total)

                print(
                    f"{name:<16}{split.order:>7}"
                    f"{('--' if error is None else f'{error:.2e}'):>12}"
                    f"{('--' if drift is None else f'{drift:+.1e}'):>13}"
                    f"{compile_seconds:>9.2f}s{format_time(per_call):>12}"
                    f"{per_call / baseline:>9.1f}x"
                )

    print()
    print("=" * 96)
    print("max err      against ScaleSquare adapted to the sum; expected at roundoff")
    print("             at these sizes, so it is a floor check, not an order estimate")
    print("relative     per-call cost against Strang on the same operator")
    print("norm drift   should be ~1e-16 for every method regardless of order:")
    print("             a real-coefficient splitting of exact operands is unitary")
    print("order        claimed, and confirmed by diagnose_splitting.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

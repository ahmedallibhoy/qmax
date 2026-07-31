"""How loose is spectral_bounds?

The test suite only checks *soundness* -- that the interval contains the true
spectrum -- because that is the property whose violation is silently wrong. This
script reports the quantitative side: how much slack each operator's bound has
relative to the tightest possible interval, obtained by diagonalizing to_matrix.

Looseness costs something. ScaleSquareExponentiator picks (m, s) from the
interval width, so a loose bound just buys extra Taylor steps. Chebyshev is worse
off: it scales by half the width and expands with a fixed number of terms, so a
loose interval degrades accuracy at fixed cost. A zero-width interval breaks it
outright by dividing by zero.

Run it:

    JAX_PLATFORMS=cpu python tests/diagnose_spectral_bounds.py

Not named test_* so pytest does not collect it. Operators and spaces are reused
from the test tables so the two never drift apart.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from conftest import dense, true_bounds
from test_operator import BOUND_COMPOSITES, CASES, PAIRS

TOL = 1e-12


def hermiticity(op, space):
    """Relative departure from Hermitian, ||M - M^H|| / ||M||.

    A real interval only means anything for a Hermitian operator, so this says
    whether the question is even well posed for a given case.
    """
    M = dense(op, space)
    scale = np.linalg.norm(M)
    return float(np.linalg.norm(M - M.conj().T) / (scale if scale > 0 else 1.0))


def measure(op, space):
    """Reported vs tightest-possible interval, plus normalized slack."""
    lo, hi = (float(v) for v in np.asarray(op.spectral_bounds(space), dtype=float))
    lo_true, hi_true = true_bounds(op, space)

    scale = max(abs(lo_true), abs(hi_true)) or 1.0
    true_width = hi_true - lo_true

    return {
        "lo": lo,
        "hi": hi,
        "lo_true": lo_true,
        "hi_true": hi_true,
        # positive means the bound is outside the spectrum, i.e. sound
        "rel_lo": (lo_true - lo) / scale,
        "rel_hi": (hi - hi_true) / scale,
        "width": (hi - lo) / true_width if true_width > TOL else None,
        "herm": hermiticity(op, space),
        "sound": (lo <= lo_true + TOL * scale) and (hi >= hi_true - TOL * scale),
    }


COLUMNS = [
    ("lo", "{:>11.4g}"),
    ("hi", "{:>11.4g}"),
    ("lo_true", "{:>11.4g}"),
    ("hi_true", "{:>11.4g}"),
    ("rel_lo", "{:>9.2e}"),
    ("rel_hi", "{:>9.2e}"),
    ("width", "{:>7.3f}"),
    ("herm", "{:>8.1e}"),
]


def _cell(value, spec):
    if value is None:
        width = spec.split(":")[1].split(".")[0].lstrip(">")
        return f"{'--':>{width}}"
    return spec.format(value)


def print_table(title, rows):
    if not rows:
        return

    label_width = max(len(label) for label, _ in rows)
    header = f"{'case':<{label_width}}"
    for name, spec in COLUMNS:
        width = int(spec.split(":")[1].split(".")[0].lstrip(">"))
        header += f"  {name:>{width}}"

    print()
    print(title)
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for label, m in rows:
        line = f"{label:<{label_width}}"
        for name, spec in COLUMNS:
            line += "  " + _cell(m[name], spec)
        if not m["sound"]:
            line += "   <-- UNSOUND"
        print(line)


def main():
    primitives = [(label, measure(c.op, c.space)) for label, c in CASES.items()]

    composites = []
    for pair_label, pair in PAIRS.items():
        space = pair.space
        for expr_label, build in BOUND_COMPOSITES.items():
            label = f"{pair_label} | {expr_label}"
            composites.append((label, measure(build(pair.a, pair.b), space)))

    print(__doc__.split("Run it:")[0].rstrip())
    print()
    print("rel_lo / rel_hi  slack outside the true spectrum, relative to its scale")
    print("                 (>= 0 is sound; 0 means the bound is exactly tight)")
    print("width            reported width / true width (1.0 is tight, -- if true width is 0)")
    print("herm             ||M - M^H|| / ||M||, so a real interval is well posed")

    print_table("PRIMITIVE OPERATORS", primitives)
    print_table("COMPOSITE OPERATORS", composites)

    every = primitives + composites
    unsound = [label for label, m in every if not m["sound"]]
    ratios = [(m["width"], label) for label, m in every if m["width"] is not None]
    degenerate = [label for label, m in every if m["width"] is None]

    print()
    print("SUMMARY")
    print("-" * 60)
    print(f"  cases measured        {len(every)}")
    print(f"  unsound               {len(unsound)}")
    for label in unsound:
        print(f"      {label}")
    if ratios:
        ratios.sort(reverse=True)
        print(f"  loosest width ratios")
        for ratio, label in ratios[:5]:
            print(f"      {ratio:6.3f}  {label}")
        tight = sum(1 for r, _ in ratios if r < 1.0 + 1e-9)
        print(f"  exactly tight         {tight} / {len(ratios)}")
    if degenerate:
        print(f"  zero true width       {len(degenerate)}  (breaks Chebyshev: a = 0)")
        for label in degenerate:
            print(f"      {label}")

    return 1 if unsound else 0


if __name__ == "__main__":
    sys.exit(main())

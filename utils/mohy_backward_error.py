"""
Generate the theta_m tables from Al-Mohy & Higham (2011), Table 3.1.

theta_m is the largest theta such that the truncated Taylor series T_m has
backward error <= tol when applied to a matrix of norm theta:

    T_m(A) = exp(A + h_{m+1}(A)),   h_{m+1}(x) = log(e^{-x} T_m(x))

Bounding ||h_{m+1}(A)|| by the dominating series (absolute-value coefficients),

    theta_m = max { theta : tilde_h_{m+1}(theta) / theta <= tol },

where tol = unit roundoff (2^-53 double, 2^-24 single). This is a purely
scalar computation, done here in high precision with mpmath.
"""

import mpmath as mp

mp.mp.dps = 80

TOL_DOUBLE = mp.mpf(2) ** -53
TOL_SINGLE = mp.mpf(2) ** -24


def _tilde_h_coeffs(m, N):
    """Coefficients |c_k| of the dominating backward-error series, k = 0..N-1."""
    # power series of g(x) = e^{-x} * T_m(x) to order N (convolution); g_0 = 1
    e = [(-1) ** j / mp.factorial(j) for j in range(N)]                        # e^{-x}
    t = [mp.mpf(1) / mp.factorial(i) if i <= m else mp.mpf(0) for i in range(N)]  # T_m
    g = [mp.fsum(e[j] * t[k - j] for j in range(k + 1)) for k in range(N)]

    # h = log(g) via the series recurrence  k*h_k = k*g_k - sum_{j<k} j*h_j*g_{k-j}
    h = [mp.mpf(0)] * N
    for k in range(1, N):
        h[k] = g[k] - mp.fsum(j * h[j] * g[k - j] for j in range(1, k)) / k

    return [abs(v) for v in h]


def theta_m(m, tol=TOL_DOUBLE, N=170):
    c = _tilde_h_coeffs(m, N)
    # tilde_h(theta)/theta = sum_{k>=1} c_k theta^{k-1}; strictly increasing in theta,
    # from -tol at theta->0 to +inf, so the positive root is unique. Bisect for it.
    f = lambda th: mp.fsum(c[k] * th ** (k - 1) for k in range(1, N)) - tol

    lo = mp.mpf(10) ** -40           # f(lo) < 0 (leading term ~ theta^m is negligible)
    hi = mp.mpf(1)
    while f(hi) < 0:
        hi *= 2
    for _ in range(300):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate_table(tol, m_max=55, N=170):
    return {m: theta_m(m, tol=tol, N=N) for m in range(1, m_max + 1)}


# A ladder of backward-error targets, not just the two unit roundoffs. theta_m
# grows as the tolerance loosens, so a looser target needs fewer Taylor terms and
# fewer squarings for the same operator -- which is the whole reason to offer the
# choice. The two machine epsilons are kept because they are the natural defaults
# for float64 and float32.
TOLERANCES = (
    ("2.0**-53", TOL_DOUBLE),          # float64 unit roundoff, 1.11e-16
    ("1e-14", mp.mpf("1e-14")),
    ("1e-12", mp.mpf("1e-12")),
    ("1e-10", mp.mpf("1e-10")),
    ("1e-8", mp.mpf("1e-8")),
    ("2.0**-24", TOL_SINGLE),          # float32 unit roundoff, 5.96e-8
    ("1e-6", mp.mpf("1e-6")),
    ("1e-4", mp.mpf("1e-4")),
)


def emit_tables(m_max=55, N=170, per_line=5):
    """Print the tables as a paste-ready dict for qmax/exp.py."""
    print("THETA_TABLES = {")
    for label, tol in TOLERANCES:
        table = generate_table(tol, m_max=m_max, N=N)
        print(f"    {label}: jnp.array([")
        for start in range(1, m_max + 1, per_line):
            row = [
                mp.nstr(table[m], 8)
                for m in range(start, min(start + per_line, m_max + 1))
            ]
            print("        " + ", ".join(row) + ",")
        print("    ]),")
    print("}")


if __name__ == "__main__":
    emit_tables()

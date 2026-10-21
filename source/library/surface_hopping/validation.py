"""Regression tests for the surface-hopping solvers.

Run directly::

    python library/surface_hopping/validation.py

Checks the MASH estimator and equations of motion against numerically exact
results for (i) a bare electronic system, (ii) a single harmonic mode, and
(iii) a single quartic (anharmonic) mode.  These lock in the operator-coefficient
and nonadiabatic-coupling sign conventions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from surface_hopping import mash
from qutils import Pauli, basis, ladder_ops


# ---------------------------------------------------------------------------
# Exact references
# ---------------------------------------------------------------------------
def exact_1mode(epsilon, Delta, w0, c0, beta, t, rho0, Aop, Nf=80, a=0.0):
    """Exact reduced dynamics for one bosonic mode (quartic optional).

    ``a`` is the physical quartic coefficient entering ``(a^2 w0/4) q^4``.
    """
    a_op, ad = ladder_ops(Nf)
    I2, Inf = np.eye(2), np.eye(Nf)
    sz, sx = Pauli("z"), Pauli("x")
    q = (a_op + ad) / np.sqrt(2.0)
    H = (np.kron(epsilon * sz + Delta * sx, Inf)
         + np.kron(I2, w0 * (ad @ a_op + 0.5 * Inf))
         + np.kron(c0 * sz, q))
    if a != 0.0:
        H = H + np.kron(I2, (a ** 2 * w0 / 4.0) * (q @ q @ q @ q))
    pn = np.exp(-beta * w0 * np.arange(Nf))
    pn /= pn.sum()
    rho = np.kron(rho0, np.diag(pn))
    evals, evecs = np.linalg.eigh(H)
    out = np.empty(len(t), complex)
    for i, tt in enumerate(t):
        rt = evecs.conj().T @ rho @ evecs
        rt = rt * np.exp(-1j * (evals[:, None] - evals[None, :]) * tt)
        rt = evecs @ rt @ evecs.conj().T
        rs = np.array([[sum(rt[i1 * Nf + n, i2 * Nf + n] for n in range(Nf))
                        for i2 in range(2)] for i1 in range(2)])
        out[i] = np.trace(Aop @ rs)
    return out


def exact_1mode_qsbm(epsilon, Delta, w0, c0, beta, t, rho0, Aop,
                     alpha0=0.0, alpha1=1.0, alpha2=0.0, a=0.0, Nf=80):
    """Exact single mode for QSBM: ``H_sys = 2 epsilon |1><1| + Delta sigma_x``,
    ``V = |1><1|``, ``U(q) = a0 + a1 Q + a2 Q^2`` with ``Q = c0 q``."""
    a_op, ad = ladder_ops(Nf)
    I2, Inf = np.eye(2), np.eye(Nf)
    sx = Pauli("x")
    kb1 = basis(2, 1, kind = "ketbra")
    Hs = 2.0 * epsilon * kb1 + Delta * sx
    V = kb1
    q = (a_op + ad) / np.sqrt(2.0)
    U = (alpha0 * Inf + alpha1 * c0 * q + alpha2 * c0 ** 2 * (q @ q))
    H = (np.kron(Hs, Inf) + np.kron(V, U)
         + np.kron(I2, w0 * (ad @ a_op + 0.5 * Inf)))
    if a != 0.0:
        H = H + np.kron(I2, (a ** 2 * w0 / 4.0) * (q @ q @ q @ q))
    pn = np.exp(-beta * w0 * np.arange(Nf))
    pn /= pn.sum()
    rho = np.kron(rho0, np.diag(pn))
    evals, evecs = np.linalg.eigh(H)
    out = np.empty(len(t), complex)
    for i, tt in enumerate(t):
        rt = evecs.conj().T @ rho @ evecs
        rt = rt * np.exp(-1j * (evals[:, None] - evals[None, :]) * tt)
        rt = evecs @ rt @ evecs.conj().T
        rs = np.array([[sum(rt[i1 * Nf + n, i2 * Nf + n] for n in range(Nf))
                        for i2 in range(2)] for i1 in range(2)])
        out[i] = np.trace(Aop @ rs)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _pairs_bare():
    rho0 = basis(2, 0, kind = "ketbra")
    X, Y, Z = Pauli("x"), Pauli("y"), Pauli("z")
    return [(X, X), (Y, X), (X, Y), (Z, Y), (X, Z), (Z, X), (Z, Z),
            (rho0, rho0)]


def test_bare(tol=0.05, n_traj=20000, seed=3):
    """MASH must be exact for a bare electronic system (Appendix B)."""
    epsilon, Delta = 1.0, 1.0
    om, cc = np.array([1.0]), np.array([1e-6])
    t = np.linspace(0.0, 8.0, 1601)
    pairs = _pairs_bare()
    Cs = mash.run_correlations(epsilon, Delta, om, cc, 1.0, t, pairs,
                               quantum=True, n_traj=n_traj, seed=seed)
    worst = 0.0
    for (A, B), C in zip(pairs, Cs):
        ex = exact_1mode(epsilon, Delta, 1.0, 1e-6, 1e6, t, A, B)
        worst = max(worst, float(np.max(np.abs(C - ex))))
    return worst, tol


def test_bare_fssh(tol=0.05, n_traj=20000, seed=3):
    """FSSH must be exact for a bare electronic system (density-matrix approach)."""
    from surface_hopping import fssh
    epsilon, Delta = 1.0, 1.0
    om, cc = np.array([1.0]), np.array([1e-6])
    t = np.linspace(0.0, 8.0, 1601)
    pairs = _pairs_bare()
    Cs = fssh.run_correlations(epsilon, Delta, om, cc, 1.0, t, pairs,
                               quantum=True, n_traj=n_traj, seed=seed)
    worst = 0.0
    for (A, B), C in zip(pairs, Cs):
        ex = exact_1mode(epsilon, Delta, 1.0, 1e-6, 1e6, t, A, B)
        worst = max(worst, float(np.max(np.abs(C - ex))))
    return worst, tol


def test_single_mode(tol=0.13, n_traj=30000, seed=11):
    """Harmonic single mode; short-time window where MASH is near-exact.

    Catches nonadiabatic-coupling (Hy) and hop-rescaling sign errors, which
    produce O(1) deviations already by t~2.
    """
    epsilon, Delta, beta = 1.0, 1.0, 0.6
    rho0 = basis(2, 0, kind = "ketbra")
    X = Pauli("x")
    worst = 0.0
    for w0, c0 in [(4.0, 0.6), (4.0, 0.2), (0.5, 0.632)]:
        om, cc = np.array([w0]), np.array([c0])
        t = np.linspace(0.0, 4.0, 801)
        res = mash.run_observables(epsilon, Delta, om, cc, beta, t,
                                   quantum=True, n_traj=n_traj, seed=seed)
        ex_pop = exact_1mode(epsilon, Delta, w0, c0, beta, t, rho0, rho0)
        ex_cmu = exact_1mode(epsilon, Delta, w0, c0, beta, t, X @ rho0, X)
        ex_cmu = ex_cmu / ex_cmu[0]
        worst = max(worst,
                    float(np.max(np.abs(res["pop"][:, 0] - ex_pop.real))),
                    float(np.max(np.abs(res["Cmu"] - ex_cmu))))
    return worst, tol


def test_single_mode_anharmonic(tol=0.06, n_traj=30000, seed=11):
    """Quartic single mode (harmonic branch + bath anharmonicity)."""
    epsilon, Delta, beta = 1.0, 1.0, 0.6
    a = 0.2 * np.sqrt(7.0 / 6.0)
    rho0 = basis(2, 0, kind = "ketbra")
    worst = 0.0
    for w0, c0 in [(4.0, 0.6), (4.0, 0.4)]:
        om, cc = np.array([w0]), np.array([c0])
        t = np.linspace(0.0, 4.0, 801)
        res = mash.run_observables(epsilon, Delta, om, cc, beta, t,
                                   quantum=True, anharmon=a,
                                   n_traj=n_traj, seed=seed)
        ex_pop = exact_1mode(epsilon, Delta, w0, c0, beta, t, rho0, rho0, a=a)
        worst = max(worst,
                    float(np.max(np.abs(res["pop"][:, 0] - ex_pop.real))))
    return worst, tol


def test_single_mode_qsbm(tol=0.08, n_traj=30000, seed=11):
    """QSBM single mode (quadratic system-bath coupling)."""
    epsilon, Delta, beta = 1.0, 1.0, 0.6
    w0, c0 = 1.0, 0.5
    alpha0, alpha1, alpha2 = 0.5, 1.0, 0.3
    rho0 = basis(2, 0, kind = "ketbra")
    pv0, pv1, pv2 = epsilon + alpha0 / 2.0, alpha1 / 2.0, alpha2 / 2.0
    model = (pv0, pv1, pv2, -pv0, -pv1, -pv2)
    om, cc = np.array([w0]), np.array([c0])
    t = np.linspace(0.0, 4.0, 801)
    res = mash.run_observables(epsilon, Delta, om, cc, beta, t, quantum=True,
                               n_traj=n_traj, seed=seed, model=model)
    ex_pop = exact_1mode_qsbm(epsilon, Delta, w0, c0, beta, t, rho0, rho0,
                              alpha0=alpha0, alpha1=alpha1, alpha2=alpha2)
    return float(np.max(np.abs(res["pop"][:, 0] - ex_pop.real))), tol


TESTS = [
    ("bare electronic exactness (MASH estimator + sigma_y sign)", test_bare),
    ("bare electronic exactness (FSSH density matrix)", test_bare_fssh),
    ("single harmonic mode (NAC sign, hop rescaling)", test_single_mode),
    ("single quartic mode (anharmonic bath)", test_single_mode_anharmonic),
    ("single QSBM mode (quadratic coupling)", test_single_mode_qsbm),
]


def main():
    failed = False
    for name, fn in TESTS:
        err, tol = fn()
        ok = err <= tol
        failed |= not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: max err = {err:.4f} "
              f"(tol {tol})")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

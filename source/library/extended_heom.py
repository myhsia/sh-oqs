"""Extended HEOM (nonlinear) solver for polynomial system-bath coupling

    H = H_S + H_B + Q_S (a0 + a1 q + a2 q^2)

with a Gaussian (harmonic) bath.

The equations solved here are the extended (nonlinear) HEOM, implemented in
the extended DEOM / dissipaton representation of

    R.-X. Xu, Y. Liu, H.-D. Zhang, Y. Yan, J. Chem. Phys. 148, 114103 (2018)
    Z.-H. Chen, Y. Wang, R.-X. Xu, Y. Yan, J. Chem. Phys. 158, 074102 (2023)

For a Gaussian bath the extended DEOM and the extended HEOM are the same
hierarchy, only written in different bookkeeping (dissipaton density
operators vs. auxiliary density operators).  In particular, for a2 = 0 the
equation below reduces to the standard (linear) HEOM.

Eq. (18) of the latter:

    d rho_n / dt =
        -(i L_S + sum_k n_k g_k) rho_n
        - i (a0 + a2 <q^2>) A rho_n
        - i a1 sum_k ( A rho_{n+_k} + n_k C_k rho_{n-_k} )
        - 2 i a2 sum_{k k'} n_k C_k rho_{n-+_{k k'}}
        - i a2 sum_{k k'} ( A rho_{n++_{k k'}}
                            + n_k (n_k' - d_{k k'}) B_{k k'} rho_{n--_{k k'}} )

with

    A O          = [Q_S, O]
    C_k O        = eta_k Q_S O - eta^*_{kbar} O Q_S
    B_{k k'} O   = eta_k eta_k' Q_S O - eta^*_{kbar} eta^*_{kbar'} O Q_S

and the bath correlation function C(t) = <q(t) q(0)> = sum_k eta_k e^{-g_k t}
(conjugate-closed set, g_{kbar} = g_k^*).

The reduced density operator is rho_0 = rho^{(0)}_{0_1 ... 0_K}.  For a2 = 0
the equation reduces to the standard linear HEOM.
"""

# Extended HEOM reference for the spin-boson model with linear and/or
# quadratic system-bath coupling.
#
# qutip's official HEOM (OhmicEnvironment + HEOMSolver) only supports linear
# coupling.  For the quadratic (and mixed) cases we use the extended HEOM,
# implemented in the extended DEOM / dissipaton representation of
#     R.-X. Xu, Y. Liu, H.-D. Zhang, Y. Yan, J. Chem. Phys. 148, 114103 (2018)
#     Z.-H. Chen, Y. Wang, R.-X. Xu, Y. Yan, J. Chem. Phys. 158, 074102 (2023)
# (see ../library/extended_heom.py).  For a2 = 0 it reduces to the standard
# linear HEOM (validated against qutip's HEOMSolver).

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass


def _to_array(x):
    """Return a complex ndarray whether ``x`` is a qutip Qobj or an array."""
    full = getattr(x, "full", None)
    x = full() if callable(full) else x
    return np.asarray(x, dtype=complex)


@dataclass
class HEOMResult:
    """Minimal stand-in for :class:`qutip.Result` (``times``/``states``)."""
    times: np.ndarray
    states: list


# ---------------------------------------------------------------------------
# Bath correlation function
# ---------------------------------------------------------------------------
def build_ohmic_bath(alpha, wc, beta, tmax, Nr=6, method="espira-I",
                     nfit=2000, combine=True):
    """Exponential decomposition of the Ohmic bath correlation function.

    The environment is the qutip ``OhmicEnvironment`` with spectral density

        J(w) = alpha * w * exp(-w / wc)          (s = 1),

    whose correlation function equals

        C(t) = (1/pi) int dw e^{-i w t} J(w) / (1 - e^{-beta w})
             = <q(t) q(0)>.

    Returns
    -------
    eta, gamma : ndarray
        Complex coefficients and (conjugate-closed) exponents of C(t).
    q2 : float
        Thermal expectation <q^2> = C(0).
    env : OhmicEnvironment
    approx : ExponentialBosonicEnvironment
    info : dict
    """
    from qutip.core.environment import OhmicEnvironment

    env = OhmicEnvironment(T=1.0 / beta, alpha=alpha, wc=wc, s=1.0)
    tlist = np.linspace(0.0, tmax, nfit)
    approx, info = env.approximate(
        method=method, tlist=tlist, separate=False, Nr=Nr, combine=combine)

    eta = np.array([exp.coefficient for exp in approx.exponents], dtype=complex)
    gamma = np.array([exp.vk for exp in approx.exponents], dtype=complex)
    q2 = float(np.real(env.correlation_function(0.0)))
    return eta, gamma, q2, env, approx, info


def conjugate_index(gamma, rtol=1e-6, atol=1e-8):
    """For each exponent gamma_k return the index kbar with gamma_kbar ~ gamma_k^*."""
    gamma = np.asarray(gamma, dtype=complex)
    kbar = np.empty(len(gamma), dtype=int)
    for k, g in enumerate(gamma):
        d = np.abs(gamma - np.conj(g))
        j = int(np.argmin(d))
        if d[j] > atol + rtol * abs(g):
            raise ValueError(
                f"Exponent {g} has no conjugate partner in the decomposition.")
        kbar[k] = j
    return kbar


# ---------------------------------------------------------------------------
# Index book-keeping
# ---------------------------------------------------------------------------
def _enumerate_states(K, nmax):
    """All occupation vectors (n_1 ... n_K) with sum_k n_k <= nmax.

    The first state is always the all-zero (reduced system) state.
    """
    states = []

    def rec(prefix, remaining, slots):
        if slots == 1:
            states.append(tuple(prefix + [remaining]))
            return
        for v in range(remaining + 1):
            rec(prefix + [v], remaining - v, slots - 1)

    for total in range(nmax + 1):
        rec([], total, K)
    index = {s: i for i, s in enumerate(states)}
    return states, index


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class ExtendedHEOMSolver:
    """Extended HEOM (extended DEOM hierarchy) for a 2-level system.

    ``H`` and ``Qs`` may be qutip ``Qobj`` or plain arrays.  The interaction
    is ``Qs (a0 + a1 q + a2 q^2)``.  Setting ``a2 = 0`` reduces the hierarchy
    to the standard linear HEOM, i.e. this solver also covers the linear
    case; the driver scripts nevertheless route linear coupling to qutip's
    ``HEOMSolver`` via their ``if quadratic`` branch.
    """

    def __init__(self, H, Qs, alpha0, alpha1, alpha2, eta, gamma, q2,
                 max_depth, options=None, verbose=False):
        self.Hs = _to_array(H)
        self.Qs = _to_array(Qs)
        self.a0 = complex(alpha0)
        self.a1 = complex(alpha1)
        self.a2 = complex(alpha2)
        self.eta = np.asarray(eta, dtype=complex)
        self.gamma = np.asarray(gamma, dtype=complex)
        self.q2 = float(q2)
        self.nmax = int(max_depth)
        self.options = dict(options) if options else {}
        self.d = self.Hs.shape[0]
        assert self.Hs.shape == (self.d, self.d)
        assert self.Qs.shape == (self.d, self.d)
        assert self.d == 2, "Only implemented for a 2-level system."

        self.K = len(self.eta)
        self.kbar = conjugate_index(self.gamma)
        self.etac = np.conj(self.eta[self.kbar])

        self.states, self.index = _enumerate_states(self.K, self.nmax)
        self.nstates = len(self.states)
        self.dim = self.nstates * self.d * self.d
        self.Narr = np.array(self.states, dtype=np.int64)
        base = self.nmax + 2
        powers = base ** np.arange(self.K, dtype=np.int64)
        self._powers = powers
        self._code = self.Narr @ powers
        self._code_to_index = {int(c): i for i, c in enumerate(self._code)}
        if verbose:
            print(f"[ExtHEOM] K={self.K}, nmax={self.nmax}, "
                  f"nstates={self.nstates}, dim={self.dim}")

        self._build_superops()
        self._build_liouvillian()

    # -- superoperators ----------------------------------------------------
    def _build_superops(self):
        d = self.d
        I = np.eye(d, dtype=complex)
        Q = self.Qs
        # column-stacking convention: vec(A rho B) = (B^T (x) A) vec(rho)
        LQ = np.kron(I, Q)          # rho -> Q rho
        RQ = np.kron(Q.T, I)        # rho -> rho Q
        self.S_A = LQ - RQ          # [Q, .]
        self.S_C = [self.eta[k] * LQ - self.etac[k] * RQ
                    for k in range(self.K)]
        self.S_B = {}
        for k in range(self.K):
            for kp in range(self.K):
                self.S_B[(k, kp)] = (
                    self.eta[k] * self.eta[kp] * LQ
                    - self.etac[k] * self.etac[kp] * RQ)
        self.S_H = np.kron(I, self.Hs) - np.kron(self.Hs.T, I)  # [H, .]

    # -- hierarchy operator ------------------------------------------------
    def _lookup(self, arr):
        """Indices of state rows ``arr`` (shape (n, K)) in the hierarchy."""
        codes = arr @ self._powers
        return np.fromiter(
            (self._code_to_index[int(c)] for c in codes),
            dtype=np.int64, count=len(codes))

    @staticmethod
    def _edges(rows, cols, data, n):
        return sp.csr_matrix(
            (data, (rows, cols)), shape=(n, n), dtype=complex)

    def _build_liouvillian(self, dtype=complex):
        d2 = self.d * self.d
        ns = self.nstates
        K = self.K
        N = self.Narr
        nmax = self.nmax
        a1 = self.a1
        a2 = self.a2
        tot = N.sum(axis=1)
        ones = np.ones(ns)
        eye4 = sp.identity(d2, dtype=complex, format="csr")

        # diagonal: -i[H,.] - sum_k n_k gamma_k + mean field
        diag_n = -(N @ self.gamma)
        L = sp.kron(sp.diags(diag_n), eye4, format="csr")
        L = L + sp.kron(
            sp.identity(ns, format="csr", dtype=complex),
            (-1j * self.S_H - 1j * (self.a0 + a2 * self.q2) * self.S_A),
            format="csr")

        # ---- linear (a1) ----
        for k in range(K):
            # raising source: n + e_k  (target row n, source col n+e_k)
            valid = (tot + 1 <= nmax)
            rows = np.where(valid)[0]
            tgt = N[valid].copy()
            tgt[:, k] += 1
            cols = self._lookup(tgt)
            E = self._edges(rows, cols, ones[valid], ns)
            L = L + sp.kron(E, (-1j * a1) * self.S_A, format="csr")
            # lowering source: n - e_k
            valid = N[:, k] > 0
            rows = np.where(valid)[0]
            tgt = N[valid].copy()
            tgt[:, k] -= 1
            cols = self._lookup(tgt)
            E = self._edges(rows, cols, N[valid, k].astype(complex), ns)
            L = L + sp.kron(E, (-1j * a1) * self.S_C[k], format="csr")

        # ---- quadratic (a2) ----
        if a2 != 0.0:
            for k in range(K):
                for kp in range(K):
                    # n -+ : lower k, raise k'
                    valid = N[:, k] > 0
                    rows = np.where(valid)[0]
                    tgt = N[valid].copy()
                    tgt[:, k] -= 1
                    tgt[:, kp] += 1
                    cols = self._lookup(tgt)
                    E = self._edges(rows, cols,
                                    N[valid, k].astype(complex), ns)
                    L = L + sp.kron(E, (-2j * a2) * self.S_C[k],
                                    format="csr")
                    # n ++ : raise k, raise k'
                    valid = (tot + 2 <= nmax)
                    rows = np.where(valid)[0]
                    tgt = N[valid].copy()
                    tgt[:, k] += 1
                    tgt[:, kp] += 1
                    cols = self._lookup(tgt)
                    E = self._edges(rows, cols, ones[valid], ns)
                    L = L + sp.kron(E, (-1j * a2) * self.S_A, format="csr")
                    # n -- : lower k, lower k'
                    ck = N[:, k]
                    ckp = N[:, kp] - (1 if k == kp else 0)
                    valid = (ck > 0) & (ckp > 0)
                    rows = np.where(valid)[0]
                    tgt = N[valid].copy()
                    tgt[:, k] -= 1
                    tgt[:, kp] -= 1
                    cols = self._lookup(tgt)
                    data = (N[valid, k] * ckp[valid]).astype(complex)
                    E = self._edges(rows, cols, data, ns)
                    L = L + sp.kron(E, (-1j * a2) * self.S_B[(k, kp)],
                                    format="csr")

        self.L = L.tocsr()

    # -- propagation -------------------------------------------------------
    def _initial_vector(self, rho0):
        y = np.zeros(self.dim, dtype=complex)
        # column-stacking convention (Fortran order), consistent with the
        # kron-based superoperators.
        y[: self.d * self.d] = _to_array(rho0).flatten(order="F")
        return y

    def _reduce(self, y):
        return y[: self.d * self.d].reshape(self.d, self.d, order="F")

    def run(self, rho0, tlist, **kwargs):
        """Propagate the hierarchy and return a :class:`HEOMResult`.

        Parameters
        ----------
        rho0 : (d, d) ndarray or qutip Qobj
            Initial system density matrix (bath in thermal equilibrium).
        tlist : (Nt,) ndarray
            Output times (must start at 0).
        **kwargs
            Overrides merged with the ``options`` passed to the constructor.
            Recognized keys: ``chunk``, ``rtol``, ``atol``, ``method``,
            ``real_form``, ``progress`` (qutip-only keys such as ``nsteps``,
            ``store_states`` or ``progress_bar`` are ignored).

        Returns
        -------
        HEOMResult
            ``.states`` is a list of qutip ``Qobj`` reduced density matrices.
        """
        from scipy.integrate import solve_ivp

        opts = {**self.options, **kwargs}
        chunk     = opts.get("chunk", 200)
        rtol      = opts.get("rtol", 1e-9)
        atol      = opts.get("atol", 1e-11)
        method    = opts.get("method", "DOP853")
        real_form = opts.get("real_form", False)
        progress  = opts.get("progress", False)

        tlist = np.asarray(tlist, dtype=float)
        y = self._initial_vector(rho0)
        out = np.empty((len(tlist), self.d, self.d), dtype=complex)
        out[0] = self._reduce(y)
        dd = self.d * self.d

        if real_form:
            Lr = self.L.real.tocsr()
            Li = self.L.imag.tocsr()
            L = sp.bmat([[Lr, -Li], [Li, Lr]], format="csr")
            y = np.concatenate([y.real, y.imag])
            f = lambda t, x: L @ x  # noqa: E731
            jac = L

            def reduce_(vec):
                return (vec[:dd] + 1j * vec[self.dim:self.dim + dd]).reshape(
                    self.d, self.d, order="F")
        else:
            L = self.L
            f = lambda t, x: L @ x  # noqa: E731
            jac = L
            reduce_ = self._reduce

        # ``jac`` is only used by the implicit methods; passing it to an
        # explicit method triggers a scipy warning.
        kwargs = {} if method.upper() in ("RK45", "RK23", "DOP853") \
            else {"jac": jac}
        t0 = tlist[0]
        i = 1
        while i < len(tlist):
            j = min(i + chunk, len(tlist))
            sub = tlist[i:j]
            sol = solve_ivp(
                f, (t0, tlist[j - 1]), y, t_eval=sub, method=method,
                rtol=rtol, atol=atol, dense_output=False, **kwargs)
            if not sol.success:
                raise RuntimeError(f"Integration failed: {sol.message}")
            y = sol.y[:, -1]
            t0 = tlist[j - 1]
            for m in range(len(sub)):
                out[i + m] = reduce_(sol.y[:, m])
            if progress:
                print(f"  t = {t0:.3f} / {tlist[-1]:.3f}")
            i = j
        from qutip import Qobj
        return HEOMResult(times=tlist, states=[Qobj(r) for r in out])

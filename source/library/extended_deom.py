"""Extended DEOM solver for polynomial system-bath coupling

    H = H_S + H_B + Q_S (a0 + a1 q + a2 q^2)

with a Gaussian (harmonic) bath.

This is the *extended dissipaton equation of motion* (extended DEOM) of

    R.-X. Xu, Y. Liu, H.-D. Zhang, Y. Yan, J. Chem. Phys. 148, 114103 (2018)
    Z.-H. Chen, Y. Wang, R.-X. Xu, Y. Yan, J. Chem. Phys. 158, 074102 (2023)

Equation (18) of the latter, in the form

    d rho_n / dt =
        -(i L_S + sum_k n_k gamma_k) rho_n
        - i (a0 + a2 <q^2>) [Q_S, rho_n]
        - i a1 sum_k ( [Q_S, rho_{n+_k}] + n_k C_k rho_{n-_k} )
        - 2 i a2 sum_{k k'} n_k C_k rho_{n-+_{k k'}}
        - i a2 sum_{k k'} ( [Q_S, rho_{n++_{k k'}}]
                            + n_k (n_k' - d_{k k'}) B_{k k'} rho_{n--_{k k'}} )

    C_k O        = eta_k Q_S O - eta^*_{kbar} O Q_S
    B_{k k'} O   = eta_k eta_k' Q_S O - eta^*_{kbar} eta^*_{kbar'} O Q_S

and the bath correlation function C(t) = <q(t) q(0)> = sum_k eta_k e^{-gamma_k t}
(conjugate-closed set, gamma_{kbar} = gamma_k^*).  The reduced density operator
is rho_0.  For a2 = 0 the hierarchy reduces to the standard linear HEOM; the
driver nevertheless routes linear coupling to qutip's ``HEOMSolver``.

Numerical representation
------------------------
The hierarchy is stored in the dissipaton (square-root ladder) normalisation

    Y_n = rho_n / c_n,      c_n = sqrt( prod_k n_k! eta_k^{n_k} ),

because ``Y_n`` *shrinks* with the tier and therefore supports magnitude-based
truncation with a single absolute threshold (``ferr``).  The equations are
unchanged: every neighbour term ``S rho_{n'}`` acquires the ratio
``c_{n'}/c_n``, which is applied in closed form (no factorials are evaluated).

The right-hand side is matrix-free: no Liouvillian is assembled.  Each DDO has
a precomputed integer adjacency vector per neighbour family, and the RHS is a
gather/apply/scatter over ``(d, d)`` blocks.  This keeps the memory at
``O(nstates * K^2)`` integers plus ``O(nstates * d^2)`` complex numbers instead
of the ``O(nstates * K^2 * d^4)`` of a sparse Liouvillian.

The architecture follows the reference implementation ``moscal2.0``
(<https://git.lug.ustc.edu.cn/czh123/moscal2.0>, C++), whose
``rem_cal_gen``/``init_qmd_quad`` and ``filter_p`` were used as the model.
"""

import numpy as np
from dataclasses import dataclass
from scipy.special import gammaln


def _to_array(x):
    """Return a complex ndarray whether ``x`` is a qutip Qobj or an array."""
    full = getattr(x, "full", None)
    x = full() if callable(full) else x
    return np.asarray(x, dtype=complex)


@dataclass
class DEOMResult:
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
        Complex coefficients and (conjugate-closed) exponents of C(t).  The
        exponents come from qutip's fit; the coefficients are refitted to the
        full complex C(t) so that its imaginary (dissipative) part is included.
    q2 : float
        Thermal expectation <q^2>, taken from the fitted kernel.
    env : OhmicEnvironment
    approx : ExponentialBosonicEnvironment
    info : dict
    """
    from qutip.core.environment import OhmicEnvironment

    env = OhmicEnvironment(T=1.0 / beta, alpha=alpha, wc=wc, s=1.0)
    tlist = np.linspace(0.0, tmax, nfit)
    # ``separate=False`` gives a conjugate-closed exponent set, which is what the
    # hierarchy needs, but its amplitudes reproduce only the real part of C(t).
    # The amplitudes are therefore refitted (least squares, complex) to the full
    # complex correlation function; the exponents stay conjugate closed.
    approx, info = env.approximate(
        method=method, tlist=tlist, separate=False, Nr=Nr, combine=combine)
    gamma = np.array([exp.vk for exp in approx.exponents], dtype=complex)

    tfit = np.linspace(0.0, tmax, nfit)
    cfun = np.array([env.correlation_function(t) for t in tfit], dtype=complex)
    basis = np.exp(-np.outer(tfit, gamma))
    eta, *_ = np.linalg.lstsq(basis, cfun, rcond=None)
    fit_err = float(np.abs(basis @ eta - cfun).max())
    if fit_err > 0.05 * float(np.abs(cfun).max()):
        raise RuntimeError(
            f"bath decomposition is inaccurate: max|dC| = {fit_err:.3e}; "
            "increase Nr")

    # the mean field uses the corner of the fitted kernel, for consistency
    q2 = float(np.real(eta.sum()))
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
def enumerate_states(K, nmax):
    """All occupation vectors (n_1 ... n_K) with sum_k n_k <= nmax.

    The first state is always the all-zero (reduced system) state.
    """
    K = int(K)
    if K == 1:
        keys = np.arange(nmax + 1, dtype=np.int32)[:, None]
        return keys
    # odometer over a bounded composition of nmax
    rows = []
    for total in range(nmax + 1):
        def rec(prefix, remaining, slots):
            if slots == 1:
                rows.append(prefix + [remaining])
                return
            for v in range(remaining + 1):
                rec(prefix + [v], remaining - v, slots - 1)
        rec([], total, K)
    return np.asarray(rows, dtype=np.int32)


def _key_codes(keys, nmax):
    """Mixed-radix code for each key row.

    The base is chosen larger than the largest digit that can occur in a
    *shifted* key (``nmax + 2``), so the encoding is injective.  Out-of-range
    keys are filtered out by the caller before encoding.
    """
    base = int(nmax) + 3
    return keys.astype(np.int64) @ (base ** np.arange(keys.shape[1], dtype=np.int64))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class ExtendedDEOMSolver:
    """Extended DEOM hierarchy for a 2-level system.

    ``H`` and ``Qs`` may be qutip ``Qobj`` or plain arrays.  The interaction is
    ``Qs (a0 + a1 q + a2 q^2)``.  Setting ``a2 = 0`` reduces the hierarchy to
    the standard linear HEOM; the driver scripts nevertheless route linear
    coupling to qutip's ``HEOMSolver`` via their ``if quadratic`` branch.

    Options (all optional, passed as ``options=`` or to :meth:`run`):

    ``method``
        ``"rk4"`` (default) fixed-step vectorised Runge-Kutta, or
        ``"dop853"`` for a scipy cross-check.
    ``dt``
        RK4 step.  Defaults to the output spacing, clamped to the stability
        limit ``2.5 / max|gamma|``.
    ``ferr``
        Magnitude threshold for dropping DDOs (``None`` disables filtering).
    ``filter_every``
        Apply the filter every this many output points (default 50).
    ``prune_frac``
        Rebuild the mapping only when at least this fraction is dead
        (default 0.1), so that a stage does not pay for a rebuild.
    ``verbose``
        Print the hierarchy size and the filtering events.
    """

    #: neighbour families, in the order used by ``_build_adjacency``
    FAMILIES = ("up", "dn", "pp", "pm", "mm")

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
        self.verbose = verbose or bool(self.options.get("verbose", False))
        self.d = self.Hs.shape[0]
        assert self.Hs.shape == (self.d, self.d)
        assert self.Qs.shape == (self.d, self.d)
        assert self.d == 2, "Only implemented for a 2-level system."

        self.K = len(self.eta)
        self.kbar = conjugate_index(self.gamma)
        self.etac = np.conj(self.eta[self.kbar])
        # The qutip ``separate=False`` fit assigns the amplitude to one member of
        # each conjugate pair and leaves the partner at zero.  A zero-amplitude
        # dissipaton cannot be rescaled, so it keeps the neutral factor 1
        # (``log_eta = 0``), which makes the change of variables the identity for
        # its ladder steps while staying exact for the others.
        self.zero_eta = np.abs(self.eta) == 0
        self.log_eta = np.zeros(self.K, dtype=complex)
        nz = ~self.zero_eta
        self.log_eta[nz] = np.log(self.eta[nz])

        self.keys = enumerate_states(self.K, self.nmax)
        self.nstates = self.keys.shape[0]
        self.dim = self.nstates * self.d * self.d
        self.tot = self.keys.sum(axis=1)

        self._build_ops()
        self._build_adjacency()
        self.bound = self._spectral_bound()

    # -- operators ---------------------------------------------------------
    def _build_ops(self):
        """Left/right ``(d, d)`` factors such that ``S O = L @ O - O @ R``."""
        Q = self.Qs
        K = self.K
        self.L_A, self.R_A = Q, Q
        self.L_C = [self.eta[k] * Q for k in range(K)]
        self.R_C = [self.etac[k] * Q for k in range(K)]
        self.L_B = {}
        self.R_B = {}
        for k in range(K):
            for kp in range(K):
                self.L_B[(k, kp)] = self.eta[k] * self.eta[kp] * Q
                self.R_B[(k, kp)] = self.etac[k] * self.etac[kp] * Q
        # mean-field shift (a0 + a2 <q^2>) Qs
        self.renorm = (self.a0 + self.a2 * self.q2) * Q

    # -- hierarchy graph ---------------------------------------------------
    def _build_adjacency(self):
        """Integer neighbour arrays, ``-1`` where the neighbour is absent."""
        K = self.K
        n = self.nstates
        nmax = self.nmax
        keys = self.keys
        tot = self.tot

        code = _key_codes(keys, nmax)
        order = np.argsort(code)
        sorted_code = code[order]

        def lookup(codes):
            pos = np.searchsorted(sorted_code, codes)
            pos = np.clip(pos, 0, len(sorted_code) - 1)
            found = sorted_code[pos] == codes
            return np.where(found, order[pos], -1).astype(np.int32)

        def shift(k, dk, extra=None):
            tgt = keys.copy()
            tgt[:, k] += dk
            if extra is not None:
                tgt[:, extra[0]] += extra[1]
            ok = (tgt >= 0).all(axis=1) & (tgt.sum(axis=1) <= nmax)
            out = np.full(n, -1, dtype=np.int32)
            if ok.any():
                out[ok] = lookup(_key_codes(tgt[ok], nmax))
            return out

        self.idx = {f: {} for f in self.FAMILIES}
        self.idx["up"] = [shift(k, +1) for k in range(K)]
        self.idx["dn"] = [shift(k, -1) for k in range(K)]
        self.idx["pp"] = {}
        self.idx["pm"] = {}
        self.idx["mm"] = {}
        for k in range(K):
            for kp in range(K):
                self.idx["pp"][(k, kp)] = shift(k, +1, (kp, +1))
                self.idx["pm"][(k, kp)] = shift(k, -1, (kp, +1))
                self.idx["mm"][(k, kp)] = shift(k, -1, (kp, -1))

        # structural validity: built from the occupation numbers, so the
        # lookup may still land on a state outside the truncated set (-1).
        self._build_coefficients()
        self._build_edges()
        if self.verbose:
            print(f"[ExtDEOM] K={K}, nmax={nmax}, nstates={n}, dim={self.dim}, "
                  f"edges={self.nedges}")

    def _build_coefficients(self):
        """Effective coefficients in the dissipaton normalisation.

        Each neighbour term of the plain hierarchy contributes
        ``plain(n, n') * c_{n'} / c_n`` with ``c_n = sqrt(prod_k n_k! eta_k^{n_k})``.
        The ratio is written in the closed form

            log(c_{n'}/c_n) = 1/2 sum_k [log(n_k+1)...(n'_k) - (n'_k-n_k) log eta_k]

        i.e. from the integer step only, so that no factorial is evaluated and
        no large quantity is subtracted.
        """
        K = self.K
        n = self.keys.astype(float)
        le = self.log_eta
        eps = 1e-300

        def logpos(x):
            return np.log(np.maximum(x, eps))

        self.coef = {f: {} for f in self.FAMILIES}
        # up (n' = n + e_k): plain 1, ratio sqrt((n_k+1) eta_k)
        self.coef["up"] = [
            np.exp(0.5 * (logpos(n[:, k] + 1.0) + le[k])) for k in range(K)]
        # dn (n' = n - e_k): plain n_k, ratio 1/sqrt(n_k eta_k)
        self.coef["dn"] = [
            np.where(n[:, k] > 0,
                     np.exp(0.5 * (logpos(n[:, k]) - le[k])),
                     0.0) for k in range(K)]

        for k in range(K):
            for kp in range(K):
                nk, nkp = n[:, k], n[:, kp]
                same = (k == kp)
                la = le[k] + le[kp]
                if same:
                    # n' = n +- 2 e_k, or n' = n for the mixed family
                    f_pp = np.exp(0.5 * (logpos(nk + 1.0) + logpos(nk + 2.0) + la))
                    f_pm = nk                      # ratio 1, plain n_k
                    s_mm = nk * (nk - 1.0)
                    f_mm = np.where(s_mm > 0,
                                    np.exp(0.5 * (logpos(s_mm) - la)), 0.0)
                else:
                    f_pp = np.exp(0.5 * (logpos(nk + 1.0) + logpos(nkp + 1.0) + la))
                    # plain n_k, ratio sqrt((n_kp+1) eta_kp / (n_k eta_k))
                    f_pm = np.exp(0.5 * (logpos(nk) + logpos(nkp + 1.0)
                                         - le[k] + le[kp]))
                    s_mm = nk * nkp
                    f_mm = np.where(s_mm > 0,
                                    np.exp(0.5 * (logpos(s_mm) - la)), 0.0)
                self.coef["pp"][(k, kp)] = f_pp
                # the n-+ family carries the explicit factor 2 of Eq. (18)
                self.coef["pm"][(k, kp)] = 2.0 * f_pm
                self.coef["mm"][(k, kp)] = f_mm

    # -- right-hand side ---------------------------------------------------
    def _build_edges(self):
        """Compressed (source, target, left weight, right weight) per family.

        Only structurally valid neighbours are stored, so the RHS work is
        proportional to the number of edges of the truncated hierarchy and not
        to ``K^2`` times the number of states.  ``wa``/``wb`` already include
        the bath operators' ``eta`` factors; the ``-i a`` prefactor is applied
        in :meth:`_rhs`.
        """
        K = self.K
        E, EC = self.eta, self.etac
        self.edges = {f: {} for f in self.FAMILIES}
        self.nedges = 0

        def build(fam, key, idx, coef, a, b):
            m = idx >= 0
            rows = np.where(m)[0]
            self.edges[fam][key] = (rows, idx[m], a * coef[m], b * coef[m])
            self.nedges += rows.size

        for k in range(K):
            build("up", k, self.idx["up"][k], self.coef["up"][k], 1.0, 1.0)
            build("dn", k, self.idx["dn"][k], self.coef["dn"][k], E[k], EC[k])
        for k in range(K):
            for kp in range(K):
                build("pp", (k, kp), self.idx["pp"][(k, kp)],
                      self.coef["pp"][(k, kp)], 1.0, 1.0)
                build("pm", (k, kp), self.idx["pm"][(k, kp)],
                      self.coef["pm"][(k, kp)], E[k], EC[k])
                build("mm", (k, kp), self.idx["mm"][(k, kp)],
                      self.coef["mm"][(k, kp)], E[k] * E[kp], EC[k] * EC[kp])

    def _assemble(self):
        """Assemble the hierarchy generator as a CSR matrix from the edges.

        Written in the column-stacking convention ``vec(A X B) = (B^T kron A)``
        with ``vec`` taken per DDO, so a term ``w (L X - X R)`` contributes
        ``w (L kron I) - w (I kron R^T)``.  Assembling from the edge lists
        avoids the ``kron`` temporaries of a dense-block construction.
        """
        import scipy.sparse as sp

        d, dd = self.d, self.d * self.d
        I = np.eye(d, dtype=complex)
        A = np.kron(I, self.Qs)                    # vec(Q X) = (I kron Q)
        B = -np.kron(self.Qs.T, I)                 # vec(X Q) = (Q^T kron I)
        nzA = np.argwhere(np.abs(A) > 0)
        nzB = np.argwhere(np.abs(B) > 0)

        blocks = {"up": -1j * self.a1, "dn": -1j * self.a1,
                  "pp": -1j * self.a2, "pm": -1j * self.a2,
                  "mm": -1j * self.a2}
        data, rows, cols = [], [], []
        for fam, c in blocks.items():
            if c == 0:
                continue
            for r_, c_, wa, wb in self.edges[fam].values():
                if r_.size == 0:
                    continue
                for (a, b) in nzA:
                    data.append(c * wa * A[a, b])
                    rows.append(r_ * dd + a)
                    cols.append(c_ * dd + b)
                for (a, b) in nzB:
                    data.append(c * wb * B[a, b])
                    rows.append(r_ * dd + a)
                    cols.append(c_ * dd + b)

        # diagonal: -sum(n gamma) - i[H,.] - i[renorm,.]
        diag = -(self.keys @ self.gamma)
        Sd = (-1j * (np.kron(I, self.Hs) - np.kron(self.Hs.T, I))
              - 1j * (np.kron(I, self.renorm) - np.kron(self.renorm.T, I)))
        nvec = np.arange(self.nstates)
        for a in range(dd):
            for b in range(dd):
                v = Sd[a, b]
                if a == b:
                    v = v + diag
                if np.any(v):
                    data.append(np.broadcast_to(v, (self.nstates,)).astype(complex))
                    rows.append(nvec * dd + a)
                    cols.append(nvec * dd + b)

        self.L = sp.coo_matrix(
            (np.concatenate(data),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.dim, self.dim), dtype=complex).tocsr()
        self.L.sum_duplicates()

    def _spectral_bound(self):
        """Gershgorin bound on ``|eigenvalue|`` of the hierarchy generator.

        Used to pick a stable RK4 step: ``dt <= 2.5 / bound``.
        """
        K = self.K
        r = np.abs(self.keys @ self.gamma).astype(float)

        def add(idx, coef, L, R):
            m = idx >= 0
            if m.any():
                r[m] += np.abs(coef[m]) * (np.abs(L).sum() + np.abs(R).sum())

        if self.a1 != 0:
            for k in range(K):
                add(self.idx["up"][k], self.coef["up"][k], self.L_A, self.R_A)
                add(self.idx["dn"][k], self.coef["dn"][k],
                    self.L_C[k], self.R_C[k])
        if self.a2 != 0:
            for k in range(K):
                for kp in range(K):
                    add(self.idx["pp"][(k, kp)], self.coef["pp"][(k, kp)],
                        self.L_A, self.R_A)
                    add(self.idx["pm"][(k, kp)], self.coef["pm"][(k, kp)],
                        self.L_C[k], self.R_C[k])
                    add(self.idx["mm"][(k, kp)], self.coef["mm"][(k, kp)],
                        self.L_B[(k, kp)], self.R_B[(k, kp)])
        return float(np.max(r)) if r.size else 0.0

    def _rhs(self, Y):
        """Matrix-free ``dY/dt`` for ``Y`` of shape ``(nstates, d, d)``.

        Every operator of the hierarchy is a scalar multiple of ``Q_S``, so
        ``Q_S @ Y`` and ``Y @ Q_S`` are formed once and reused by all
        neighbour families.  Each edge then costs two gathers instead of two
        matrix products.
        """
        H = self.Hs
        gam_n = self.keys @ self.gamma
        dY = -gam_n[:, None, None] * Y
        dY -= 1j * (np.matmul(H, Y) - np.matmul(Y, H))
        dY -= 1j * (np.matmul(self.renorm, Y) - np.matmul(Y, self.renorm))

        QY = np.matmul(self.Qs, Y)
        YQ = np.matmul(Y, self.Qs)

        if self.a1 != 0:
            c = -1j * self.a1
            for fam in ("up", "dn"):
                for rows, cols, wa, wb in self.edges[fam].values():
                    if rows.size:
                        dY[rows] += (c * wa)[:, None, None] * QY[cols] \
                                    - (c * wb)[:, None, None] * YQ[cols]

        if self.a2 != 0:
            c = -1j * self.a2
            for fam in ("pp", "pm", "mm"):
                for rows, cols, wa, wb in self.edges[fam].values():
                    if rows.size:
                        dY[rows] += (c * wa)[:, None, None] * QY[cols] \
                                    - (c * wb)[:, None, None] * YQ[cols]

        return dY

    # -- propagation -------------------------------------------------------
    def _initial(self, rho0):
        Y = np.zeros((self.nstates, self.d, self.d), dtype=complex)
        # c_0 = 1, so the reduced DDO is the initial system density matrix
        Y[0] = _to_array(rho0)
        return Y

    def run(self, rho0, tlist, **kwargs):
        """Propagate the hierarchy and return a :class:`DEOMResult`.

        Parameters
        ----------
        rho0 : (d, d) ndarray or qutip Qobj
            Initial system density matrix (bath in thermal equilibrium).
        tlist : (Nt,) ndarray
            Output times (must start at 0).
        **kwargs
            Overrides merged with the ``options`` passed to the constructor.

        Returns
        -------
        DEOMResult
            ``.states`` is a list of qutip ``Qobj`` reduced density operators.
        """
        opts = {**self.options, **kwargs}
        method = str(opts.get("method", "dop853")).lower()
        ferr = opts.get("ferr", None)
        filter_every = int(opts.get("filter_every", 50))
        prune_frac = float(opts.get("prune_frac", 0.1))

        tlist = np.asarray(tlist, dtype=float)
        Y = self._initial(rho0)
        # mask of live DDOs, always keeping the reduced one
        alive = np.ones(self.nstates, dtype=bool)

        out = np.empty((len(tlist), self.d, self.d), dtype=complex)
        out[0] = Y[0]

        if method == "dop853":
            self._run_scipy(Y, tlist, out, opts)
        else:
            for i in range(1, len(tlist)):
                span = tlist[i] - tlist[i - 1]
                h_max = opts.get("dt") or min(span, 2.5 / max(self.bound, 1e-12))
                nsub = max(1, int(np.ceil(span / h_max)))
                h = span / nsub
                for _ in range(nsub):
                    Y = self._rk4_step(Y, h)
                out[i] = Y[0]
                if ferr is not None and (i % filter_every == 0):
                    Y = self._prune(Y, ferr, prune_frac)

        from qutip import Qobj
        return DEOMResult(times=tlist, states=[Qobj(r) for r in out])

    def _rk4_step(self, Y, h):
        k1 = self._rhs(Y)
        k2 = self._rhs(Y + 0.5 * h * k1)
        k3 = self._rhs(Y + 0.5 * h * k2)
        k4 = self._rhs(Y + h * k3)
        return Y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _run_scipy(self, Y, tlist, out, opts):
        """Adaptive ``solve_ivp`` on the assembled hierarchy generator.

        The generator is built from the edge lists on first use (per-DDO
        column stacking), and the march is stepped in chunks of output points
        so that the solution array stays bounded.
        """
        from scipy.integrate import solve_ivp

        shape = Y.shape

        def to_vec(A):
            return A.transpose(0, 2, 1).reshape(-1)

        def from_vec(v):
            return v.reshape(shape).transpose(0, 2, 1)

        flat = to_vec(Y)
        chunk = int(opts.get("chunk", 200))
        method = opts.get("method_scipy", "DOP853")
        rtol = opts.get("rtol", 1e-9)
        atol = opts.get("atol", 1e-11)

        L = getattr(self, "L", None)
        if L is None:
            self._assemble()
            L = self.L

        def f(t, x):
            return L @ x

        y = flat
        i = 1
        while i < len(tlist):
            j = min(i + chunk, len(tlist))
            sub = tlist[i:j]
            sol = solve_ivp(f, (tlist[i - 1], tlist[j - 1]), y, t_eval=sub,
                            method=method, rtol=rtol, atol=atol)
            if not sol.success:
                raise RuntimeError(f"Integration failed: {sol.message}")
            y = sol.y[:, -1]
            for m in range(len(sub)):
                out[i + m] = from_vec(sol.y[:, m])[0]
            i = j

    # -- filtering ---------------------------------------------------------
    def _prune(self, Y, ferr, prune_frac):
        """Drop DDOs below ``ferr`` and rebuild the hierarchy (in place)."""
        keep = (np.abs(Y).max(axis=(1, 2)) > ferr)
        keep[0] = True
        dead = 1.0 - keep.mean()
        if dead <= prune_frac:
            return Y
        if self.verbose:
            print(f"[ExtDEOM] prune {self.nstates - int(keep.sum())} of "
                  f"{self.nstates} DDOs (ferr={ferr:g})")
        # rebuild the truncated hierarchy on the survivor keys
        sub = ExtendedDEOMSolver.__new__(ExtendedDEOMSolver)
        sub.__dict__.update(self.__dict__)
        sub.keys = self.keys[keep]
        sub.nstates = sub.keys.shape[0]
        sub.dim = sub.nstates * self.d * self.d
        sub.tot = sub.keys.sum(axis=1)
        sub._build_adjacency()
        self.__dict__.update(sub.__dict__)
        return Y[keep].copy()

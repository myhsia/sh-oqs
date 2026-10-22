"""Mapping approach to surface hopping (MASH) for the two-level spin-boson.

Reference: J. R. Mannouch and J. O. Richardson, "A mapping approach to surface
hopping", J. Chem. Phys. 158, 104111 (2023); arXiv:2212.11773.

Working equations (their Eqs. 20, 21, 24-30), with spin radius ``rs = 1`` and
adiabatic (plus/minus) representation of the electronic system::

    energy      E = sum_j (w_j/2) p_j^2 + Vbar + V_z sgn(Sz)
    force       F_j = -dV_z/dq_j sgn(Sz)   (+ bath force)
    spin EOM    dS/dt = H x S,  H = (0, 2 sum_j d_j (p_j/m_j), 2 V_z)
    observables sigma_x -> Sx, sigma_y -> Sy, P_pm -> h(+-Sz)
    weights     W = 3 (CC), 2 (CP/PC), 2|Sz| (PP)

Here ``V_z = (V_+ - V_-)/2`` is the adiabatic half-gap and ``d_j`` the
nonadiabatic coupling vector.  Hops are deterministic (crossing ``Sz = 0``)
and rescale the momentum along the NAC direction to conserve energy.
"""

import os, sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from numba import njit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qutils import sample_bath, Pauli, basis, qubit
from surface_hopping.fssh import _rescale_momentum, _sizes, _spawn_seeds


@njit(cache=True, fastmath=True)
def _mash_traj_numba(Delta, omega, c, q, p, S0, t, aq,
                     pv0, pv1, pv2, pa0, pa1, pa2, meanfield):
    """Adiabatic model: ``Vbar(Q) = pv0 + pv1 Q + pv2 Q^2`` and
    ``A(Q) = pa0 + pa1 Q + pa2 Q^2`` with ``Q = sum_j c_j q_j``; the adiabatic
    half-gap is ``Vz = sqrt(A^2 + Delta^2)`` and ``phi = atan2(Delta, A)``.
    ``aq`` is the quartic bath coefficient (a^2).  LSBM corresponds to
    ``(0,0,0, epsilon, 1, 0)``."""
    nT = t.shape[0]
    dt = t[1] - t[0]
    n = omega.shape[0]
    P_plus = np.empty(nT); P_minus = np.empty(nT)
    Sx = np.empty(nT); Sy = np.empty(nT); Sz = np.empty(nT); Qh = np.empty(nT)
    E = np.empty(nT)

    cc2 = 0.0
    for j in range(n):
        cc2 += c[j] * c[j]
    cn = np.sqrt(cc2)
    chat = np.empty(n)
    chatM = 0.0
    for j in range(n):
        chat[j] = c[j] / cn
        chatM += chat[j] * chat[j] / omega[j]

    Sx0 = S0[0]; Sy0 = S0[1]; Sz0 = S0[2]
    for it in range(nT):
        Q = 0.0
        for j in range(n):
            Q += c[j] * q[j]
        A = pa0 + pa1 * Q + pa2 * Q * Q
        R = np.sqrt(A * A + Delta * Delta)
        if R < 1e-12:
            R = 1e-12
        sinphi = Delta / R
        Vz = R
        dAdQ = pa1 + 2.0 * pa2 * Q
        Hy = 0.0
        for j in range(n):
            Hy += c[j] * sinphi * dAdQ / (2.0 * R) * omega[j] * p[j]
        Hy *= 2.0
        Qh[it] = Q
        # total energy: bath (harmonic + quartic) + Vbar + Vz sgn(Sz)
        e = Vz if Sz0 > 0.0 else -Vz
        e += pv0 + pv1 * Q + pv2 * Q * Q
        for j in range(n):
            e += 0.5 * omega[j] * (p[j] * p[j] + q[j] * q[j]) \
                 + 0.25 * omega[j] * aq * q[j] ** 4
        E[it] = e
        if Sz0 > 0.0:
            P_plus[it] = 1.0; P_minus[it] = 0.0
        else:
            P_plus[it] = 0.0; P_minus[it] = 1.0
        Sx[it] = Sx0; Sy[it] = Sy0; Sz[it] = Sz0
        if it == nT - 1:
            break

        # spin RK4
        h = 0.5 * dt
        k1x = Hy * Sz0 - 2.0 * Vz * Sy0; k1y = 2.0 * Vz * Sx0; k1z = -Hy * Sx0
        a1x = Sx0 + h * k1x; a1y = Sy0 + h * k1y; a1z = Sz0 + h * k1z
        k2x = Hy * a1z - 2.0 * Vz * a1y; k2y = 2.0 * Vz * a1x; k2z = -Hy * a1x
        a2x = Sx0 + h * k2x; a2y = Sy0 + h * k2y; a2z = Sz0 + h * k2z
        k3x = Hy * a2z - 2.0 * Vz * a2y; k3y = 2.0 * Vz * a2x; k3z = -Hy * a2x
        a3x = Sx0 + dt * k3x; a3y = Sy0 + dt * k3y; a3z = Sz0 + dt * k3z
        k4x = Hy * a3z - 2.0 * Vz * a3y; k4y = 2.0 * Vz * a3x; k4z = -Hy * a3x
        nSx = Sx0 + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        nSy = Sy0 + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
        nSz = Sz0 + dt / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z)
        nn = np.sqrt(nSx * nSx + nSy * nSy + nSz * nSz)
        nSx /= nn; nSy /= nn; nSz /= nn

        # deterministic hop at the equator (Eq. 25 rescaling)
        if (not meanfield) and Sz0 * nSz < 0.0:
            pt = 0.0
            for j in range(n):
                pt += c[j] * p[j]
            pt /= cn
            sgn = 1.0 if pt * Sx0 > 0.0 else -1.0
            dE = -2.0 * Vz * sgn
            # kinetic energy / projection along chat (mass m_j = 1/omega_j)
            kin = 0.0; coef = 0.0
            for j in range(n):
                u = omega[j] * p[j]
                kin += u * u / (2.0 * omega[j])
                coef += chat[j] * u / omega[j]
            coef /= chatM
            kin_par = 0.0
            for j in range(n):
                up = coef * chat[j]
                kin_par += up * up / (2.0 * omega[j])
            kin_perp = kin - kin_par
            target = kin - dE
            ok = False
            if kin_par > 0.0:
                kap2 = (target - kin_perp) / kin_par
                if kap2 >= 0.0:
                    kap = np.sqrt(kap2)
                    for j in range(n):
                        u = omega[j] * p[j]
                        up = coef * chat[j]
                        u = (u - up) + kap * up
                        p[j] = u / omega[j]
                    ok = True
            if not ok:
                for j in range(n):
                    u = omega[j] * p[j]
                    up = coef * chat[j]
                    u = (u - up) - up
                    p[j] = u / omega[j]
                nSz = -nSz

        Sx0 = nSx; Sy0 = nSy; Sz0 = nSz

        # nuclear velocity-Verlet on the sgn(Sz) surface:
        # F_j = -dVbar/dq_j - dVz/dq_j sgn(Sz) - w_j q_j - aq w_j q_j^3,
        # with dVz/dq_j = (A/Vz) dA/dQ c_j.
        # spin-LSC (mean field) uses the continuous Sz; MASH uses sgn(Sz)
        sgnz = Sz0 if meanfield else (1.0 if Sz0 > 0.0 else -1.0)
        for half in range(2):
            Q = 0.0
            for j in range(n):
                Q += c[j] * q[j]
            A = pa0 + pa1 * Q + pa2 * Q * Q
            R = np.sqrt(A * A + Delta * Delta)
            dVdQ = pv1 + 2.0 * pv2 * Q
            dAdQ = pa1 + 2.0 * pa2 * Q
            for j in range(n):
                F = (-c[j] * dVdQ - c[j] * (A / R) * dAdQ * sgnz
                     - omega[j] * (q[j] + aq * q[j] ** 3))
                p[j] += 0.5 * dt * F
            if half == 0:
                for j in range(n):
                    q[j] += dt * (omega[j] * p[j])

    return P_plus, P_minus, Sx, Sy, Sz, Qh, E


def _model_default(epsilon):
    """LSBM model coefficients ``(pv0,pv1,pv2,pa0,pa1,pa2)``."""
    return (0.0, 0.0, 0.0, epsilon, 1.0, 0.0)


def _phi_of_Q(Delta, Q, model):
    """Adiabatic mixing angle at collective coordinate ``Q``."""
    pa0, pa1, pa2 = model[3], model[4], model[5]
    A = pa0 + pa1 * Q + pa2 * Q * Q
    return np.arctan2(Delta, A)


def _adiabatic(Delta, c, q, model):
    """Return ``Q, Vbar, A, Vz, cosphi, sinphi`` at the geometry ``q``."""
    Q = float(np.dot(c, q))
    pv0, pv1, pv2, pa0, pa1, pa2 = model
    Vbar = pv0 + pv1 * Q + pv2 * Q * Q
    A = pa0 + pa1 * Q + pa2 * Q * Q
    R = np.hypot(A, Delta)
    if R < 1e-12:
        R = 1e-12
    return Q, Vbar, A, R, A / R, Delta / R


def _mash_trajectory_py(Delta, omega, c, q, p, S, tlist, model, anharmon=0.0):
    """Propagate one MASH trajectory; returns adiabatic populations/coherences.

    Returns
    -------
    P_plus, P_minus : (nT,) ndarray  (Heaviside measurement, Eq. 27c)
    Sx, Sy, Sz : (nT,) ndarray
    """
    omega = np.asarray(omega, dtype=float)
    c = np.asarray(c, dtype=float)
    nT = len(tlist)
    dt = tlist[1] - tlist[0]
    a2 = float(anharmon) ** 2
    q = np.array(q, dtype=float, copy=True)
    p = np.array(p, dtype=float, copy=True)
    S = np.array(S, dtype=float, copy=True)

    chat = c / np.sqrt(np.sum(c * c))
    chat_norm_M = float(np.sum(chat * chat / omega))

    P_plus = np.empty(nT); P_minus = np.empty(nT)
    Sx = np.empty(nT); Sy = np.empty(nT); Sz = np.empty(nT)
    Q_hist = np.empty(nT)

    def spin_rhs(Sv, Hy, Vz):
        return np.array([Hy * Sv[2] - 2.0 * Vz * Sv[1],
                         2.0 * Vz * Sv[0],
                         -Hy * Sv[0]])

    for it in range(nT):
        Q, Vbar, A, R, cosphi, sinphi = _adiabatic(Delta, c, q, model)
        Vz = R
        dAdQ = model[4] + 2.0 * model[5] * Q
        d_j = c * sinphi * dAdQ / (2.0 * R)
        Hy = 2.0 * np.sum(d_j * omega * p)
        # measurement (Eq. 27): hysteresis-free signs at the equator
        P_plus[it] = 1.0 if S[2] > 0 else 0.0
        P_minus[it] = 1.0 if S[2] < 0 else 0.0
        Sx[it], Sy[it], Sz[it] = S
        Q_hist[it] = Q
        if it == nT - 1:
            break

        # --- spin propagation (RK4, H frozen at current geometry) ---
        k1 = spin_rhs(S, Hy, Vz)
        k2 = spin_rhs(S + 0.5 * dt * k1, Hy, Vz)
        k3 = spin_rhs(S + 0.5 * dt * k2, Hy, Vz)
        k4 = spin_rhs(S + dt * k3, Hy, Vz)
        S_new = S + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        # keep on the unit sphere
        S_new = S_new / np.linalg.norm(S_new)

        # --- hop when the equator is crossed (deterministic) ---
        if S[2] * S_new[2] < 0.0:
            Vz_h = R
            # MASH rescaling (Eq. 25): the nuclear energy changes by
            # +2 Vz sgn(p~_init Sx); _rescale_momentum takes the electronic
            # energy change, hence the minus sign.
            pt = float(np.dot(c, p) / np.sqrt(np.sum(c * c)))
            sgn = 1.0 if pt * S[0] > 0 else -1.0
            dE = -2.0 * Vz_h * sgn
            if _rescale_momentum(p, omega, chat, chat_norm_M, dE):
                pass
            else:
                # frustrated hop: reflect p~ and Sz (Eqs. 23)
                u = omega * p
                coef = np.sum(chat * u / omega) / chat_norm_M
                u = (u - coef * chat) - coef * chat
                p[:] = u / omega
                S_new[2] = -S_new[2]
        S = S_new

        # --- nuclear velocity-Verlet step on the active (sgn(Sz)) surface ---
        sgnz = 1.0 if S[2] > 0 else -1.0
        Q, Vbar, A, R, cosphi, sinphi = _adiabatic(Delta, c, q, model)
        dVdQ = model[1] + 2.0 * model[2] * Q
        dAdQ = model[4] + 2.0 * model[5] * Q
        F = -c * dVdQ - c * (A / R) * dAdQ * sgnz - omega * (q + a2 * q ** 3)
        p = p + 0.5 * dt * F
        q = q + dt * (omega * p)
        Q, Vbar, A, R, cosphi, sinphi = _adiabatic(Delta, c, q, model)
        dVdQ = model[1] + 2.0 * model[2] * Q
        dAdQ = model[4] + 2.0 * model[5] * Q
        F = -c * dVdQ - c * (A / R) * dAdQ * sgnz - omega * (q + a2 * q ** 3)
        p = p + 0.5 * dt * F

    return P_plus, P_minus, Sx, Sy, Sz, Q_hist


def sample_full_sphere(n, rng):
    """Uniform full-sphere sampling (Eq. 29)."""
    cost = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    st = np.sqrt(1.0 - cost ** 2)
    return np.stack([st * np.cos(phi), st * np.sin(phi), cost], axis=1)


def run_mash(epsilon, Delta, omega, c, beta, tlist, init_hemisphere=None,
             quantum=True, anharmon=0.0, n_traj=500, rng=None, seed=None):
    """MASH ensemble.  If ``init_hemisphere`` is +1/-1, trajectories are
    sampled full-sphere but restricted to that hemisphere (for a pure
    adiabatic population initial condition).

    Returns ``(P_plus_avg, P_minus_avg)`` normalised by the hemisphere weight.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    nT = len(tlist)
    acc_p = np.zeros(nT); acc_m = np.zeros(nT); norm_p = 0.0; norm_m = 0.0
    done = 0
    while done < n_traj:
        m = min(n_traj - done, 1000)
        S_all = sample_full_sphere(m, rng)
        q_all, p_all = sample_bath(omega, beta, quantum=quantum, n_traj=m, rng=rng)
        for i in range(m):
            S0 = S_all[i]
            if init_hemisphere is not None:
                if init_hemisphere > 0 and S0[2] <= 0:
                    continue
                if init_hemisphere < 0 and S0[2] >= 0:
                    continue
            Pp, Pm, _, _, _, _ = mash_trajectory(epsilon, Delta, omega, c, q_all[i],
                                              p_all[i], S0, tlist,
                                              anharmon=anharmon)
            w = 2.0 * abs(S0[2])
            acc_p += w * Pp; acc_m += w * Pm
            norm_p += w if init_hemisphere in (None, 1) else 0.0
            norm_m += w if init_hemisphere in (None, -1) else 0.0
        done += m
    P_plus = acc_p / norm_p if norm_p > 0 else acc_p * np.nan
    P_minus = acc_m / norm_m if norm_m > 0 else acc_m * np.nan
    return P_plus, P_minus


def mash_trajectory(epsilon, Delta, omega, c, q, p, S, tlist, anharmon=0.0,
                    model=None, meanfield=False, return_energy=False):
    """Numba-accelerated MASH trajectory (see ``_mash_trajectory_py``).

    ``model`` is an optional ``(pv0,pv1,pv2,pa0,pa1,pa2)`` coefficient tuple
    (default: LSBM with bias ``epsilon``).
    """
    if model is None:
        model = _model_default(epsilon)
    pv0, pv1, pv2, pa0, pa1, pa2 = model
    q = np.ascontiguousarray(q, dtype=float)
    p = np.ascontiguousarray(p, dtype=float)
    S = np.ascontiguousarray(S, dtype=float)
    out = _mash_traj_numba(Delta, np.ascontiguousarray(omega, dtype=float),
                            np.ascontiguousarray(c, dtype=float), q, p, S,
                            np.ascontiguousarray(tlist, dtype=float),
                            float(anharmon) ** 2,
                            pv0, pv1, pv2, pa0, pa1, pa2, bool(meanfield))
    return out if return_energy else out[:6]


def run_density_matrix(epsilon, Delta, omega, c, beta, tlist, rho_S_dia,
                       quantum=True, anharmon=0.0, n_traj=1000, rng=None, seed=None,
                       model=None):
    """Ensemble MASH diabatic reduced density matrix ``rho(t)`` for a factorised
    initial condition ``rho_S_dia (x) rho_B``.

    Implements the Eq.(32) density-matrix estimator with the Eq.(30) weights,
    decomposed into the adiabatic basis ``{P+, P-, sigma_x, sigma_y}`` and the
    Eq.(29) sphere measure (factor 2 on the uniform-sphere Monte Carlo average).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if model is None:
        model = _model_default(epsilon)
    rho_S_dia = np.asarray(rho_S_dia, dtype=complex)
    nT = len(tlist)
    acc = np.zeros((nT, 2, 2), dtype=complex)
    m = 0
    while m < n_traj:
        k = min(n_traj - m, 2000)
        S = sample_full_sphere(k, rng)
        q, p = sample_bath(omega, beta, quantum=quantum, n_traj=k, rng=rng)
        for i in range(k):
            Pp, Pm, Sx, Sy, Sz, Q = mash_trajectory(epsilon, Delta, omega, c, q[i], p[i], S[i], tlist,
                                                    anharmon=anharmon, model=model,
                                                    meanfield=meanfield)
            phi0 = _phi_of_Q(Delta, Q[0], model); cp0, sp0 = np.cos(phi0/2), np.sin(phi0/2)
            U0 = np.array([[cp0, -sp0], [sp0, cp0]])
            r0 = U0.conj().T @ rho_S_dia @ U0
            aPp, aPm = r0[0, 0], r0[1, 1]
            aCx = 0.5*(r0[0, 1]+r0[1, 0]); aCy = (r0[0, 1]-r0[1, 0])/(2j)
            sz0, sx0 = S[i, 2], S[i, 0]
            fP = (aPp if sz0 > 0 else 0.0) + (aPm if sz0 < 0 else 0.0)   # P-part of A
            fC = aCx*sx0 + aCy*S[i, 1]                                    # C-part of A
            wPP = 2.0*abs(sz0)
            hP = (Sz > 0).astype(float); hM = (Sz < 0).astype(float)
            popP = wPP*(hP+hM)                                            # =wPP
            # adiabatic components of rho(t)
            rpp = fP*wPP*hP + fC*2.0*hP
            rmm = fP*wPP*hM + fC*2.0*hM
            rpm = (fP*2.0 + fC*3.0)*(Sx-1j*Sy)/2
            phi = _phi_of_Q(Delta, Q, model); cp, sp = np.cos(phi/2), np.sin(phi/2)
            # transform adiabatic -> diabatic per time step
            U00, U01, U10, U11 = cp, -sp, sp, cp
            d00 = U00*U00.conj()*rpp + U00*U01.conj()*rpm + U01*U00.conj()*np.conj(rpm) + U01*U01.conj()*rmm
            d01 = U00*U10.conj()*rpp + U00*U11.conj()*rpm + U01*U10.conj()*np.conj(rpm) + U01*U11.conj()*rmm
            acc[:, 0, 0] += d00
            acc[:, 0, 1] += d01
            acc[:, 1, 0] += np.conj(d01)
            acc[:, 1, 1] += (rpp+rmm) - d00
        m += k
    return 2.0 * acc / n_traj


def _adiabatic_coeffs(B, c, s):
    """Coefficients {P+,P-,sx,sy} of operator B in the adiabatic basis
    (rotation by half-angle c=cos(phi/2), s=sin(phi/2); arrays allowed)."""
    B00, B01, B10, B11 = B[0, 0], B[0, 1], B[1, 0], B[1, 1]
    r00 = c*c*B00 + c*s*B01 + s*c*B10 + s*s*B11
    r01 = -c*s*B00 + c*c*B01 - s*s*B10 + s*c*B11
    r10 = -s*c*B00 - s*s*B01 + c*c*B10 + c*s*B11
    r11 = s*s*B00 - s*c*B01 - c*s*B10 + c*c*B11
    return r00, r11, 0.5*(r01+r10), (r10-r01)/(2j)


def _unique_ops(ops):
    """Deduplicate a list of operators; return ``(unique, index)``."""
    seen = {}
    uniq = []
    idx = []
    for op in ops:
        op = np.asarray(op, dtype=complex)
        key = op.tobytes()
        if key not in seen:
            seen[key] = len(uniq)
            uniq.append(op)
        idx.append(seen[key])
    return uniq, idx


def _accumulate_mash(epsilon, Delta, omega, c, beta, tlist, pairs, model,
                     quantum, anharmon, rng, n_traj, meanfield=False):
    """Unnormalised MASH estimator sum over ``n_traj`` trajectories."""
    A_list, A_idx = _unique_ops([p[0] for p in pairs])
    B_list, B_idx = _unique_ops([p[1] for p in pairs])
    nPairs = len(pairs)
    nT = len(tlist)
    acc = np.zeros((nPairs, nT), dtype=complex)
    # Eq.(30) weight table indexed by {P+,P-,sx,sy}: 3 (CC), 2 (CP/PC), 2|Sz| (PP)
    cat = np.array([0, 0, 1, 1])                      # 0 = population, 1 = coherence
    base = np.where(cat[:, None] == cat[None, :], 3.0, 2.0)  # CC=3 else 2
    pp_mask = np.outer(cat == 0, cat == 0)            # PP -> 2|Sz0| (overrides)

    m = 0
    while m < n_traj:
        k = min(n_traj - m, 2000)
        S = sample_full_sphere(k, rng)
        q, p = sample_bath(omega, beta, quantum=quantum, n_traj=k, rng=rng)
        for i in range(k):
            Pp, Pm, Sx, Sy, Sz, Q = mash_trajectory(epsilon, Delta, omega, c, q[i], p[i], S[i], tlist,
                                                    anharmon=anharmon, model=model,
                                                    meanfield=meanfield)
            phi = _phi_of_Q(Delta, Q, model)
            ch, sh = np.cos(phi/2), np.sin(phi/2)
            phi0 = phi[0]; c0, s0 = np.cos(phi0/2), np.sin(phi0/2)
            sz0 = S[i, 2]; sx0 = S[i, 0]; sy0 = S[i, 1]
            w = base.copy()
            w[pp_mask] = 2.0*abs(sz0)
            f = np.array([1.0 if sz0 > 0 else 0.0, 1.0 if sz0 < 0 else 0.0, sx0, sy0])
            hP = (Sz > 0).astype(float); hM = (Sz < 0).astype(float)
            g = np.stack([hP, hM, Sx, Sy])               # (4, nT)
            a_cache = [_adiabatic_coeffs(A_list[j], c0, s0) for j in range(len(A_list))]
            b_cache = [_adiabatic_coeffs(B_list[j], ch, sh) for j in range(len(B_list))]
            for ip in range(nPairs):
                a = a_cache[A_idx[ip]]                   # 4 scalars
                b = b_cache[B_idx[ip]]                   # 4 (nT,) arrays
                inner = np.array([sum(a[kk]*w[kk, ll]*f[kk] for kk in range(4))
                                  for ll in range(4)], dtype=complex)
                acc[ip] += sum(inner[ll]*b[ll]*g[ll] for ll in range(4))
        m += k
    return acc


def _slice_mash(args):
    """One ensemble slice (module level, so the pool worker is picklable)."""
    (epsilon, Delta, omega, c, beta, tlist, pairs, model,
     quantum, anharmon, seed, m, meanfield) = args
    return _accumulate_mash(epsilon, Delta, omega, c, beta, tlist, pairs, model,
                            quantum, anharmon, np.random.default_rng(seed), m,
                            meanfield)


def run_correlations(epsilon, Delta, omega, c, beta, tlist, pairs,
                     quantum=True, anharmon=0.0, n_traj=4000, rng=None, seed=None,
                     model=None, n_jobs=1, meanfield=False):
    """MASH correlations ``C_AB(t) = Tr[B U(t) A_op (x) rho_B]`` for many
    operator pairs in a *single* trajectory ensemble.

    Parameters
    ----------
    pairs : sequence of ``(A_op, B_op)`` 2x2 arrays (diabatic basis)
        ``A_op`` sets the time-zero electronic operator, ``B_op`` the measured
        one (Eq. 28).  ``B`` is measured at each ``tlist`` entry.
    n_jobs : int
        ``> 1`` runs independent slices in parallel processes (deterministic
        per-slice seeds); ``1`` uses the serial loop.

    Returns
    -------
    list of (nT,) complex ndarray, one per pair (unnormalised).
    """
    if model is None:
        model = _model_default(epsilon)
    nPairs = len(pairs)

    if n_jobs > 1 and n_traj > 1:
        sizes = _sizes(n_traj, 4 * n_jobs)
        seeds = _spawn_seeds(rng, seed, len(sizes))
        args = [(epsilon, Delta, omega, c, beta, tlist, pairs, model,
                 quantum, anharmon, seeds[i], m, meanfield)
                for i, m in enumerate(sizes) if m > 0]
        with ProcessPoolExecutor(max_workers = n_jobs,
                mp_context = multiprocessing.get_context("fork")) as pool:
            parts = list(pool.map(_slice_mash, args))
        acc = np.sum(parts, axis = 0)
        return [2.0 * acc[ip] / n_traj for ip in range(nPairs)]

    if rng is None:
        rng = np.random.default_rng(seed)
    acc = _accumulate_mash(epsilon, Delta, omega, c, beta, tlist, pairs, model,
                           quantum, anharmon, rng, n_traj, meanfield)
    return [2.0 * acc[ip] / n_traj for ip in range(nPairs)]


def run_correlation(epsilon, Delta, omega, c, beta, tlist, A_op, B_op,
                    quantum=True, anharmon=0.0, n_traj=4000, rng=None, seed=None,
                    model=None):
    """MASH correlation ``C_AB(t) = Tr[B U(t) A_op (x) rho_B]`` for a factorised
    initial condition, using the Eq.(28) estimator with the Eq.(30) weights.

    ``A_op`` and ``B_op`` are arbitrary 2x2 system operators (in the diabatic
    basis).  The result is the *unnormalised* correlation; normalise as needed.
    """
    return run_correlations(epsilon, Delta, omega, c, beta, tlist, [(A_op, B_op)],
                            quantum=quantum, anharmon=anharmon,
                            n_traj=n_traj, rng=rng, seed=seed, model=model)[0]


def _standard_pairs():
    """The (A_op, B_op) pairs matching the MKCT/HEOM observables.

    ``A_op`` is the time-zero operator (``Brho0`` in ``mkct_matrix.py``) and
    ``B_op`` the measured one, both in the diabatic basis.
    """
    rho0 = basis(2, 0, kind = "ketbra")
    rho1 = basis(2, 1, kind = "ketbra")
    X, Y = Pauli("x"), Pauli("y")
    Sp, Sm = Pauli("+"), Pauli("-")
    chi = qubit(theta = 2.0 * np.pi / 3.0, kind = "ketbra")   # |chi><chi|
    return {
        "pop0":  (rho0, rho0),
        "pop1":  (rho0, rho1),
        "cohe_p": (rho0, Sp),
        "cohe_m": (rho0, Sm),
        "Cmu":   (X @ rho0, X),
        "Cx":    (Y @ chi, X),
    }


def run_observables(epsilon, Delta, omega, c, beta, tlist,
                    quantum=True, anharmon=0.0, n_traj=4000, rng=None, seed=None,
                    model=None, n_jobs=1, meanfield=False):
    """All MKCT observables in one MASH ensemble.

    ``model`` selects the adiabatic model (default LSBM); see
    ``mash_trajectory``.
    """
    pairs = _standard_pairs()
    keys = ["pop0", "pop1", "cohe_p", "cohe_m", "Cmu", "Cx"]
    Cs = run_correlations(epsilon, Delta, omega, c, beta, tlist, [pairs[k] for k in keys],
                          quantum=quantum, anharmon=anharmon,
                          n_traj=n_traj, rng=rng, seed=seed, model=model,
                          n_jobs=n_jobs, meanfield=meanfield)
    C = dict(zip(keys, Cs))
    Cmu = C["Cmu"]
    if Cmu[0] != 0:
        Cmu = Cmu / Cmu[0]
    return {
        "pop": np.column_stack((C["pop0"].real, C["pop1"].real)),
        "cohe_p": C["cohe_p"],
        "cohe_m": C["cohe_m"],
        "Cmu": Cmu,
        "Cx": C["Cx"],
    }

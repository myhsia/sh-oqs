"""Fewest-switches surface hopping (FSSH) for the two-level spin-boson model.

The electronic (2-level) subsystem is quantum; the bath modes are classical
nuclei.  The electronic problem is solved in the diabatic basis (exact 2x2
propagator for frozen ``H_el``), while hopping probabilities use the adiabatic
representation.

General adiabatic model (as for MASH)::

    Vbar(Q) = pv0 + pv1 Q + pv2 Q^2
    A(Q)    = pa0 + pa1 Q + pa2 Q^2,     Q = sum_j c_j q_j
    H_el    = Vbar I + A sigz + Delta sigx
    Vz      = hypot(A, Delta),   phi = atan2(Delta, A)
    |+> = (cos(phi/2), sin(phi/2)),   E_+ = Vbar + Vz   (active = 0)
    |-> = (-sin(phi/2), cos(phi/2)),  E_- = Vbar - Vz   (active = 1)

LSBM corresponds to ``(0,0,0, epsilon,1,0)``.
"""

import os, sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from numba import njit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qutils import sample_bath


@njit(cache=True, fastmath=True)
def _fssh_traj_numba(Delta, omega, c, q, p, psi, t, aq,
                     pv0, pv1, pv2, pa0, pa1, pa2, seed):
    nT = t.shape[0]
    dt = t[1] - t[0]
    n = omega.shape[0]
    rho = np.zeros((nT, 2, 2), dtype=np.complex128)
    np.random.seed(seed)

    cc2 = 0.0
    for j in range(n):
        cc2 += c[j] * c[j]
    cn = np.sqrt(cc2)
    chat = np.empty(n)
    chatM = 0.0
    for j in range(n):
        chat[j] = c[j] / cn
        chatM += chat[j] * chat[j] / omega[j]

    q = q.copy(); p = p.copy()
    psi0 = psi[0]; psi1 = psi[1]

    Q = 0.0
    for j in range(n):
        Q += c[j] * q[j]
    A = pa0 + pa1 * Q + pa2 * Q * Q
    phi = np.arctan2(Delta, A)
    hp = 0.5 * phi
    v00 = np.cos(hp); v01 = np.sin(hp)
    p0 = (v00 * psi0 + v01 * psi1)
    pr = p0.real * p0.real + p0.imag * p0.imag
    active = 0 if pr >= np.random.random() else 1

    for it in range(nT):
        rho[it, 0, 0] = psi0 * np.conj(psi0)
        rho[it, 0, 1] = psi0 * np.conj(psi1)
        rho[it, 1, 0] = psi1 * np.conj(psi0)
        rho[it, 1, 1] = psi1 * np.conj(psi1)
        if it == nT - 1:
            break

        Q = 0.0
        for j in range(n):
            Q += c[j] * q[j]
        Vbar = pv0 + pv1 * Q + pv2 * Q * Q
        A = pa0 + pa1 * Q + pa2 * Q * Q
        R = np.sqrt(A * A + Delta * Delta)
        phi = np.arctan2(Delta, A)
        hp = 0.5 * phi
        ch = np.cos(hp); sh = np.sin(hp)

        # electronic step: exp(-i H dt) with H = Vbar I + [[A,Delta],[Delta,-A]]
        cd = np.cos(R * dt); sd = np.sin(R * dt)
        mp0 = A * psi0 + Delta * psi1
        mp1 = Delta * psi0 - A * psi1
        phv = np.cos(Vbar * dt) - 1j * np.sin(Vbar * dt)
        npsi0 = phv * (cd * psi0 - 1j * sd * mp0 / R)
        npsi1 = phv * (cd * psi1 - 1j * sd * mp1 / R)
        psi0 = npsi0; psi1 = npsi1

        chi0 = ch * psi0 + sh * psi1
        chi1 = -sh * psi0 + ch * psi1

        # fewest-switches hopping
        dAdQ = pa1 + 2.0 * pa2 * Q
        Qdot = 0.0
        for j in range(n):
            Qdot += c[j] * omega[j] * p[j]
        D01 = np.sin(phi) * dAdQ * Qdot / (2.0 * R)
        prob = 0.0
        if active == 0:
            a0 = chi0.real * chi0.real + chi0.imag * chi0.imag
            if a0 > 0.0:
                tc = chi0 * np.conj(chi1) * D01
                prob = -2.0 * tc.real * dt / a0
        else:
            a1 = chi1.real * chi1.real + chi1.imag * chi1.imag
            if a1 > 0.0:
                tc = chi1 * np.conj(chi0) * (-D01)
                prob = -2.0 * tc.real * dt / a1
        if prob > np.random.random():
            new_active = 1 - active
            dE = -2.0 * R if new_active == 1 else 2.0 * R
            # _rescale_momentum (mass m_j = 1/omega_j)
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
                        p[j] = ((u - up) + kap * up) / omega[j]
                    ok = True
            if ok:
                active = new_active

        # nuclear velocity-Verlet on the active surface
        sgn = 1.0 if active == 0 else -1.0
        for half in range(2):
            Q = 0.0
            for j in range(n):
                Q += c[j] * q[j]
            dVdQ = pv1 + 2.0 * pv2 * Q
            A = pa0 + pa1 * Q + pa2 * Q * Q
            R = np.sqrt(A * A + Delta * Delta)
            dAdQ = pa1 + 2.0 * pa2 * Q
            for j in range(n):
                F = (-omega[j] * (q[j] + aq * q[j] ** 3)
                     - c[j] * (dVdQ + (A / R) * dAdQ * sgn))
                p[j] += 0.5 * dt * F
            if half == 0:
                for j in range(n):
                    q[j] += dt * (omega[j] * p[j])

    return rho


def _default_model(epsilon):
    """LSBM model coefficients ``(pv0,pv1,pv2,pa0,pa1,pa2)``."""
    return (0.0, 0.0, 0.0, epsilon, 1.0, 0.0)


def _phi_of_Q(Delta, Q, model):
    """Adiabatic mixing angle at collective coordinate ``Q``."""
    pa0, pa1, pa2 = model[3], model[4], model[5]
    A = pa0 + pa1 * Q + pa2 * Q * Q
    return np.arctan2(Delta, A)


def _initial_active(psi, phi, rng):
    """Sample the initial active surface from the adiabatic populations."""
    hp = 0.5 * phi
    v0 = np.array([np.cos(hp), np.sin(hp)])
    p0 = abs(v0 @ psi) ** 2
    return 0 if p0 >= rng.random() else 1


def fssh_trajectory(epsilon, Delta, omega, c, q, p, psi, tlist,
                    anharmon=0.0, rng=None, model=None, seed=None):
    """Propagate one FSSH trajectory (numba-accelerated).

    ``model`` is an optional ``(pv0,pv1,pv2,pa0,pa1,pa2)`` coefficient tuple
    (default: LSBM with bias ``epsilon``).  Returns ``(rho, 0)`` where ``rho`` is
    the ``(nT, 2, 2)`` diabatic density matrix per step.
    """
    if model is None:
        model = _default_model(epsilon)
    if seed is None:
        if rng is None:
            rng = np.random.default_rng()
        seed = int(rng.integers(0, 2 ** 31 - 1))
    pv0, pv1, pv2, pa0, pa1, pa2 = model
    rho = _fssh_traj_numba(
        Delta, np.ascontiguousarray(omega, dtype=float),
        np.ascontiguousarray(c, dtype=float),
        np.ascontiguousarray(q, dtype=float),
        np.ascontiguousarray(p, dtype=float),
        np.ascontiguousarray(psi, dtype=complex),
        np.ascontiguousarray(tlist, dtype=float),
        float(anharmon) ** 2, pv0, pv1, pv2, pa0, pa1, pa2, int(seed))
    return rho, 0


def _rescale_momentum(p, omega, chat, chat_norm_M, dE):
    """Rescale the momentum component along the NAC direction ``chat`` to
    conserve energy (mass-weighted, ``m_j = 1/w_j``).  Returns False if the
    available kinetic energy is insufficient (frustrated hop)."""
    u = omega * p                                   # physical velocity dq/dt
    kin = np.sum(u * u / (2.0 * omega))
    coef = np.sum(chat * u / omega) / chat_norm_M   # M-projection of u on chat
    u_par = coef * chat
    kin_par = np.sum(u_par * u_par / (2.0 * omega))
    kin_perp = kin - kin_par
    target = kin - dE
    if kin_par <= 0.0:
        return False
    kappa2 = (target - kin_perp) / kin_par
    if kappa2 < 0.0:
        return False
    kappa = np.sqrt(kappa2)
    u = (u - u_par) + kappa * u_par
    p[:] = u / omega
    return True


def _spawn_seeds(rng, seed, n):
    """``n`` independent integer seeds (drawn from ``rng`` when one is given)."""
    if rng is not None:
        seed = int(rng.integers(0, 2 ** 63 - 1))
    return [int(s.generate_state(1, dtype=np.uint32)[0])
            for s in np.random.SeedSequence(seed).spawn(n)]


def _sizes(n_traj, n_slices):
    """Split ``n_traj`` into (almost) equal slice sizes."""
    n_slices = max(1, min(int(n_traj), int(n_slices)))
    base, extra = divmod(int(n_traj), n_slices)
    return [base + (1 if i < extra else 0) for i in range(n_slices)]


def _slice_dm(args):
    """One ensemble slice (module level, so the worker is picklable)."""
    (epsilon, Delta, omega, c, beta, tlist, psi0, seed, m,
     quantum, anharmon, model) = args
    rng = np.random.default_rng(seed)
    nT = len(tlist)
    rho_sum = np.zeros((nT, 2, 2), dtype=complex)
    n_frustrated = 0
    q_all, p_all = sample_bath(omega, beta, quantum=quantum, n_traj=m, rng=rng)
    for i in range(m):
        rho, nf = fssh_trajectory(epsilon, Delta, omega, c, q_all[i], p_all[i],
                                  psi0, tlist, anharmon=anharmon, rng=rng,
                                  model=model)
        rho_sum += rho
        n_frustrated += nf
    return rho_sum, n_frustrated


def _slice_corr(args):
    """One correlation slice; returns ``{pair index: unnormalised series}``."""
    (epsilon, Delta, omega, c, beta, tlist, states, tasks, seed, m,
     quantum, anharmon, model) = args
    rng = np.random.default_rng(seed)
    nT = len(tlist)
    rho_s = [np.zeros((nT, 2, 2), dtype=complex) for _ in states]
    q_all, p_all = sample_bath(omega, beta, quantum=quantum, n_traj=m, rng=rng)
    for i in range(m):
        for s, psi in enumerate(states):
            rho, _ = fssh_trajectory(epsilon, Delta, omega, c, q_all[i], p_all[i],
                                     psi, tlist, anharmon=anharmon, rng=rng,
                                     model=model)
            rho_s[s] += rho
    acc = {}
    for (s, wgt, ip, B) in tasks:
        measured = np.einsum("tij,ji->t", rho_s[s], B)
        acc[ip] = acc.get(ip, 0) + wgt * measured
    return acc


def run_trajectories(epsilon, Delta, omega, c, beta, tlist, psi0,
                     quantum=True, anharmon=0.0, n_traj=500, chunk=100,
                     rng=None, seed=None, model=None, n_jobs=1):
    """Run an FSSH ensemble and return the averaged diabatic density matrix.

    ``n_jobs > 1`` splits the ensemble into independent slices run in parallel
    processes (deterministic per-slice seeds); ``n_jobs == 1`` uses the serial
    loop.

    Returns
    -------
    rho_avg : (nT, 2, 2) complex ndarray
    n_frustrated : int
    """
    if model is None:
        model = _default_model(epsilon)
    if n_jobs > 1 and n_traj > 1:
        sizes = _sizes(n_traj, 4 * n_jobs)
        seeds = _spawn_seeds(rng, seed, len(sizes))
        args = [(epsilon, Delta, omega, c, beta, tlist, psi0, seeds[i], m,
                 quantum, anharmon, model)
                for i, m in enumerate(sizes) if m > 0]
        with ProcessPoolExecutor(max_workers = n_jobs,
                mp_context = multiprocessing.get_context("fork")) as pool:
            parts = list(pool.map(_slice_dm, args))
        rho_sum = np.sum([p[0] for p in parts], axis = 0)
        return rho_sum / n_traj, int(sum(p[1] for p in parts))

    if rng is None:
        rng = np.random.default_rng(seed)
    nT = len(tlist)
    rho_sum = np.zeros((nT, 2, 2), dtype=complex)
    n_frustrated = 0
    done = 0
    while done < n_traj:
        m = min(chunk, n_traj - done)
        q_all, p_all = sample_bath(omega, beta, quantum=quantum,
                                   n_traj=m, rng=rng)
        for i in range(m):
            rho, nf = fssh_trajectory(epsilon, Delta, omega, c, q_all[i], p_all[i],
                                      psi0, tlist, anharmon=anharmon,
                                      rng=rng, model=model)
            rho_sum += rho
            n_frustrated += nf
        done += m
    return rho_sum / n_traj, n_frustrated


def run_correlations(epsilon, Delta, omega, c, beta, tlist, pairs,
                     quantum=True, anharmon=0.0, n_traj=4000, chunk=100,
                     rng=None, seed=None, model=None, n_jobs=1):
    """FSSH correlation functions ``C_AB(t) = Tr[B U(t) A_op (x) rho_B]``.

    The time-zero operator ``A_op`` need not be Hermitian: it is split into
    Hermitian parts ``A_op = A_R + i A_I``, each diagonalised and propagated
    from its eigenstates with signed weights.  Bath samples are shared across
    the initial electronic states of each chunk.

    ``n_jobs > 1`` runs independent slices in parallel processes; the workers
    return unnormalised sums and the parent divides by ``n_traj``.

    Returns a list of (nT,) complex arrays, one per pair (unnormalised).
    """
    if model is None:
        model = _default_model(epsilon)
    nT = len(tlist)
    pairs = [(np.asarray(A, dtype=complex), np.asarray(B, dtype=complex))
             for A, B in pairs]

    # Decompose each A_op into Hermitian eigenpairs, accumulating signed
    # (initial state, weight, pair, measured operator) tasks.
    state_map = {}
    states = []
    tasks = []
    for ip, (A, _B) in enumerate(pairs):
        AR = 0.5 * (A + A.conj().T)
        AI = (A - A.conj().T) / (2j)
        for M, w0 in ((AR, 1.0), (AI, 1j)):
            if np.max(np.abs(M)) < 1e-14:
                continue
            lam, vec = np.linalg.eigh(M)
            for k in range(2):
                if abs(lam[k]) < 1e-14:
                    continue
                v = np.ascontiguousarray(vec[:, k], dtype=complex)
                key = v.tobytes()
                if key not in state_map:
                    state_map[key] = len(states)
                    states.append(v)
                tasks.append((state_map[key], w0 * lam[k], ip, pairs[ip][1]))

    acc = np.zeros((len(pairs), nT), dtype=complex)

    if n_jobs > 1 and n_traj > 1:
        sizes = _sizes(n_traj, 4 * n_jobs)
        seeds = _spawn_seeds(rng, seed, len(sizes))
        args = [(epsilon, Delta, omega, c, beta, tlist, states, tasks,
                 seeds[i], m, quantum, anharmon, model)
                for i, m in enumerate(sizes) if m > 0]
        with ProcessPoolExecutor(max_workers = n_jobs,
                mp_context = multiprocessing.get_context("fork")) as pool:
            parts = list(pool.map(_slice_corr, args))
        for part in parts:
            for ip, series in part.items():
                acc[ip] += series
        return [acc[ip] / n_traj for ip in range(len(pairs))]

    if rng is None:
        rng = np.random.default_rng(seed)
    done = 0
    while done < n_traj:
        k = min(chunk, n_traj - done)
        q_all, p_all = sample_bath(omega, beta, quantum=quantum,
                                   n_traj=k, rng=rng)
        rho_s = [np.zeros((nT, 2, 2), dtype=complex) for _ in states]
        for i in range(k):
            for s, psi in enumerate(states):
                rho, _ = fssh_trajectory(epsilon, Delta, omega, c, q_all[i], p_all[i],
                                         psi, tlist, anharmon=anharmon,
                                         rng=rng, model=model)
                rho_s[s] += rho
        for (s, wgt, ip, B) in tasks:
            rho_avg = rho_s[s] / n_traj
            measured = np.einsum("tij,ji->t", rho_avg, B)
            acc[ip] += wgt * measured
        done += k
    return [acc[ip] for ip in range(len(pairs))]


def run_observables(epsilon, Delta, omega, c, beta, tlist,
                    quantum=True, anharmon=0.0, n_traj=4000, chunk=100,
                    rng=None, seed=None, model=None, n_jobs=1):
    """All MKCT observables via FSSH, mirroring ``mash.run_observables``.

    Returns a dict with ``pop`` (nT,2 real), ``cohe_p``, ``cohe_m``, ``Cmu``,
    ``Cx`` (nT complex).  ``Cmu`` is normalised so ``Cmu[0] == 1``.
    """
    from surface_hopping.mash import _standard_pairs
    pairs = _standard_pairs()
    keys = ["pop0", "pop1", "cohe_p", "cohe_m", "Cmu", "Cx"]
    Cs = run_correlations(epsilon, Delta, omega, c, beta, tlist, [pairs[k] for k in keys],
                          quantum=quantum, anharmon=anharmon, n_traj=n_traj,
                          chunk=chunk, rng=rng, seed=seed, model=model,
                          n_jobs=n_jobs)
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

# %%
# 1. Requirements
import os, sys, argparse, multiprocessing
from concurrent.futures import ProcessPoolExecutor
# Single-threaded BLAS in the process-parallel trajectory workers.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "./library"))
sys.pycache_prefix = os.path.abspath("./output/pycache")
from surface_hopping import mash, fssh
from qutils import ohmic_modes, parse_coeffs
from fsutils import load_config, save_time
from pltools import parse_xlims


# Run selection — comment out entries to skip.
METHODS = ["fssh"] # "mash" is also available
CASES   = ["qhm", "qah", "chm", "cah"]

# Per-method trajectory budgets (used when ``-n_traj`` is not given).
N_TRAJ = {"mash": 40000, "fssh": 20000}

# Mirror of the MKCT ``cases`` matrix: case -> (quantum sampling, anharmonic).
_CASE_MATRIX = {
    "qhm": (True,  False),
    "qah": (True,  True),
    "chm": (False, False),
    "cah": (False, True),
}


def _case_name(quantum: bool, anharm: bool) -> str:
    for name, (q, ah) in _CASE_MATRIX.items():
        if q == quantum and ah == anharm:
            return name
    raise ValueError(f"No case for quantum={quantum}, anharm={anharm}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description = "Surface-hopping reference dynamics for the spin-boson model.")
    parser.add_argument("prefix", nargs = "?", default = None,
                        help = "Config prefix (default: run RUNLIST)")
    parser.add_argument("-method", choices = ["mash", "fssh", "all"],
                        default = "all", help = "Trajectory method")
    parser.add_argument("-n_traj", type = int, default = None,
                        help = "Number of trajectories (default: per-method budget)")
    parser.add_argument("-N", type = int, default = 200,
                        help = "Number of bath modes")
    parser.add_argument("-wmax_factor", type = float, default = 10.0,
                        help = "Quadrature cutoff, in units of wC")
    parser.add_argument("-dt", type = float, default = 0.005,
                        help = "Output time step")
    parser.add_argument("-seed", type = int, default = 0,
                        help = "RNG seed")
    parser.add_argument("-nproc", type = int, default = 4,
                        help = "Number of cases to run in parallel (default: %(default)s)")
    parser.add_argument("-njobs", type = int, default = -1,
                        help = "Inner workers per case (default: -1, share the cores)")
    return parser.parse_args()


def _compute(prefix: str, method: str = "mash", quantum: bool = True,
             anharm: bool = False, n_traj: int = 40000, N: int = 200,
             wmax_factor: float = 10.0, dt: float = 0.005, seed: int = 0,
             n_jobs: int = 1):
    """Compute one ``(prefix, method, case)``; return ``(t, tag, result)``."""
    # 3. Parameters (physical values; ``scale`` is a numerical device only)
    params    = load_config(prefix)
    epsilon   = params["epsilon"]
    Delta     = params["Delta"]
    lambd     = params["lambd"]
    wC        = params["wC"]
    beta      = params["beta"]
    quadratic = params["quadratic"]
    model     = None
    if quadratic:
        # QSBM: H_sys = 2 epsilon |1><1| + Delta sigma_x, V = |1><1|,
        # U(q) = a0 + a1 Q + a2 Q^2, so Vbar = epsilon + U/2 and A = -Vbar.
        theta_B = params["theta_B"]
        wB      = params["wB"]
        alp0    = lambd * theta_B ** 2
        alp1    = -(2.0 * lambd * wB) ** 0.5 * theta_B ** 2
        alp2    = 0.5 * wB * (theta_B ** 2 - 1.0)
        a2, a1, a0 = parse_coeffs([alp2, alp1, alp0],
                                  linear = params["linear"], quadratic = True)
        pv0, pv1, pv2 = epsilon + a0 / 2.0, a1 / 2.0, a2 / 2.0
        model = (pv0, pv1, pv2, -pv0, -pv1, -pv2)
    a = float(params["anharmon"]) * np.sqrt(7.0 / 6.0) if anharm else 0.0
    xlims = parse_xlims(params["xlims"], reorder = True)
    tf    = xlims[0][1]
    omega, c = ohmic_modes(lambd, wC, N = N, wmax_factor = wmax_factor)
    # The quartic stiffening raises the fastest effective frequency, so an
    # anharmonic run needs a shorter step than a harmonic one.
    if a > 0.0:
        dt = min(dt, 0.15 / float(np.max(omega)))
    t     = np.linspace(0.0, tf, int(round(tf / dt)) + 1)

    tag = f"{method}_{_case_name(quantum, anharm)}"
    print(f"[{prefix}] {tag}: N={N} n_traj={n_traj} tf={tf:g} dt={dt:g} "
          f"beta={beta} wC={wC} lambd={lambd} a={a:.4f} n_jobs={n_jobs}",
          flush = True)

    # 4. Dynamics
    if method == "mash":
        res = mash.run_observables(epsilon, Delta, omega, c, beta, t, quantum = quantum,
                                   anharmon = a, n_traj = n_traj, seed = seed,
                                   model = model)
    else:
        res = fssh.run_observables(epsilon, Delta, omega, c, beta, t,
                                   quantum = quantum, anharmon = a,
                                   n_traj = n_traj, seed = seed,
                                   model = model, n_jobs = n_jobs)
    return t, tag, res


def _save(prefix: str, tag: str, t, res):
    """Write the five observables (only the parent process ever writes)."""
    outdir = "./output/data/" + prefix
    save_time(t, res["pop"],
              outdir, "pop_" + tag + ".npz", identifier = "P", compress = True)
    save_time(t, res["cohe_p"],
              outdir, "cohe_" + tag + ".npz", identifier = "S+", compress = True)
    save_time(t, res["cohe_m"],
              outdir, "cohem_" + tag + ".npz", identifier = "S-", compress = True)
    if "Cmu" in res:
        save_time(t, res["Cmu"],
                  outdir, "corr_" + tag + ".npz",
                  identifier = "⟨μ(t)μ(0)⟩", compress = True)
        save_time(t, res["Cx"],
                  outdir, "xorr_" + tag + ".npz",
                  identifier = "⟨Sx(t)Sy(0)⟩", compress = True)


def main(prefix: str, method: str = "mash", quantum: bool = True,
         anharm: bool = False, n_traj: int = 40000, N: int = 200,
         wmax_factor: float = 10.0, dt: float = 0.005, seed: int = 0,
         n_jobs: int = 1):
    t, tag, res = _compute(prefix, method, quantum, anharm, n_traj, N,
                           wmax_factor, dt, seed, n_jobs)
    _save(prefix, tag, t, res)


def _case_job(spec):
    """Worker entry point: one ``(prefix, method, case)`` specification."""
    t, tag, res = _compute(*spec)
    return t, tag, res


RUNLIST = [
    "A-base_line",
    "B-slow_system",
    "C-no_bias",
    "D-strong_tunneling",
    "E-fast_bath",
    "F-slow_bath",
    "G-lowT",
    "H-highT",
    "I-weak_coupling",
    "J-strong_coupling",
    "K-linear_quad",
]
def run(prefix: str, methods = METHODS, cases = CASES, n_traj = None,
        N: int = 200, wmax_factor: float = 10.0, dt: float = 0.005,
        seed: int = 0, nproc: int = 4, njobs: int = -1):
    """Run one or more methods across the selected cases for a single config.

    ``n_traj`` defaults to the per-method ``N_TRAJ`` budget.  The cases run as
    ``nproc`` parallel processes (at most one per case) and, when there are
    spare cores, each case splits its trajectories over ``njobs`` inner
    workers (``-1``: share the cores over the cases).  The parent process
    writes every archive.
    """
    nproc   = max(1, int(nproc))
    workers = max(1, min(nproc, len(cases)))
    if njobs is not None and int(njobs) > 0:
        n_inner = int(njobs)
    elif workers >= len(cases) and len(cases) > 1:
        n_inner = 1          # nested pools only add overhead here
    else:
        n_inner = max(1, (os.cpu_count() or 1) // workers)
    for method in methods:
        nt = N_TRAJ[method] if n_traj is None else n_traj
        inner = n_inner if method != "mash" else 1   # MASH has no inner pool
        specs = [(prefix, method, *_CASE_MATRIX[case], nt, N, wmax_factor,
                  dt, seed, inner) for case in cases]
        if workers == 1 or len(specs) == 1:
            for spec in specs:
                t, tag, res = _compute(*spec)
                _save(prefix, tag, t, res)
        else:
            with ProcessPoolExecutor(max_workers = workers,
                    mp_context = multiprocessing.get_context("fork")) as pool:
                for t, tag, res in pool.map(_case_job, specs):
                    _save(prefix, tag, t, res)


if __name__ == "__main__":
    args = _parse_args()
    methods = METHODS if args.method == "all" else [args.method]
    for prefix in ([args.prefix] if args.prefix else RUNLIST):
        run(prefix, methods = methods, cases = CASES, n_traj = args.n_traj,
            N = args.N, wmax_factor = args.wmax_factor, dt = args.dt,
            seed = args.seed, nproc = args.nproc, njobs = args.njobs)

# %%

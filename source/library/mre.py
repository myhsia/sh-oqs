"""Mean relative error of MKCT against HEOM, and the Padé-order scan core.

Shared tool (parallel to ``pltools.py``): the definition of the error, the
per-order scan, and the markdown/JSON reports.  The figure helpers
(``matrix_panel``, ``log_range``) live in ``pltools.GridPlt``.

For each config, scan all Pade orders (m, n) with m, n >= 2 and m + n <= N - 2
(N = number of moments), reconstruct the QHM observables with
``MKCT_solver.solve_pade``, and score them against the HEOM reference with the
relative L2 error

    E(m, n) = sum_obs || y_obs - y'_obs ||_2 / || y_obs ||_2

over six observables on the full ``[0, tf]`` grid: the real parts of the two
populations and of the auto-/cross-correlation functions, and the imaginary
parts of the latter two, i.e. ``{Re P0, Re P1, Re C_xx, Re C_xy, Im C_xx,
Im C_xy}``.  Every pair is ranked, and the script writes

    output/pade_scan_results.json   all pairs (infeasible pairs as ``null``),
                                    each row carrying its metric window ``tf``
    output/pade_scan.md             the worst-to-best ranking per config, above
                                    ``MAX_ERR_PCT`` (the JSON keeps all pairs)

reporting the mean relative error ``(E / n_obs) * 100 %``.

The driver (``source/pade_scan.py``) owns the run list, the observables and the
output paths::

    python pade_scan.py            # scan (or reuse cache)
    python pade_scan.py --rescan   # force a fresh scan
"""

import contextlib, io, json, os, time, warnings
import numpy as np

from qutils import get_expval_ABrho0
from fsutils import load_config, load_npz, load_moments, ensure_dir, moments_filename
from pltools import parse_xlims


N_OBS = 6            # default observable count of the metric
MAX_ERR_PCT = 1e3    # default cutoff for the markdown tables (%)


def _rnorm(y):
    return float(np.sqrt(np.sum(np.abs(y) ** 2)))


def _rel(a, b):
    """Relative ``L2`` error ``||a - b||_2 / ||b||_2`` (0 if ``b`` vanishes)."""
    den = _rnorm(b)
    return _rnorm(a - b) / den if den > 0 else 0.0


def _interp(yh, th, t):
    """Resample the HEOM reference ``yh(th)`` onto the MKCT times ``t``."""
    return np.interp(t, th, yh.real) + 1j * np.interp(t, th, yh.imag)


def scan_config(prefix, observables, dt=1e-2, tf_factor=20.0, solve=None,
                data_dir="./output/data"):
    """Return ``[{'m','n','Nmax','E','tf'}]`` over all admissible Pade orders.

    ``observables`` supplies the ``A``/``Brho0`` operator tables and the
    ``basis_list`` used to evaluate the QHM series; ``solve`` holds the
    ``solve_pade`` keyword arguments other than ``tf``/``tfout``/``dt``.
    """
    # Imported lazily: the error functions also serve repositories that do
    # not ship the MKCT solver (e.g. the surface-hopping reference).
    from mkct_pade.mkct_solver import MKCT_solver
    solve = dict(solve or {})
    A, Brho0 = observables["A"], observables["Brho0"]
    basis_list = observables["basis_list"]
    params = load_config(prefix)
    rescale = params["scale"]
    tf = parse_xlims(params["xlims"], reorder=True)[0][1]
    d = os.path.join(data_dir, prefix) + os.sep
    mom = load_moments(d + moments_filename("matrix"))["qhm"]
    N = mom.shape[0]
    Mmax = N - 2
    th, ph = load_npz(d + "pop_heom.npz")
    _, Cmuh = load_npz(d + "corr_heom.npz")
    _, Cxh = load_npz(d + "xorr_heom.npz")
    if th[-1] > tf * (1.0 + 1e-6):
        warnings.warn(
            f"[{prefix}] HEOM reference reaches t={th[-1]:g} but the metric "
            f"window is [0, {tf:g}] (config xlims); the extra range is ignored.",
            stacklevel=2)
    # Keep only the calculation window: numerator and denominator both use it.
    win = th <= tf * (1.0 + 1e-9)
    thw, phw, Cmw, Cxw = th[win], ph[win], Cmuh[win], Cxh[win]

    solver = MKCT_solver.init(mom, rescale=rescale)
    res = []
    t0 = time.time()
    for m in range(2, Mmax - 1):
        for n in range(2, Mmax - m + 1):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    t, Ct = solver.solve_pade(
                        tf=tf_factor * tf, tfout=tf, dt=dt, kernel_order=1,
                        pade_order=(m, n), **solve)
                pop = np.zeros((len(t), 2), complex)
                for i in range(2):
                    pop[:, i] = get_expval_ABrho0(
                        Ct, A["pop"][i], Brho0["pop"], basis_list)
                Cmu = get_expval_ABrho0(Ct, A["corr"], Brho0["corr"], basis_list)
                Cmu = Cmu / Cmu[0]
                Cx = get_expval_ABrho0(Ct, A["xorr"], Brho0["xorr"], basis_list)
                phi = np.stack([np.interp(t, thw, phw[:, i]) for i in range(2)], axis=1)
                Cmu_i = _interp(Cmw, thw, t)
                Cx_i = _interp(Cxw, thw, t)
                E = (_rel(pop[:, 0].real, phi[:, 0])
                     + _rel(pop[:, 1].real, phi[:, 1])
                     + _rel(Cmu.real, Cmu_i.real)
                     + _rel(Cx.real, Cx_i.real)
                     + _rel(Cmu.imag, Cmu_i.imag)
                     + _rel(Cx.imag, Cx_i.imag))
                E = float(E)
                if not np.isfinite(E):
                    E = np.inf
            except Exception:
                E = np.inf
            res.append({"m": m, "n": n, "Nmax": m + n + 2, "E": E,
                        "tf": float(tf)})
    feas = sorted([r for r in res if np.isfinite(r["E"])], key=lambda r: r["E"])
    print(f"[{prefix}] N={N} pairs={len(res)} feasible={len(feas)} "
          f"time={time.time()-t0:.0f}s argmin=({feas[0]['m']},{feas[0]['n']}) "
          f"N{feas[0]['Nmax']} E={feas[0]['E']:.4e}", flush=True)
    return res


def write_markdown(raw, path, n_obs=N_OBS, max_pct=MAX_ERR_PCT, runlist=None):
    """Worst-to-best ranking of the feasible pairs, above the error cutoff."""
    runlist = list(runlist if runlist is not None else raw)
    lines = [
        "# Pade-order scan (QHM vs HEOM)", "",
        "Error metric: `E = sum_obs ||y_obs - y'_obs||_2 / ||y_obs||_2` over the "
        "full `[0, tf]` (the config's window from `xlims`, listed per config), "
        "QHM vs HEOM; the tables report the **mean relative "
        f"error** `(E/{n_obs})*100%` over the six observables (the real parts of "
        "P0, P1, C_xx, C_xy and the imaginary parts of C_xx, C_xy).",
        "`Nmax = m + n + 2`; pairs satisfy `m, n >= 2` and singular/infeasible "
        f"pairs are excluded. Pairs with a mean error above `{max_pct:g}%` are "
        "omitted from the tables (the JSON keeps them); each config lists the "
        "remaining pairs ordered from **largest (#1)** to **smallest** error.", "",
        "| config | best (m, n) | Nmax | mean error (%) | tf | feasible pairs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for prefix in runlist:
        feas = sorted([r for r in raw[prefix] if np.isfinite(r["E"])],
                      key=lambda r: r["E"])
        b = feas[0]
        lines.append(f"| {prefix} | ({b['m']}, {b['n']}) | {b['Nmax']} | "
                     f"{b['E'] / n_obs * 100:.4g} | {b['tf']:g} | {len(feas)} |")
    for prefix in runlist:
        feas = sorted([r for r in raw[prefix] if np.isfinite(r["E"])],
                      key=lambda r: r["E"])
        shown = [r for r in feas if r["E"] / n_obs * 100 <= max_pct]
        lines += ["", f"## {prefix} ({len(shown)} of {len(feas)} shown)", "",
                  "| # | m | n | Nmax | mean error (%) |",
                  "|---:|---:|---:|---:|---:|"]
        for i, r in enumerate(reversed(shown), start=1):
            lines.append(f"| {i} | {r['m']} | {r['n']} | {r['Nmax']} | "
                         f"{r['E'] / n_obs * 100:.4g} |")
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("WROTE", path, flush=True)


def _to_jsonable(raw):
    """Replace non-finite ``E`` (infeasible) with JSON ``null``."""
    out = {}
    for p, rows in raw.items():
        out[p] = []
        for r in rows:
            r = dict(r)
            if not np.isfinite(r["E"]):
                r["E"] = None
            out[p].append(r)
    return out


def _from_jsonable(raw):
    """Restore JSON ``null`` back to ``float('inf')`` for the report logic."""
    for rows in raw.values():
        for r in rows:
            if r["E"] is None:
                r["E"] = float("inf")
    return raw


def run(runlist, observables, json_path, md_path, *, n_obs=N_OBS,
        max_pct=MAX_ERR_PCT, rescan=False, **scan_kwargs):
    """Scan ``runlist`` (or load the JSON cache), then write both artifacts."""
    ensure_dir(os.path.dirname(json_path))
    if rescan or not os.path.exists(json_path):
        raw = {p: scan_config(p, observables, **scan_kwargs) for p in runlist}
        print("scanned", len(runlist), "configs", flush=True)
    else:
        with open(json_path) as f:
            raw = _from_jsonable(json.load(f))
        print("loaded", json_path, flush=True)
    with open(json_path, "w") as f:
        json.dump(_to_jsonable(raw), f, indent=2)
    print("WROTE", json_path, flush=True)
    write_markdown(raw, md_path, n_obs=n_obs, max_pct=max_pct, runlist=runlist)
    return raw

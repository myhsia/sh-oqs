# %%
"""Accuracy of the surface-hopping references against HEOM.

The metric is the one of the Padé-order scan (``library/mre.py``): over the
calculation window ``[0, tf]`` of each config, the relative ``L2`` error of
six curves

    {Re P0, Re P1, Re C_xx, Re C_xy, Im C_xx, Im C_xy}

against the HEOM reference, averaged over the six and reported in percent,
``(E / 6) * 100``.  Only the window is compared: numerator and denominator
both use the same slice, exactly as ``mre.scan_config`` does.

This is the runner; the algorithm is imported from the shared ``mre`` module.
It writes ``output/sh_error.json`` (one value per config/method, each row
carrying the window ``tf``) and prints the table.

Usage::

    python sh_error.py                       # RUNLIST x all methods
    python sh_error.py --methods mash fssh
    python sh_error.py --configs A_base_line --labels "" best maxA
"""
# %%
# 1. Requirements
import argparse, json, os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "./library"))
sys.pycache_prefix = os.path.abspath("./output/pycache")
import mre as core
from fsutils import load_config, load_npz, load_group, ensure_dir
from pltools import parse_xlims


# 2. Run selection
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
CASES   = ["qhm", "qah", "chm", "cah"]
METHODS = ["mkct", "mash", "fssh", "ehrenfest", "lsc"]
# Suffixes of labelled runs (the parameter sweep of the reference scenario).
LABELS  = ["", "base", "wf8", "wf14", "N400", "N800", "dt0025", "traj160k",
           "best", "maxA", "maxB"]

# The HEOM and MKCT references live in the canonical mkct4oqs checkout.
REF  = os.environ.get("SH_REF_DATA", "../../mkct4oqs/source/output/data")
JSON = "./output/sh_error.json"
MD   = "./output/sh_error.md"


def _series(prefix, method, case, quantity, label=""):
    """``(t, y)`` of one observable for one (method, case, label)."""
    d = "./output/data/" + prefix + "/"
    if method == "mkct":
        data, _ = load_group(os.path.join(REF, prefix) + os.sep +
                             quantity + "_mkct.npz")
        return data[case]
    tag = f"{method}_{case}" + (f"_{label}" if label else "")
    merged = d + ("cohe" if quantity == "cohem" else quantity)
    merged += f"_{method}" + (f"_{label}" if label else "") + ".npz"
    if not os.path.exists(merged):
        if not label:
            raise FileNotFoundError(f"{merged}: run sh_merge.py first")
        # labelled sweep runs stay per-case (they are not merged)
        return load_npz(d + f"{quantity}_{tag}.npz")
    data, _ = load_group(merged)
    t, y = data[case]
    if quantity == "cohem":
        y = y[1]                     # branch ("+", "-") on the leading axis
    elif quantity == "cohe":
        y = y[0]
    return t, y


def _heom(prefix, quantity, tf):
    d = os.path.join(REF, prefix) + os.sep
    t, y = load_npz(d + f"{quantity}_heom.npz")
    win = t <= tf * (1.0 + 1e-9)
    return t[win], y[win]


def error(prefix, method, case, label=""):
    """Mean relative error (percent) over the six curves; None if absent."""
    d = "./output/data/" + prefix + "/"
    tag = f"{method}_{case}" + (f"_{label}" if label else "")
    stem = "pop_heom.npz" if method == "mkct" else f"pop_{tag}.npz"
    exists = (os.path.exists(os.path.join(REF, prefix) + os.sep + stem)
              if method == "mkct" else os.path.exists(d + stem))
    if not exists:
        return None
    tf = parse_xlims(load_config(prefix)["xlims"], reorder = True)[0][1]
    thp, php = _heom(prefix, "pop", tf)
    thc, chc = _heom(prefix, "corr", tf)
    thx, chx = _heom(prefix, "xorr", tf)

    t, pop = _series(prefix, method, case, "pop", label)
    win = t <= tf * (1.0 + 1e-9)
    t = t[win]
    pop = pop[win]
    _, c = _series(prefix, method, case, "corr", label)
    _, x = _series(prefix, method, case, "xorr", label)

    cmu = core._interp(chc, thc, t)
    cx = core._interp(chx, thx, t)
    # The six curves of the canonical metric, exactly as mre.scan_config.
    e = (core._rel(pop[:, 0].real, np.interp(t, thp, php[:, 0].real))
         + core._rel(pop[:, 1].real, np.interp(t, thp, php[:, 1].real))
         + core._rel(c.real, cmu.real)
         + core._rel(x.real, cx.real)
         + core._rel(c.imag, cmu.imag)
         + core._rel(x.imag, cx.imag))
    return 100.0 * e / core.N_OBS


def main():
    ap = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    ap.add_argument("--configs", nargs = "*", default = RUNLIST)
    ap.add_argument("--methods", nargs = "*", default = METHODS)
    ap.add_argument("--labels", nargs = "*", default = LABELS)
    args = ap.parse_args()

    ensure_dir("./output")
    raw = {}
    for prefix in args.configs:
        rows = raw.setdefault(prefix, {})
        for method in args.methods:
            for label in args.labels:
                key = method if not label else f"{method}[{label}]"
                per = {c: error(prefix, method, c, label) for c in CASES}
                got = [v for v in per.values() if v is not None]
                if not got:
                    continue
                rows[key] = {"E": float(np.mean(got)),
                             "cases": {c: v for c, v in per.items()
                                       if v is not None},
                             "tf": float(
                    parse_xlims(load_config(prefix)["xlims"],
                                reorder = True)[0][1])}
    if os.path.exists(JSON):
        with open(JSON) as f:
            old = json.load(f)
        for p, rows in raw.items():          # merge, do not lose other runs
            old.setdefault(p, {}).update(rows)
        raw = old
    with open(JSON, "w") as f:
        json.dump(raw, f, indent = 2, sort_keys = True)
    lines = ["# Surface-hopping accuracy against HEOM", "",
             "Metric `(E/6)*100 %` over `[0, tf]` for "
             "`{Re P0, Re P1, Re C_xx, Re C_xy, Im C_xx, Im C_xy}`.", "",
             "| config | " + " | ".join(args.methods) + " |",
             "|---|" + "---:|" * len(args.methods)]
    for prefix in args.configs:
        cells = []
        for m in args.methods:
            v = raw.get(prefix, {}).get(m)
            cells.append(f"{v['E']:.1f}" if v else "--")
        lines.append(f"| {prefix} | " + " | ".join(cells) + " |")
    with open(MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("WROTE", JSON, "and", MD, flush = True)


if __name__ == "__main__":
    main()

# %%

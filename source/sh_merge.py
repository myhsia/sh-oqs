# %%
"""Group the per-case surface-hopping files into one archive per quantity.

Each CI job computes one ``(config, case)``, so the raw output is
``<quantity>_<method>_<case>.npz`` (one file per case, unique per job: no
concurrent writes).  This script groups the four cases into
``<quantity>_<method>.npz`` with ``fsutils.save_group``, the same layout as
``<quantity>_mkct.npz`` (``t_<case>`` plus ``<case>``, with ``component`` for
the populations and ``branch`` for the coherence).

When several independent seeds were run per case (``..._<case>_s0.npz``,
``_s1``, ...), their observables are averaged first: every file is already an
ensemble average, so the mean of equal-size chunks is the average over the
union.

Usage::

    python sh_merge.py                      # every config/method found
    python sh_merge.py --configs A-base_line
    python sh_merge.py --methods mash fssh
"""
# %%
import argparse, glob, os, re, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "./library"))
sys.pycache_prefix = os.path.abspath("./output/pycache")
from fsutils import load_npz, save_group

CASES = ("qhm", "qah", "chm", "cah")
QUANTITIES = ("pop", "cohe", "cohem", "corr", "xorr")
META = {"pop": dict(component = ("P0", "P1")),
        "cohe": dict(branch = ("+", "-")),
        "cohem": dict(branch = ("+", "-"))}


def _seed_files(directory, quantity, method, case, label=""):
    """All files of one case, one per seed (or a single one)."""
    stem = f"{quantity}_{method}" + (f"_{label}" if label else "") + f"_{case}"
    # Only the plain file and explicit seeds ``_s0``/``_s1``/...: the labelled
    # sweep files (``_base``, ``_maxA``, ...) belong to a different setting and
    # may have a different time grid.
    files = []
    plain = os.path.join(directory, stem + ".npz")
    if os.path.exists(plain):
        files.append(plain)
    files += sorted(glob.glob(os.path.join(directory, stem + "_s[0-9]*.npz")))
    return files


def _load_avg(files):
    """Average the ``(t, y)`` of several seed files (equal weights)."""
    ts, ys = zip(*(load_npz(f) for f in files))
    t = ts[0]
    y = sum(ys) / len(ys)
    return t, y


def merge_config(prefix, methods, labels, root, clean=False):
    directory = os.path.join(root, prefix)
    made = []
    for method in methods:
        for label in labels:
            series = {q: {} for q in QUANTITIES}
            for case in CASES:
                for q in QUANTITIES:
                    files = _seed_files(directory, q, method, case, label)
                    if not files:
                        continue
                    if q == "cohem":
                        continue          # coherence is handled by "cohe"
                    if q == "cohe":
                        # The two branches live in separate files: stack them
                        # as (2, nt) with branch = ("+", "-"), like MKCT.
                        fs_p = _seed_files(directory, "cohe", method, case, label)
                        fs_m = _seed_files(directory, "cohem", method, case, label)
                        if not fs_p:
                            continue
                        t, cp = _load_avg(fs_p)
                        if fs_m:
                            _, cm = _load_avg(fs_m)
                            series[q][case] = (t, np.stack([cp, cm]))
                        else:
                            series[q][case] = (t, np.stack([cp, cp * 0]))
                        continue
                    t, y = _load_avg(files)
                    series[q][case] = (t, y)
            if clean and label == "":
                for case in CASES:
                    for q in QUANTITIES:
                        for f in _seed_files(directory, q, method, case, label):
                            os.remove(f)
            for q in ("pop", "cohe", "corr", "xorr"):
                if len(series[q]) != len(CASES):
                    continue
                tag = method + (f"_{label}" if label else "")
                name = f"{q}_{tag}.npz"
                save_group(series[q], directory, name, identifier = q,
                           **META.get(q, {}))
                made.append(name)
    return made


def main():
    ap = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    ap.add_argument("--configs", nargs = "*", default = None)
    ap.add_argument("--methods", nargs = "*",
                    default = ["fssh", "mash", "ehrenfest", "lsc"])
    ap.add_argument("--labels", nargs = "*", default = [""])
    ap.add_argument("--root", default = "./output/data")
    ap.add_argument("--clean", action = "store_true",
                    help = "delete the per-case files once merged")
    args = ap.parse_args()
    configs = args.configs or sorted(
        d for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d)))
    total = 0
    for prefix in configs:
        made = merge_config(prefix, args.methods, args.labels, args.root,
                            clean = args.clean)
        if made:
            print(f"[{prefix}] merged {len(made)} archives", flush = True)
            total += len(made)
    print(f"merged {total} archives under {args.root}")


if __name__ == "__main__":
    main()

# %%

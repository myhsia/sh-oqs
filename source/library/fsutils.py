"""Filesystem and IO helpers: config loading, output writers and loaders."""
import copy, os, yaml
import numpy as np
from pathlib import Path


def _apply_relative(value, base):
    if not isinstance(value, str) or base is None:
        return value
    s = value.strip()
    if len(s) < 2 or s[0] not in "*/+-":
        return value
    try:
        factor = float(s[1:])
    except ValueError:
        return value
    if s[0] == "*":
        return base * factor
    if s[0] == "/":
        return base / factor
    if s[0] == "+":
        return base + factor
    return base - factor


def _merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = _apply_relative(value, merged.get(key))
    return merged


def load_config(prefix: str, config_dir: str = "./config"):
    with open(os.path.join(config_dir, prefix + ".yaml"), "r") as f:
        override = yaml.safe_load(f)
    base_name = override.pop("base", None)
    if base_name is None:
        return override
    base_name = os.path.splitext(str(base_name))[0]
    with open(os.path.join(config_dir, base_name + ".yaml"), "r") as f:
        params = yaml.safe_load(f)
    return _merge(params, override)


def ensure_dir(directory) -> None:
    """Create ``directory`` (and any missing parents) if it does not exist."""
    Path(directory).mkdir(parents = True, exist_ok = True)


def save_time(t, Y, directory, filename, identifier = "",
              compress: bool = False) -> None:
    filepath = Path(directory) / filename
    ensure_dir(filepath.parent)
    if compress:
        np.savez(filepath, t = np.asarray(t), y = np.asarray(Y),
                 identifier = np.array(identifier))
        return
    if np.iscomplexobj(Y):
        datout = np.column_stack((t, Y.real, Y.imag))
        header_list = (["t", "Re", "Im"] if identifier == "" else
                       ["t", f"Re[{identifier}]", f"Im[{identifier}]"])
    else:
        Y = np.asarray(Y)
        Y = Y[:, None] if Y.ndim == 1 else Y
        datout = np.column_stack((t, Y))
        prefix = identifier if identifier else "Y"
        header_list = ["t"] + [f"{prefix}{i}" for i in range(Y.shape[1])]
    fmt = "%15.6e" * datout.shape[1]
    header = f"{header_list[0]:>13}" + "".join(f"{h:>15}" for h in header_list[1:])
    np.savetxt(filepath, datout, fmt = fmt, header = header)


def load_npz(path):
    data = np.load(path)
    return data["t"], data["y"]


# Metadata keys of a ``save_group`` archive (not per-case series).
GROUP_META = ("identifier", "component", "branch")


def save_group(series, directory, filename, identifier = "",
               component = None, branch = None, compress: bool = True) -> None:
    """Save per-case series with their own time axes in a single ``.npz``.

    ``series`` maps a case name (or any key, e.g. the Padé order ``"2/3"``) to
    its ``(t, y)`` pair; each pair is stored as ``t_<key>`` and ``<key>``, so
    the archive reads as ``np.load(path)["qhm"]`` with ``["t_qhm"]``.
    ``identifier`` records the observable, ``component`` labels the inner axis
    of a two-component series (e.g. ``("P0", "P1")`` for populations) and
    ``branch`` the first axis of stacked branches (e.g. ``("+", "-")`` for the
    two coherences).
    """
    filepath = Path(directory) / filename
    ensure_dir(filepath.parent)
    payload = {}
    for key, (t, y) in series.items():
        payload["t_" + str(key)] = np.asarray(t)
        payload[str(key)]        = np.asarray(y)
    if identifier:
        payload["identifier"] = np.array(identifier)
    if component is not None:
        payload["component"] = np.asarray(component)
    if branch is not None:
        payload["branch"] = np.asarray(branch)
    (np.savez_compressed if compress else np.savez)(filepath, **payload)


def load_group(path):
    """Load a ``save_group`` archive as ``(series, meta)``.

    ``series`` maps each case (or order) key to its ``(t, y)`` pair; ``meta``
    holds the ``identifier``/``component``/``branch`` labels that are present.
    """
    with np.load(path) as f:
        series = {k: (f["t_" + k], f[k]) for k in f.files
                  if k not in GROUP_META and "t_" + k in f.files}
        meta   = {k: f[k] for k in GROUP_META if k in f.files}
    return series, meta


def moments_filename(projection: str = "matrix") -> str:
    """Canonical moments archive ``moments.{mat,scl}.npz`` (one array per case),
    where ``projection`` is ``"matrix"`` (``mat``) or ``"scalar"`` (``scl``).
    """
    return f"moments.{'scl' if projection == 'scalar' else 'mat'}.npz"


def save_moments(path, series) -> None:
    """Save one moments array per case, each under its case name.

    ``series`` maps a case (``qhm``/``qah``/``chm``/``cah``) to its array; a
    scalar ``(n_moments,)`` or matrix ``(n_moments, d, d)`` complex array keeps
    its native shape (no flattening).
    """
    filepath = Path(path)
    ensure_dir(filepath.parent)
    np.savez(filepath, **{str(k): np.asarray(v) for k, v in series.items()})


def load_moments(path) -> dict:
    """Load a moments archive as ``{case: array}``.

    Each array keeps its native shape: ``(n_moments, d, d)`` for the matrix
    projection or ``(n_moments,)`` for the scalar projection.
    """
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def tab_str(s, N = 80, char = '=', border = '!'):
    """Pad ``s`` to a border line of constant width ``N`` for console logging."""
    n = len(s)
    n_left = (N - n - 4) // 2
    n_right = N - n - 4 - n_left
    return border + char * n_left + ' ' + s + ' ' + char * n_right + border

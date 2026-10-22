"""Quantum utilities: operators/states, bath statistics, and the spin-boson model.

This is the shared library for the well-known quantum primitives used across the
sub-libraries (``surface_hopping``, ``dmrg``, ``mkct_pade``, ``moment``):
Pauli/basis operators, bosonic ladder operators, the spin-boson model, the Ohmic
spectral density and its Gauss-Legendre discretization, and thermal statistics.
It depends only on numpy/numba (mpmath is imported lazily by :func:`eta0`), so it
never imports a sub-library.
"""
import numpy as np
from numba import njit
from typing import Union, Tuple, Literal


# ===========================================================================
# Operators and states
# ===========================================================================
def _comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.dot(A, B) - np.dot(B, A)

def comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if B.ndim == 2:
        return _comm(A, B)
    elif B.ndim == 3:
        return np.array([_comm(A, b) for b in B])


def detect_projection(A: np.ndarray,
                      sigma_0: np.ndarray = None) -> Literal["scalar", "matrix"]:
    """Infer the projection mode from the shape of ``A`` (and ``sigma_0``).

    - ``A`` and ``sigma_0`` both ``(d, d)``: ``"scalar"`` (single operator and
      single initial density matrix);
    - ``A`` ``(N, d, d)``: ``"matrix"`` (an operator basis, e.g. the matrix
      units; if ``sigma_0`` is given it must be the matching density-matrix
      list ``(N, N, d, d)``).

    A scalar ``A`` requires ``sigma_0``; a matrix ``A`` does not (the block
    formulation absorbs ``sigma_0`` into the basis).  A mono/list mismatch or
    any other shape raises ``ValueError``.
    """
    A = np.asarray(A)
    if A.ndim == 2:
        if sigma_0 is None:
            raise ValueError("scalar projection requires sigma_0")
        sigma_0 = np.asarray(sigma_0)
        if sigma_0.ndim != 2 or sigma_0.shape != A.shape:
            raise ValueError(
                "A and sigma_0 must be matching (d, d) operators; got "
                f"{A.shape} and {sigma_0.shape}.")
        return "scalar"

    if A.ndim == 3:
        N, d, d2 = A.shape
        if d != d2:
            raise ValueError(
                "A must be a list of square operators (N, d, d); got "
                f"{A.shape}.")
        if sigma_0 is not None:
            sigma_0 = np.asarray(sigma_0)
            if sigma_0.ndim != 4 or sigma_0.shape != (N, N, d, d):
                raise ValueError(
                    "A (N, d, d) and sigma_0 must be the matching pair "
                    f"(N, N, d, d); got A {A.shape} and sigma_0 {sigma_0.shape}.")
        return "matrix"

    raise ValueError(
        "A must be a single operator (d, d) or an operator list (N, d, d); "
        f"got ndim {A.ndim}.")


_PAULI_DICT = {
    "x": np.array([[0,   1], [1,   0]], dtype = np.complex128),
    "y": np.array([[0, -1j], [1j,  0]], dtype = np.complex128),
    "z": np.array([[1,   0], [0,  -1]], dtype = np.complex128),
    "+": np.array([[0,   1], [0,   0]], dtype = np.complex128),
    "-": np.array([[0,   0], [1,   0]], dtype = np.complex128),
}
def Pauli(arg: str) -> np.ndarray:
    """Returns the Pauli matrix for 'x', 'y', 'z', '+', or '-'."""
    key = str(arg).lower()
    if key in _PAULI_DICT:
        return _PAULI_DICT[key].copy()
    raise ValueError(
        f"Invalid Pauli label '{arg}': Must be 'x', 'y', 'z', '+', or '-'.")


def basis(d: int, n: Union[int, Tuple[int, int]],
          kind: Union[str, None] = None) -> np.ndarray:
    """Generates a quantum basis state vector or operator in d dimensions.
    Parameters:
        d (int): Hilbert space dimension
        n (int | (int, int)): State index.
            For 'ket'/'bra', n must be an int with 0 <= n < d.
            For 'ketbra', n may be an int or a 2-tuple producing |i><j|.
        kind (str | None): Output format: 'ket', 'bra', or 'ketbra'.
            Defaults to 'ketbra' when n is a 2-tuple, otherwise 'ket'.
    """
    if kind is None:
        kind = "ketbra" if isinstance(n, tuple) else "ket"
    kind = kind.lower()
    if kind in ("ket", "bra"):
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"For kind = '{kind}', n must be an int, got {n!r}")
        if not (0 <= n < d):
            raise ValueError(f"Index n = {n} must satisfy 0 <= n < {d}")
        # Base column vector |n> with complex dtype for general QM
        k = np.zeros((d, 1), dtype = complex)
        k[n, 0] = 1.0
        return k if kind == "ket" else k.conj().T  # Shape: (d, 1) or (1, d)
    elif kind == "ketbra":
        if isinstance(n, tuple):
            if len(n) != 2:
                raise ValueError(
                    "For kind = 'ketbra', n must be an int or 2-tuple")
            i, j = n
        else:
            if not isinstance(n, (int, np.integer)):
                raise TypeError(
                    "For kind = 'ketbra', n must be an int or 2-tuple")
            i = j = n
        if not (0 <= i < d) or not (0 <= j < d):
            raise ValueError(
                f"Indices ({i}, {j}) must each satisfy 0 <= k < {d}")
        ki = np.zeros((d, 1), dtype = complex)
        ki[i, 0] = 1.0
        kj = np.zeros((d, 1), dtype = complex)
        kj[j, 0] = 1.0
        return ki @ kj.conj().T  # Shape: (d, d)
    else:
        raise ValueError("kind must be 'ket', 'bra', or 'ketbra'")


def qubit(theta: float = 0.0, phi: float = 0.0,
          kind: str = "ket") -> np.ndarray:
    """Standard qubit state on the Bloch sphere.
    |psi(theta, phi)> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>,
    with Bloch vector (sin theta cos phi, sin theta sin phi, cos theta).
    Parameters:
        theta (float): Polar angle, 0 <= theta <= pi.
        phi   (float): Azimuthal angle, 0 <= phi < 2 pi.
        kind  (str):   'ket', 'bra', or 'ketbra' (returns |psi><psi|).
    """
    if not (0.0 <= theta <= np.pi):
        raise ValueError(f"theta must be in [0, pi], got {theta}")
    if not (0.0 <= phi < 2 * np.pi):
        raise ValueError(f"phi must be in [0, 2 pi), got {phi}")
    kind = kind.lower()
    psi = np.array([[np.cos(theta/2)],
                    [np.exp(1j * phi) * np.sin(theta/2)]], dtype = complex)
    if kind == "ket":
        return psi
    if kind == "bra":
        return psi.conj().T
    if kind == "ketbra":
        return psi @ psi.conj().T
    raise ValueError("kind must be 'ket', 'bra', or 'ketbra'")


def ladder_ops(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bosonic annihilation/creation operators ``(a, a^dagger)`` (N x N).

    ``a = sum_n sqrt(n) |n-1><n|``; ``a^dagger = a.T`` (real matrices).
    """
    a = np.diag(np.sqrt(np.arange(1, N)), 1)
    return a, a.T


# ===========================================================================
# Model Hamiltonians
# ===========================================================================
def spin_boson(
    model: str = "LSBM", 
    epsilon: float = 1.0, 
    Delta: float = 0.0,
    nohalf: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Hamiltonian & Coupling Generator for the Spin-Boson Model variants.
    Parameters:
      model   (str):     Model variant ('LSBM' / 'SBM' or 'QSBM' / 'QSB').
      epsilon (float):   Energy gap / bias parameter (epsilon).
      Delta   (float):   Tunneling amplitude (default: 0.0).
      nohalf  (boolean): Double H_sys and V under 'LSBM', or
                         double H_sys only under 'QSBM'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (H_sys, V) as (2, 2) complex arrays.
    """
    model = str(model).upper()
    if model in ("LSBM", "SBM", "SSBM"):
        # Pauli matrices
        V     = Pauli("z") * 0.5
        H_sys = Pauli("z") * 0.5 * epsilon + Pauli("x") * 0.5 * Delta
        if nohalf:
            V, H_sys = V * 2, H_sys * 2
    elif model in ("QSBM", "QSB"):
        # |1⟩⟨1| projection operator
        kb_01 = basis(d = 2, n = (0, 1), kind = "ketbra")
        kb_10 = basis(d = 2, n = (1, 0), kind = "ketbra")
        kb_11 = basis(d = 2, n = 1,      kind = "ketbra")
        V     = kb_11
        H_sys = kb_11 * epsilon + 0.5 * Delta * (kb_01 + kb_10)
        if nohalf:
            H_sys *= 2
    else:
        raise ValueError(f"Unknown model '{model}'. Available: 'LSBM', 'QSBM'.")
    return H_sys, V


def parse_coeffs(
    coeffs: list[float],
    linear: bool = True,
    quadratic: bool = False
) -> list[float]:
    """Normalise the system-bath coupling polynomial coefficients.

    ``coeffs`` are in numpy order, ``U(q) = a0 + a1 q + a2 q^2 + ...``; the
    returned array is the coupling polynomial used by the moment machinery
    (``H_SB = V U``).  Leading zeros are trimmed; for the linear (default) case
    the canonical ``[1, 0]`` (``H_SB = V q``) is returned; for quadratic
    coupling the last one or three coefficients are kept.
    """
    # Remove leading continuous zeros
    trimmed = np.trim_zeros(coeffs, 'f')
    # Edge case: if input was all zeros, H_SB = V q
    edge_coeffs = [1.0, 0.0]
    if not trimmed:
        trimmed = np.array(edge_coeffs)
    if quadratic and linear:
        # Return the last 3 elements: [alp2, alp1, alp0]
        return np.array(trimmed[-3:])
    elif quadratic:
        # Return alp2 only
        alp2 = trimmed[-3] if len(trimmed) >= 3 else 0.0
        return np.array([alp2, 0.0, 0.0])
    else:
        # Return edge_coeffs for the default (including only-linear) case
        return np.array(edge_coeffs)


# ===========================================================================
# Bath spectral density and modes
# ===========================================================================
def ohmic_J(w, lambd, wC):
    """Ohmic spectral density ``J(w) = pi*lambd/wC * w * exp(-w/wC)``."""
    w = np.asarray(w, dtype=float)
    return np.pi * lambd / wC * w * np.exp(-w / wC)


def discretize(J, N, wmax):
    """Gauss-Legendre discretization of a spectral density.

    Parameters
    ----------
    J : callable
        Spectral density ``J(w)``.
    N : int
        Number of bath modes.
    wmax : float
        Upper cutoff of the quadrature grid.

    Returns
    -------
    omega : (N,) ndarray
        Mode frequencies.
    c : (N,) ndarray
        Mode couplings, such that ``(pi/2) sum_j c_j^2 delta(w - w_j) ~ J(w)``
        i.e. ``sum_j c_j^2 f(w_j) ~ (2/pi) int J(w) f(w) dw``.
    """
    x, wq = np.polynomial.legendre.leggauss(N)
    omega = 0.5 * wmax * (x + 1.0)
    wq = 0.5 * wmax * wq
    c = np.sqrt(np.maximum((2.0 / np.pi) * J(omega) * wq, 0.0))
    return omega, c


def ohmic_modes(lambd, wC, N=100, wmax_factor=8.0):
    """Discretized Ohmic modes for ``J(w)=pi*lambd/wC * w * exp(-w/wC)``."""
    return discretize(lambda w: ohmic_J(w, lambd, wC), N, wmax_factor * wC)


def reorganization_energy(omega, c):
    """``E_r = sum_j c_j^2 / (2 w_j)``."""
    return float(np.sum(c**2 / (2.0 * omega)))


def quartic_coefficient(a_prime=0.2):
    """Physical quartic coefficient ``a = a' * sqrt(7/6)``."""
    return a_prime * np.sqrt(7.0 / 6.0)


# ===========================================================================
# Thermal statistics
# ===========================================================================
def thermal_variances(omega, beta, quantum=True):
    """``sigma_q^2 = sigma_p^2`` of each mode in the harmonic equilibrium.

    quantum   : sigma^2 = (1/2) coth(beta*w/2)
    classical : sigma^2 = 1 / (beta*w)
    """
    omega = np.asarray(omega, dtype=float)
    if quantum:
        return 0.5 / np.tanh(beta * omega / 2.0)
    return 1.0 / (beta * omega)


def thermal_weights(w, beta, nmax):
    """Harmonic Gibbs weights ``p_n propto exp(-beta w n)``, ``n = 0..nmax``.

    Returns the normalized occupation probabilities (length ``nmax + 1``).
    """
    n = np.arange(nmax + 1)
    p = np.exp(-beta * w * n)
    return p / p.sum()


def sample_bath(omega, beta, quantum=True, n_traj=1, rng=None):
    """Sample ``(q, p)`` of shape ``(n_traj, N)`` from the bath equilibrium."""
    if rng is None:
        rng = np.random.default_rng()
    sig = np.sqrt(thermal_variances(omega, beta, quantum=quantum))
    q = rng.normal(size=(n_traj, len(omega))) * sig
    p = rng.normal(size=(n_traj, len(omega))) * sig
    return q, p


def eta0(J, beta, quantum=True, wmax=np.inf):
    """Reference ``<q^2> = eta_0``.

    quantum   : (1/pi) int J(w) / tanh(beta w / 2) dw
    classical : (2 / (pi beta)) int J(w) / w dw
    """
    import mpmath as mp
    wmax = mp.inf if np.isinf(wmax) else mp.mpf(wmax)

    def integrand(w):
        jw = J(float(w))
        if quantum:
            return 1.0 / mp.pi * mp.mpf(jw) / mp.tanh(beta * w / 2.0)
        return 2.0 / (mp.pi * beta) * mp.mpf(jw) / w

    return float(mp.quad(integrand, [0, wmax]))


# ===========================================================================
# Observables
# ===========================================================================
@njit
def get_expval_ABrho0(C: np.ndarray,
                      A: np.ndarray,
                      B_rho0: np.ndarray,
                      basis_list: np.ndarray) -> np.ndarray:
    """
    Compute ⟨A(t) B(0)⟩ = Tr[Aρ(t)] given
    C:          (Nt, dim2, dim2)  Liouville-space correlation matrices
    A:                (dim, dim)  Operator A in Hilbert space
    B_rho0:           (dim, dim)  Operator B·ρ(0) in Hilbert space
    basis_list: (dim2, dim, dim)  Liouville basis |i><j|
    """
    dim2 = C.shape[1]
    assert dim2 == len(basis_list), \
        "The dimension of C and basis_list should match."
    aa = np.zeros(dim2, dtype = np.complex128)
    bb = np.zeros(dim2, dtype = np.complex128)
    # project A and B_rho0 onto Liouville basis
    for i in range(dim2):
        # A wrong formula for Pupulation/autocorr/xingcorr at first
        # aa[i] = np.trace(np.dot(A, basis_list[i]))
        # bb[i] = np.trace(np.dot(B_rho0, basis_list[i].T.conj()))
        aa[i] = np.trace(np.dot(A, basis_list[i].T.conj()))
        bb[i] = np.trace(np.dot(basis_list[i], B_rho0))
    Nt = C.shape[0]
    res = np.zeros(Nt, dtype = np.complex128)
    for t_idx in range(Nt):
        for j in range(dim2):
            for k in range(dim2):
                res[t_idx] += aa[j] * C[t_idx, j, k] * bb[k]
    return res


def do_fft(t, y, zero_padding = 0, nblocks = 20, inversew: bool = False):
    dt = t[1] - t[0]
    if zero_padding > 0:
        y_blockd = np.array_split(y, nblocks)
        y_blockd_avg = np.array([np.mean(y_block) for y_block in y_blockd])
        y_last = y_blockd_avg[-1]
        y_first = y[0]
        rate_est = (np.abs(y_last) - np.abs(y_first)) / (t[-1] - t[0])

        zeros = y_last * np.exp(np.arange(1, zero_padding + 1)  * rate_est * dt)
        y = np.concatenate((y, zeros))

    w = np.fft.fftfreq(len(y), t[1] - t[0]) * 2 * np.pi
    w = np.fft.fftshift(w)

    Y = np.fft.fft(y)
    Y = np.fft.fftshift(Y) * dt
    return -w if inversew else w, Y

# %%
# 1. Requirements
import mpmath as mp
mp.mp.dps = 31
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "./library"))
from qutip import expect, Qobj
from qutip.core.environment import OhmicEnvironment
from qutip.solver.heom import HEOMSolver
sys.pycache_prefix = os.path.abspath("./output/pycache")
from extended_heom import ExtendedHEOMSolver, build_ohmic_bath
from fsutils import load_config, save_time
from qutils import spin_boson, basis, Pauli, qubit, parse_coeffs
from pltools import parse_xlims


# 2. Common Configurations
## 2.1 Basis operators
X, Y       = Qobj(Pauli("x")), Qobj(Pauli("y"))
P_list     = [Qobj(basis(2, 0, kind = "ketbra")),
              Qobj(basis(2, 1, kind = "ketbra"))]
# Basis: |0><0|, |0><1|, |1><0|, |1><1|
basis_list = np.array([basis(2, (0, 0)), basis(2, (0, 1)),
                       basis(2, (1, 0)), basis(2, (1, 1))])
## 2.2 Initial system density matrices
rho0  = {"0": Qobj(basis(2, 0, kind = "ketbra")),
         "x": Qobj(qubit(theta = 2 * np.pi/3, kind = "ketbra"))}
## 2.3 Measured operator A and time-zero state Brho0  ->  <A(t) B(0)>
A     = {"pop": P_list, "+": Qobj(Pauli("+")), "-": Qobj(Pauli("-")),
         "corr": X,     "xorr": X}
Brho0 = {"pop": rho0["0"], "+": rho0["0"], "-": rho0["0"],
         "corr": X @ rho0["0"], "xorr": Y @ rho0["x"]}

def main(prefix: str):
    # 3. Parameter
    ## 3.1 Loading parameters from YAML
    Lmax          = 6
    params        = load_config(prefix)
    linear        = params["linear"]
    quadratic     = params["quadratic"]
    rescale       = params["scale"]
    lambd         = params["lambd"]
    theta_B       = params["theta_B"]
    wB            = params["wB"]
    wC            = params["wC"]
    Delta         = params["Delta"]
    epsilon       = params["epsilon"]
    beta          = params["beta"]
    anharmon      = float(params["anharmon"] * mp.sqrt(7/6))
    xlims         = parse_xlims(params["xlims"], reorder = True)
    ## 4.2 Rescale the parameters to avoid numerical issues
    beta /= rescale
    wC, Delta, epsilon, lambd, wB, anharmon = [x * rescale \
                                for x in [wC, Delta, epsilon, lambd, wB, anharmon]]
    tf     = xlims[0][1]
    dt     = 1e-3
    tscale = 1/rescale
    tlist  = np.linspace(0, tf,  int(tf/dt))
    wlist  = np.linspace(0, 100, int(1e3))
    ## 3.2 System Hamiltonian, Interaction Operator
    Hs, V = spin_boson(
        model   = "QSBM" if quadratic else "LSBM",
        epsilon = epsilon, Delta = Delta, nohalf = True)
    # 4. Calculation
    ## 4.1 Setup
    #### 4.1.1 Environment & Decompose
    if quadratic:
        poly_coeffs = parse_coeffs(
            [0.5 * wB * (theta_B ** 2 - 1.0),
             -(2.0 * lambd * wB) ** 0.5 * theta_B ** 2,
             lambd * theta_B ** 2],
            linear = linear, quadratic = quadratic)
        a2, a1, a0 = float(poly_coeffs[0]), float(poly_coeffs[1]), \
                     float(poly_coeffs[2])
        tmax_solver = tf * tscale
        fit_tmax    = min(tmax_solver, 30.0)
        alpha       = np.pi * lambd / wC / rescale ** 2
        eta, gamma, q2, env, approx_env, fit_info = build_ohmic_bath(
            alpha = alpha,
            wc    = wC,
            beta  = beta,
            tmax  = fit_tmax,
            Nr    = 6,
        )
    else:
        env = OhmicEnvironment(
            T     = 1 / beta,
            alpha = np.pi * lambd / wC,
            wc    = wC,
            s     = 1.0,
        )
        approx_env, fit_info = env.approximate(
            method   = "espira-I",
            tlist    = wlist,
            Nr       = 4,
            Ni       = 4,
            separate = True,
        )
    #### 4.1.2 Use the approximated environment to set up HEOM solver
    if quadratic:
        default_options = {
            "rtol": 1e-9,
            "atol": 1e-11,
            "method": "DOP853",
        }
        HEOMMats = ExtendedHEOMSolver(
            Qobj(Hs), Qobj(V), a0, a1, a2, eta, gamma, q2,
            max_depth = Lmax, options = default_options)
    else:
        default_options = {
            "nsteps": 1500,
            "store_states": True,
            "rtol": 1e-12,
            "atol": 1e-12,
            "method": "vern9",
            "progress_bar": "enhanced",
        }
        HEOMMats = HEOMSolver(
            Qobj(Hs), (approx_env, Qobj(V)),
            max_depth = Lmax, options = default_options)
    ## 4.2 Propagate
    #### - To calculate <A(t) B(0)>, we evolve Brho0 forward in time and
    ####   measure A at each time step
    # Auto-correlation <Sx(t)Sx(0)>: evolve Brho0["corr"]
    heom_corr = HEOMMats.run(Brho0["corr"], tlist * tscale)
    C_mu = expect(A["corr"], heom_corr.states)
    C_mu /= C_mu[0]
    # Cross-correlation <Sx(t)Sy(0)>: evolve Brho0["xorr"]
    heom_xorr = HEOMMats.run(Brho0["xorr"], tlist * tscale)
    C_x = expect(A["xorr"], heom_xorr.states)
    # Single-time quantities: evolve Brho0["+"]
    res_heom = HEOMMats.run(Brho0["+"], tlist * tscale)
    P0_t   = expect(A["pop"][0], res_heom.states)
    P1_t   = expect(A["pop"][1], res_heom.states)
    cohe_p = expect(A["+"], res_heom.states)
    cohe_m = expect(A["-"], res_heom.states)

    # 5. Save to npz
    save_time(tlist, np.column_stack((P0_t, P1_t)).real,
              "./output/data/" + prefix, "pop_heom.npz",
              identifier = "P", compress = True)
    save_time(tlist, cohe_p,
              "./output/data/" + prefix, "cohe_heom.npz",
              identifier = "S+", compress = True)
    save_time(tlist, cohe_m,
              "./output/data/" + prefix, "cohem_heom.npz",
              identifier = "S-", compress = True)
    save_time(tlist, C_mu,
              "./output/data/" + prefix, "corr_heom.npz",
              identifier = "⟨μ(t)μ(0)⟩", compress = True)
    save_time(tlist, C_x,
              "./output/data/" + prefix, "xorr_heom.npz",
              identifier = "⟨Sx(t)Sy(0)⟩", compress = True)


RUNLIST  = [
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
if __name__ == "__main__":
    for prefix in RUNLIST:
        main(prefix = prefix)

# %%

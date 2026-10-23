---
name: surface-hopping
description: FSSH surface-hopping reference dynamics for the spin-boson model (harmonic and anharmonic bath, linear and quadratic coupling)
license: MIT
compatibility: opencode
metadata:
  audience: maintainers
  workflow: github
---

# Surface-hopping reference dynamics

An independent, classical-nuclei reference for the same spin-boson model as the memory-kernel
calculation: the system is propagated on the adiabatic surfaces while the bath is sampled
classically.

## Model

The four cases `QHM` (quantum harmonic), `QAH` (quantum anharmonic), `CHM` (classical harmonic)
and `CAH` (classical anharmonic), with the Ohmic density
`J(omega) = pi*lambd/wC * omega * exp(-omega/wC)` discretized by the shared Gauss-Legendre modes.
The linear coupling is `V = sigma_z`; the quadratic case (`K-linear_quad`) uses `V = |1><1|`.
The anharmonic quartic coefficient is `a = anharmon * sqrt(7/6)`.

The initial bath is sampled per mode: a **quantum** case draws from the Wigner/Gaussian
distribution with the quantum thermal variance, a **classical** case from the classical
(Boltzmann) variance, which lies below the quantum zero-point.

## Methods

- `library/surface_hopping/fssh.py` — fewest-switches
  surface hopping (the enabled method).
- `library/surface_hopping/mash.py` — the mapping approach to
  surface hopping; both methods are enabled (`METHODS = ["fssh", "mash"]` in `sh.py`) and share
  the same `n_jobs` slice parallelization.
- The drivers use numba-jitted kernels; the trajectory loop is the expensive part.

`sh.py` runs the selected methods and cases for each config in its `RUNLIST` and writes
`output/data/<prefix>/{pop,cohe,cohem,corr,xorr}_<method>_<case>.npz` (the shared `save_time`
schema, `<method>` = `fssh` or `mash`).

## Conventions

- Run from `./source` with the `sh` conda environment active.
- Selection is by the `METHODS` / `CASES` lists in `sh.py` (comment out entries) or by the
  `-method` CLI flag; `-n_traj` defaults to the per-method `N_TRAJ` budget
  (`mash` 40000, `fssh` 20000).  Flags: `-n_traj -N -wmax_factor -dt -seed`.
- Parallelism: `-nproc` (default 4) runs the cases as separate processes; `-njobs` (default -1)
  splits each case's ensemble over inner workers, sharing the cores over the cases.  The workers
  return unnormalised sums and the parent process writes every archive, so no two processes write
  the same file.  With one case per process (`nproc >= #cases`) the inner pool is skipped, since
  nested pools only add overhead.
- **Step size:** harmonic runs use `-dt` (default 0.005); the quartic stiffening raises the fastest
  effective frequency, so an anharmonic run caps the step at `dt <= 0.15 / w_max` (e.g. 0.00125 for
  the fast-bath scenario, where `w_max = 120`).  Without the cap a rare high-energy trajectory
  diverges at the Verlet stability limit.
- **Caveat:** classical nuclei, so this is a cross-check rather than an exact reference;
  it is reliable mainly for slow baths and high temperature.
- The shared pieces (`reference.py`, `config/`, `qutils.py`, `fsutils.py`, `pltools.py`, `extended_deom.py`) are copies of
  the canonical files in `mkct4oqs`; after a change there, run `tools/sync_from_mkct.sh` and
  commit.
- System-wide conventions (repo etiquette, Python style, git/PR, LaTeX, tooling) are in the
  maintainer's workspace `repo-rules` skill.

## Workflows

```bash
cd source
python sh.py A-base_line -method fssh          # one config, all four cases
python sh.py A-base_line -method fssh -n_traj 20000 -N 200
python library/surface_hopping/validation.py   # regression suite
```

CI: `.github/workflows/ci.yaml` (syntax + regression suite) and
`.github/workflows/data.yaml` (manual matrix over `<config> x <case> x <method>`).  Each data job
uploads `data/<config>/*.npz` as `sh-<config>-<case>-<method>`, and a `combine` job packs the whole
run into the `sh-data` tarball (retained 30 days).  Defaults: `configs=A-base_line` (the other
scenarios stay commented out in `RUNLIST` while the pipeline is being tuned), `methods="fssh mash"`.  The repository is public, so the runner minutes are free and
the full production data is generated on CI rather than locally:

```bash
gh workflow run data.yaml                       # defaults: A-base_line x 4 cases x 2 methods
gh run watch <id> --exit-status
gh run download <id> -n sh-data -D /tmp
tar -xzf /tmp/sh-data/sh-data.tar.gz -C source/output/data/   # -> source/output/data/<config>/
```

Inputs: `configs`, `cases`, `n_traj` (empty: the `N_TRAJ` budget), `N`, `njobs`.  There is no
automatic trigger, because the configs are owned by the other repository.

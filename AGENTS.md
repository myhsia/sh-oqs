# sh-oqs

Standalone **surface-hopping** reference dynamics for the spin-boson model: FSSH (MASH retained
but disabled) with a classical bath sampled per mode, for the harmonic/anharmonic and
linear/quadratic coupling cases.  The companion repositories are the private memory-kernel code
(`mkct-oqs`) and the public `heomXdmrg-spin_boson` (HEOM / MPS-tDMRG, which also owns the shared
`source/config/` and `source/library/{qutils,fsutils,pltools}.py`).

Scientific background and methodology are described in
[`.opencode/skills/SKILL.md`](./.opencode/skills/SKILL.md); load the `surface-hopping` skill.

## Layout

All code is under `source/`, the working directory for the drivers (paths such as `./config` and
`./output` are relative to it):

- `source/sh.py` — driver: runs the selected methods/cases for a config and writes
  `output/data/<prefix>/{pop,cohe,cohem,corr,xorr}_<method>_<case>.npz`.
- `source/library/surface_hopping/` — `fssh.py` (fewest-switches), `mash.py` (disabled),
  `validation.py` (regression suite).
- `source/environment.yaml`, `source/requirements.txt` — the `sh` conda environment.

Shared, non-sensitive pieces owned by the public `heomXdmrg-spin_boson` repository and symlinked
here (gitignored): `source/config/` and `source/library/{qutils,fsutils,pltools}.py`.  CI checks
that repository out anonymously and copies them (no token).

## Conventions

- Run everything from `source/` with the `sh` environment active.
- Four cases: `QHM`, `QAH`, `CHM`, `CAH`; the anharmonic coefficient is `a = anharmon * sqrt(7/6)`.
- The classical cases use the classical per-mode variance (below the quantum zero-point).
- Linear coupling `V = sigma_z`; quadratic (`K-linear_quad`) `V = |1><1|`.
- Selection by `METHODS`/`CASES` or `-method`/`-case`; trajectory budgets `N_TRAJ` (FSSH 20000,
  MASH 40000); flags `-n_traj -N -wmax_factor -dt -seed`.
- Classical nuclei: a cross-check, reliable mainly for slow baths and high temperature.
- Coding style: see the maintainer's shared `repo-rules` skill.

## Workflows

```bash
cd source
python sh.py A-base_line -method fssh
python library/surface_hopping/validation.py
```

## CI

- `.github/workflows/ci.yaml` — syntax check and the regression suite.
- `.github/workflows/data.yaml` — manual matrix over `<config> x <case>`; artifacts
  `sh-<config>-<case>`.  Retrieve with `gh run download <id> -D <tmp>`.

## Git style

See the maintainer's shared `repo-rules` skill (`type(scope): summary`; commit only when asked;
never commit secrets or `output/`).

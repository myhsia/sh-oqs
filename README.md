# sh-oqs

Reference **surface-hopping** dynamics for the spin-boson model used by the MKCT manuscript:
fewest-switches surface hopping (FSSH) with classical bath nuclei, covering the quantum/classical
and harmonic/anharmonic cases with linear and linear-quadratic coupling.

## Install

```bash
conda env create -f source/environment.yaml
conda activate sh
```

## Run

From `source/`:

```bash
python sh.py A-base_line -method fssh            # all four cases
python sh.py A-base_line -method fssh -n_traj 20000 -N 200
```

Results are written to `output/data/<prefix>/{pop,cohe,cohem,corr,xorr}_fssh_<case>.npz`
(`t`, `y`, `identifier`).

## Regression suite

```bash
python library/surface_hopping/validation.py
```

## Shared pieces

`source/config/` and `source/library/{qutils,fsutils,pltools}.py` are owned by the public
[heomXdmrg-spin_boson](https://github.com/myhsia/heomXdmrg-spin_boson) repository and symlinked
locally (gitignored); CI copies them from there.

## Environment

`conda env create -f source/environment.yaml` builds the `sh` environment (pinned interpreter,
unpinned `requirements.txt`).

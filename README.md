# sh4oqs

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

The four cases run as parallel processes (`-nproc`, default 4) and, when there are spare cores,
each case splits its ensemble over inner workers (`-njobs`, default -1: share the cores).

Results are written to `output/data/<prefix>/{pop,cohe,cohem,corr,xorr}_fssh_<case>.npz`
(`t`, `y`, `identifier`).

## CI data generation

The full production run is cheaper on GitHub Actions: dispatch the `Data` workflow and download
the results (public repository, so the runner minutes are free).

```bash
gh workflow run data.yaml                     # A-base_line x 4 cases x {fssh, mash}
gh run watch <id> --exit-status
gh run download <id> -n sh-data -D /tmp       # or one sh-<config>-<case> artifact
tar -xzf /tmp/sh-data/sh-data.tar.gz -C source/output/data/   # -> source/output/data/<config>/
```

Each job handles one `<config> x <case> x <method>` and uploads `data/<config>/*.npz`; the
`combine` job packs every job into the single `sh-data` tarball (retained 30 days).  Both methods
(FSSH and MASH) are enabled; the other scenarios are commented out in `RUNLIST` for now.

## Regression suite

```bash
python library/surface_hopping/validation.py
```

## Shared pieces

`source/reference.py`, `source/config/` and `source/library/{qutils,fsutils,pltools,extended_deom}.py` are copies of
the canonical files in the private `mkct4oqs` repository.  After a change there, run
`bash tools/sync_from_mkct.sh` and commit (the public
[dmrg4oqs](https://github.com/myhsia/dmrg4oqs) mirror does the same).

## Environment

`conda env create -f source/environment.yaml` builds the `sh` environment (pinned interpreter,
unpinned `requirements.txt`).

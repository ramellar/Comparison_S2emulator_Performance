# Tau workflow

Run commands from the repository root, with the `s2_emulator` Conda
environment active. Input and output locations are configured in
`configs/config.py` (`PARQUET_BASE`).

The workflow is:

```text
ROOT ntuples -> load_data --tau -> merge event Parquets
             -> matching_test -> merge matching parts -> plots
```

## 1. Load ROOT files

The input directory is `<base_path>/TauTau_PU200/` and must contain files
named `ntuple_<index>.root`. `--n` is the event limit **per job**.

For a small local test:

```bash
python3 -m scripts.load_data \
  -n 100 --particles TauTau --pileup PU200 \
  --base_path /path/to/root/files --n_files 1 --n_jobs 1 --tau
```

For the configured HTCondor submission:

```bash
condor_submit submit_load.sub
```

Check `submit_load.sub` and `run_load_data.sh` before submitting: the wrapper
expects a proxy path as its first argument, but the submit file does not
currently provide one. Fix this mismatch and set the input path before
launching jobs. Wait for all jobs to succeed, then merge their chunks:

```bash
python3 -m data_handling.stich_parquets --particles TauTau --pileup PU200
```

The merged event files are written to
`<PARQUET_BASE>/TauTau_PU200_new_branch/`. The merge removes the chunk files
from `single_jobs/`, so run it only after all expected loading jobs finish.

## 2. Match taus to clusters or jets

Choose `gen_cluster` for single clusters or `antikt_jets` for anti-kT jets.
For a local run (one job):

```bash
python3 -m scripts.matching_test \
  --particles TauTau --pileup PU200 --gen_pt_cut 20 \
  --matching_type gen_cluster --n_jobs 1
```

`submit_matching.sub` is configured for 20 anti-kT jobs with `gen_pt_cut=20`
and `n_events=99900`:

```bash
condor_submit submit_matching.sub
```

When all jobs finish, merge their output using the **same** matching type:

```bash
python3 -m data_handling.stitch_matching_parquets \
  --particles TauTau --pileup PU200 --gen_pt_cut 20 \
  --matching_type antikt_jets
```

Use `gen_cluster` instead if that was the strategy used. The output directory
is `<PARQUET_BASE>/TauTau_PU200_new_branch/gen_pt_cut20.0_GeV/`; matching
results are tagged `_cluster` or `_jet`. The merge removes files in `parts/`.
Use `--overwrite` only when intentionally replacing merged results.

## 3. Make plots

For example, plot matched response profiles:

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20 \
  --matching_type antikt_jets --matched --resolution_plots
```

Replace `antikt_jets` with `gen_cluster` to plot single-cluster matching.
Useful plot flags include `--distribution`, `--scale_distribution`,
`--resolution_plots`, `--two_d_dist`, `--efficiency`, and
`--binned_distributions`. Output is saved under `plots_PU200/`, in a folder
for the matching type and pT selection. Use `--events` instead of `--matched`
for plots from the loaded, unmatched event files.

For all options:

```bash
python3 -m scripts.run_new_perfomance_plots --help
```

## Calibration

`scripts.apply_calib` supports `offset` and `MC` calibration, while plotting
supports `--pt_type raw`, `offset`, or `mc`. However, `apply_calib` currently
expects unsuffixed matching files, whereas matching produces `_cluster` or
`_jet` files. Apply calibration only after resolving that filename mismatch.

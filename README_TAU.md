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

Replace `antikt_jets` with `gen_cluster` to plot single-cluster matching. Choose the input sample with `--matched` (matched gen-particle/object pairs),
`--events` (loaded events, without matching), or `--filtered_events` (events retained by the matching selection). The plot flags produce:

- `--distribution`: 1D distributions of the selected sample's kinematic variables, such as pT, eta, phi and delta-R (where available).
- `--scale_distribution`: 1D response distributions for matched objects: pT, eta and phi response.
- `--resolution_plots`: response profiles versus generated pT, absolute eta and phi. For each response variable (pT, eta, phi), it produces the mean, resolution and RMS profiles.
- `--binned_distributions`: response distributions split into bins of generated pT, eta or phi, for each of the pT, eta and phi responses.
- `--two_d_dist`: 2D correlations between kinematic variables of the selected sample.
- `--efficiency`: matching efficiency versus generated pT, absolute eta and phi; requires `--matched`.
- `--n_clusters_plots`: cluster-multiplicity distributions and profiles versus generated pT and absolute eta.
- `--mult_decaymode`, `--distributions_per_decaymode`,
  `--response_per_decaymode`, `--profile_per_decaymode`: cluster multiplicity, kinematic distributions, response distributions, and response profiles, respectively, split by generated tau decay mode.
- `--beta-eta-plot`: saved tau-calibration beta points and their fit. `--diagnostic` produces this plot as well as anti-kT rho and area distributions; both flags require `--matched`.
- `--mc-response-2d`: saved MC response maps versus absolute eta.
- `--custom_tau_plot`: custom-label generated-tau pT spectrum; requires `--events`.

Output is saved under `plots_PU200/`, in a folder for the matching type and pT selection.

For all options:

```bash
python3 -m scripts.run_new_perfomance_plots --help
```

## Calibration

After producing the anti-kT matching files, apply the offset calibration with:

```bash
python3 -m scripts.apply_calib \
  --particles TauTau --pileup PU200 --calibration offset
```

To apply both the offset and the additional MC response correction, use:

```bash
python3 -m scripts.apply_calib \
  --particles TauTau --pileup PU200 --calibration MC
```

`MC` performs the offset correction first and then derives/applies the MC
response correction, so it does not require running the `offset` command
beforehand. The script reads the anti-kT matched files
`pair_cluster_<config>_matched_jet.parquet` and
`pair_gen_masked_<config>_jet.parquet`. It writes the corrected pT fields back
into the matched cluster file (offset: `pt_corrected`; MC: also
`pt_corr_tot`); it does not create a separate matched file for each method.
Calibration outputs are written in the Parquet directory configured by
`PARQUET_BASE` in `configs/config.py`.

When plotting the matched anti-kT sample, select which pT field to use with
`--pt_type`:

```bash
# Offset-corrected pT
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --matching_type antikt_jets \
  --matched --gen_pt_cut 20 --pt_type offset --resolution_plots

# Offset + MC-corrected pT
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --matching_type antikt_jets \
  --matched --pt_type mc --resolution_plots
```

For response plots use the command 

```bash
# Offset-corrected pT
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --matching_type antikt_jets \
  --matched --gen_pt_cut 20 --pt_type offset --scale_distribution

# Offset + MC-corrected pT
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --matching_type antikt_jets \
  --matched --pt_type mc --scale_distribution
```

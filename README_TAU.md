# Tau workflow: commands grouped by parser

This guide collects the commands for the repository's `TauTau` workflow. Run the commands from the project root directory:

```bash
cd /home/llr/cms/pivato/Comparison_S2emulator_Performance
```

The workflow normally follows this sequence:

```text
ROOT ntuples -> load_data --tau -> Parquet -> matching_test
             -> stitch_matching_parquets -> apply_calib
             -> run_new_perfomance_plots
```

## 0. Environment

Activate the Conda environment used by the scripts:

```bash
conda activate s2_emulator
#export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

<!--
Accessing ROOT files on EOS also requires a valid proxy:

```bash
export X509_USER_PROXY=/percorso/al/proxy.pem
voms-proxy-info -all
```
-->

Input and output paths are defined in `configs/config.py`, especially
`PARQUET_BASE`.

## 1. Parser `scripts.load_data`

Loads ROOT ntuples, selects hadronic taus in the endcap, and saves the data as Parquet. The `--tau` flag is required for the Tau workflow.

### Local test command

```bash
python3 -m scripts.load_data \
  -n 100 \
  --particles TauTau \
  --pileup PU200 \
  --base_path /percorso/alla/directory/root \
  --n_files 1 \
  --job_id 0 \
  --n_jobs 1 \
  --tau
```

### Parser options

| Option | Default | Purpose |
|---|---:|---|
| `-n` | `1` | Number of events to read |
| `--particles` | `photons` | Sample name; use `TauTau` for taus |
| `--pileup` | `PU0` | Pileup scenario, normally `PU200` |
| `--base_path` | Path configured in the script | Directory containing ROOT files |
| `--name_tree` | `l1tHGCalTriggerNtuplizer/HGCalTriggerNtuple` | TTree name |
| `--pt_cut` | `0` | Cluster pT threshold |
| `--n_files` | `10` | Number of available ROOT files |
| `--job_id` | `0` | Job index |
| `--n_jobs` | `1` | Total number of jobs |
| `--tau` | disabled | Enables Tau selection and variables |

Each job writes its files to:

```text
<PARQUET_BASE>/TauTau_PU200_new_branch/single_jobs/
```

### HTCondor submission

`submit_load.sub` is already configured for `TauTau`, `PU200`, `10` jobs, and
the `--tau` flag:

```bash
condor_submit submit_load.sub
```

Before submitting, check `ProxyPath`, `--n`, `--n_files`, `--n_jobs`, and the
`queue` count. The `run_load_data.sh` wrapper treats its first argument as the
proxy path; to run the wrapper directly:

```bash
./run_load_data.sh "$X509_USER_PROXY" \
  -n 100 --particles TauTau --pileup PU200 --n_files 1 --tau
```

The loader builds the input directory by concatenating `--base_path` with
`<particles>_<pileup>`, then reads files named `ntuple_<index>.root` from
there. The `--n_files` input files are divided among `--n_jobs` partitions;
each Condor process uses its `$(Process)` value as `--job_id`. The `-n` limit
applies to each job, not to the total number of events across the queue.

Each successful job writes the collections `events_gen`, `events_0p0113`,
`events_0p016`, `events_0p03`, `events_0p045`, and `events_Ref` as
`*_part_<job_id>.parquet` files under:

```text
<PARQUET_BASE>/TauTau_PU200_new_branch/single_jobs/
```

Monitor the jobs with `condor_q` and inspect their matching `.out` and `.err`
files in `logs/`. Recover failed job IDs before merging. Once all expected
jobs have succeeded, merge the loading chunks:

```bash
python3 -m data_handling.stich_parquets \
  --particles TauTau \
  --pileup PU200
```

This writes the merged `events_*.parquet` input files into the parent
`TauTau_PU200_new_branch/` directory, which the matching step uses.

> **Current submit/wrapper check:** `submit_load.sub` passes `$(ProxyPath)` as
> the first argument but does not currently define `ProxyPath`. Also,
> `run_load_data.sh` exports its first argument as `X509_USER_PROXY` but
> forwards the full argument list to Python, including that proxy path. Define
> the proxy in the submit file and ensure the wrapper removes the proxy
> argument before calling `scripts.load_data`; otherwise Condor jobs can fail
> before loading data.

To recover failed jobs, update the IDs in `recovery.sub` and run:

```bash
condor_submit recovery.sub
```

## 2. Parser `data_handling.stich_parquets`

Merges the Parquet files produced by the loading jobs into complete files.

```bash
python3 -m data_handling.stich_parquets \
  --particles TauTau \
  --pileup PU200 \
  --n_files 333 \
  --n_jobs 10
```

Main options are `-n`, `--particles`, `--pileup`, `--base_path`,
`--name_tree`, `--pt_cut`, `--n_files`, `--job_id`, and `--n_jobs`.

## 3. Parser `scripts.matching_test`

Matches generated taus to clusters, or first reconstructs anti-$k_t$ jets and
uses those for matching.

### Local matching

```bash
python3 -m scripts.matching_test \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --matching_type gen_cluster \
  --deltaR 0.2 \
  --n_jobs 1
```

`--matching_type gen_cluster` matches the generator taus directly to the
reconstructed single clusters. `--matching_type antikt_jets` first reclusters
the single-cluster inputs into anti-kT jets, then matches the generator taus
to those jets. Choose one of these parser values exactly; the output suffixes
are `_cluster` and `_jet`, respectively, so both results can be kept side by
side. The bare ROOT-loaded event Parquets are shared inputs and are not
overwritten by either matching strategy.

### Matching split across jobs

```bash
python3 -m scripts.matching_test \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --matching_type antikt_jets \
  --job_id 0 \
  --n_jobs 20 \
  --n_events 99900
```

| Option | Default | Purpose |
|---|---:|---|
| `--particles` | `photons` | Input sample |
| `--pileup` | `PU0` | Input pileup |
| `--pt_cut` | `0` | Cluster pT threshold |
| `--gen_pt_cut` | `0` | Generated-tau pT threshold |
| `--deltaR` | `0.2` | Maximum matching distance |
| `--matching_type` | `gen_cluster` | Either `gen_cluster` or `antikt_jets` |
| `--total_efficiency` | disabled | Computes the integrated efficiency |
| `--only_efficiency` | disabled | Uses already-saved matching results |
| `--job_id` | `0` | Job index |
| `--n_jobs` | `10` | Total number of jobs |
| `--n_events` | all | Limits the events before splitting |

### Matching on HTCondor

`submit_matching.sub` is configured for `TauTau`, `PU200`, `20` jobs, and
`gen_pt_cut=20`. Its `arguments` line selects the matching type. Set
`--matching_type gen_cluster` for single-cluster matching or
`--matching_type antikt_jets` for anti-kT matching. The `--n_jobs 20` value
must agree with `queue 20`:

```bash
condor_submit submit_matching.sub
```

Wait until all matching jobs finish successfully before running the merge.
The chunks are written to the matching output directory's `parts/` subfolder
and include the matching suffix, for example
`pair_cluster_0p03_matched_cluster_part_0.parquet` or
`pair_cluster_0p03_matched_jet_part_0.parquet`.

## 4. Parser `data_handling.stitch_matching_parquets`

After all matching jobs have finished, merges the results stored in `parts/`:

```bash
python3 -m data_handling.stitch_matching_parquets \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --matching_type gen_cluster
```

Use `--matching_type antikt_jets` for anti-kT chunks. To intentionally replace
existing final files, add `--overwrite`:

```bash
python3 -m data_handling.stitch_matching_parquets \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --matching_type antikt_jets \
  --overwrite
```

The merge writes the final Parquet files with the same `_cluster` or `_jet`
suffix and removes merged chunks from `parts/`. Always pass the same
`--matching_type` to the merge as was used to create those chunks.

## 5. Parser `scripts.apply_calib`

Applies Tau calibration to matched clusters. The matching files must already
have been merged.

### Offset calibration

```bash
python3 -m scripts.apply_calib \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --calibration offset
```

### Offset + MC calibration

```bash
python3 -m scripts.apply_calib \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --calibration MC
```

The parser accepts `--particles`, `--pileup`, `--pt_cut`, `--gen_pt_cut`, and
`--calibration {offset,MC}`. The script adds `pt_corrected` to the
matched cluster files and saves the Tau fits in the same directory. The
current `apply_calib.py` expects unsuffixed matching filenames; adapt that
reader before applying it to the new `_cluster` or `_jet` outputs.

## 6. Parser `scripts.run_new_perfomance_plots`

Generates plots using raw, offset, or MC results. For results produced by
`apply_calib.py`, use `--pt_type offset` or `--pt_type mc`.

### Basic plots

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 \
  --gen_pt_cut 20.0 --matching_type gen_cluster \
  --matched --distribution --pt_type raw
```

Run matched plots once per matching strategy, changing
`--matching_type gen_cluster` to `--matching_type antikt_jets` for the
anti-kT results. The script reads the matching files with the corresponding
`_cluster` or `_jet` suffix and writes plots to separate folders:

```text
plots_PU200/matched_cluster/single_clusters/pt_raw/
plots_PU200/matched_cluster/antikt_jets/pt_raw/
```

Use `--pt_type offset` or `--pt_type mc` only when those calibrated fields
exist in the matching files. Otherwise use `raw`.

### Total efficiency printed by matching

To print total particle- and event-level efficiency while running matching,
use `--total_efficiency`. For a full local run, set `--n_jobs 1`; the default
splits the input into 10 jobs and one invocation would process only one part.
Select the desired matching strategy:

```bash
python3 -m scripts.matching_test \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matching_type gen_cluster --n_jobs 1 --total_efficiency
```

Replace `gen_cluster` with `antikt_jets` to calculate the efficiency using
anti-kT jets.

### One-dimensional properties

Matched-event distributions (pT, eta, |eta|, phi, and deltaR):

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matching_type gen_cluster --matched --distribution
```

Use the same command with `--matching_type antikt_jets` for anti-kT matched
events.

Bare-event distributions use the loaded events, not matching results:

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 \
  --matching_type gen_cluster --events --distribution
```

Here `--matching_type` only selects the output subfolder; bare-event plots
read the same input Parquets for either value. They are saved under
`plots_PU200/bare_cluster/<matching-folder>/pt_raw/`.

### Two-dimensional distributions

Matched events:

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matching_type gen_cluster --matched --two_d_dist
```

Bare events:

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matching_type gen_cluster --events --two_d_dist
```

For matched plots, use `antikt_jets` to select anti-kT results. For bare plots,
the matching type only determines the output subfolder.

### Response and profile plots

Response distributions:

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matching_type gen_cluster --matched --scale_distribution
```

Profiles (mean, resolution, and RMS):

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matching_type gen_cluster --matched --resolution_plots
```

Replace `gen_cluster` with `antikt_jets` to plot anti-kT matching results.

### Efficiency plots

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matching_type gen_cluster --matched --efficiency
```

Use `--matching_type antikt_jets` for anti-kT efficiency. `--matched` is
required; do not add `--events` to this command.

### Binned response distributions

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matching_type gen_cluster --matched --binned_distributions
```

Again, replace `gen_cluster` with `antikt_jets` for anti-kT.

### Decay-mode plots

Cluster multiplicity by decay mode:

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matching_type gen_cluster --matched --mult_decaymode
```

Properties by decay mode:

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matching_type gen_cluster --matched --distributions_per_decaymode
```

Response by decay mode:

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matching_type gen_cluster --matched --response_per_decaymode
```

Profiles by decay mode:

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matching_type gen_cluster --matched --profile_per_decaymode
```

For each command, select `antikt_jets` instead to use anti-kT matching files.

### Anti-kT diagnostics

Only run this for anti-kT results, which produce the required rho and jet-area
diagnostic files:

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matching_type antikt_jets --matched --diagnostic
```

### Copy plots to a local desktop

Run `scp` from the local computer (not from the remote SSH session). To copy
all matched plots for one strategy:

```bash
scp -r pivato@llrui01.in2p3.fr:/home/llr/cms/pivato/Comparison_S2emulator_Performance/plots_PU200/matched_cluster/single_clusters/ ~/Desktop/
```

For anti-kT plots:

```bash
scp -r pivato@llrui01.in2p3.fr:/home/llr/cms/pivato/Comparison_S2emulator_Performance/plots_PU200/matched_cluster/antikt_jets/ ~/Desktop/
```

For the complete list of parser options:

python3 -m scripts.run_new_perfomance_plots --help

Note: this script uses `--pt_type` with an underscore; `run_calibration_plots`
uses `--pt-type` with a hyphen instead.

## 7. Parser `scripts.run_calibration_plots`

Compares the calibration strategies defined in `configs/config.py`. Select the
configurations in `COMPARISONS` first.

```bash
python3 -m scripts.run_calibration_plots \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --triangle 0p03 \
  --matched \
  --resolution_plots \
  --pt-type corrected \
  --tag TauTau_PU200
```

Use `--all_triangles` for all triangles. Available plots are `--distribution`,
`--scale_distribution`, `--resolution_plots`, `--binned_distributions`,
`--weights`, `--eta_residual`, and `--weight_table`. Limit the strategies, for
example, with:

```bash
python3 -m scripts.run_calibration_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --all_triangles --matched --resolution_plots \
  --strategies raw PU200_bounds --tag TauTau_comparison
```

For the complete parser options:

```bash
python3 -m scripts.run_calibration_plots --help
```

## 8. Parser `scripts.derive_calibration`

This parser derives the coefficients for the generic calibrations defined by
`CALIB_CONFIGS` and `STRATEGIES` in `configs/config.py`.

```bash
python3 -m scripts.derive_calibration \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0 \
  --pt_cut 0 \
  --offset 0
```

Options are `--particles`, `--pileup`, `--tag`, `--gen_pt_cut`, `--pt_cut`, and
`--offset`.

For the Tau calibration described in section 5, use `scripts.apply_calib`
instead; it directly implements the `offset` and `MC` calibrations on matched
results.

## Recommended complete order

```bash
# 1. Local or Condor loading
condor_submit submit_load.sub

# 2. Merge the loading Parquet files
python3 -m data_handling.stich_parquets \
  --particles TauTau --pileup PU200 --n_files 333 --n_jobs 10

# 3. Local or Condor matching
condor_submit submit_matching.sub

# 4. Merge the matching results
python3 -m data_handling.stitch_matching_parquets \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0

# 5. Calibration
python3 -m scripts.apply_calib \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 --calibration offset

# 6. Plot corrected pT
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matched --resolution_plots --pt_type offset
```

To quickly check the syntax of every parser without running the pipeline:

```bash
python3 -m scripts.load_data --help
python3 -m scripts.matching_test --help
python3 -m data_handling.stich_parquets --help
python3 -m data_handling.stitch_matching_parquets --help
python3 -m scripts.apply_calib --help
python3 -m scripts.run_new_perfomance_plots --help
python3 -m scripts.run_calibration_plots --help
python3 -m scripts.derive_calibration --help
```
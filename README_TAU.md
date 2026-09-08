# Tau workflow: commands grouped by parser

This guide collects the commands for the repository's `TauTau` workflow. Run
the commands from the project root directory:

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
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Accessing ROOT files on EOS also requires a valid proxy:

```bash
export X509_USER_PROXY=/percorso/al/proxy.pem
voms-proxy-info -all
```

Input and output paths are defined in `configs/config.py`, especially
`PARQUET_BASE`.

## 1. Parser `scripts.load_data`

Loads ROOT ntuples, selects hadronic taus in the endcap, and saves the data as
Parquet. The `--tau` flag is required for the Tau workflow.

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
  --matching_type antikt_jets \
  --deltaR 0.2
```

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

`submit_matching.sub` is configured for `TauTau`, `PU200`, anti-$k_t$, `20` jobs,
and `gen_pt_cut=20`:

```bash
condor_submit submit_matching.sub
```

## 4. Parser `data_handling.stitch_matching_parquets`

After all matching jobs have finished, merges the results stored in `parts/`:

```bash
python3 -m data_handling.stitch_matching_parquets \
  --particles TauTau \
  --pileup PU200 \
  --gen_pt_cut 20.0
```

To replace files that already exist:

```bash
python3 -m data_handling.stitch_matching_parquets \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 --overwrite
```

Options are `--particles`, `--pileup`, `--pt_cut`, `--gen_pt_cut`, and
`--overwrite`.

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
`pair_cluster_*_matched.parquet` files and saves the Tau fits in the same
directory.

## 6. Parser `scripts.run_new_perfomance_plots`

Generates plots using raw, offset, or MC results. For results produced by
`apply_calib.py`, use `--pt_type offset` or `--pt_type mc`.

### Basic plots

```bash
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 \
  --gen_pt_cut 20.0 --matching_type antikt_jets \
  --matched --resolution_plots --pt_type offset
```

### Useful Tau plots

```bash
# pT/eta/phi distributions by decay mode
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matched --distributions_per_decaymode --pt_type offset

# Response by decay mode
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matched --response_per_decaymode --pt_type offset

# Profiles by decay mode
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matched --profile_per_decaymode --pt_type offset

# Cluster multiplicity by decay mode
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matched --mult_decaymode --pt_type raw

# Anti-kT diagnostics and beta(|eta|) calibration
python3 -m scripts.run_new_perfomance_plots \
  --particles TauTau --pileup PU200 --gen_pt_cut 20.0 \
  --matched --diagnostic --pt_type offset
```

Efficiency and general-distribution plots use `--efficiency`, `--distribution`,
`--scale_distribution`, `--two_d_dist`, `--binned_distributions`, and
`--n_clusters_plots`, respectively. The selectable datasets are `--matched`,
`--events`, and `--filtered_events`.

View all options with:

```bash
python3 -m scripts.run_new_perfomance_plots --help
```

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
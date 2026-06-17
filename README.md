# Performance Studies for S2 Emulation Algorithms

This repository is based on [S2_emulator](https://github.com/mchiusi/S2_emulator) framework but is only focused on performances plots. Its goal is to compare different S2 emulator algorithms, each configured with varying triangle sizes, against the reference CMSSW simulation.
Currently, we are working with a semi-emulator setup. As such, this repository does not perform cluster reconstruction directly. Instead, it uses pre-produced Ntuples containing clusters generated both by specific triangle configurations and the CMSSW simulation. Finally, in addition to the [S2_emulator](https://github.com/mchiusi/S2_emulator) framework, these scripts use awkward arrays for the matching to make it easier to handle a big number of events.

----

## Project Structure

```text
project/
│
├── data_handling/
│   ├── calibration_functions.py   # Functions to derive calibration and class to apply it
│   ├── utils.py                   # Functions to build parquet files
│   ├── event_performaces.py       # Functions to load events into Awkward/Parquet and apply matching/gen cuts
│   ├── efficiency.py              # Functions to compute total efficiency
│   └── files.py                   # Utilities to load and save files as Parquet
│
├── plotting/       
│   └── run_plots.py       # Contains 1 class allowing to do all the possible plots
│
├── configs/
│   └── config.py           # Contains all the plotting/ calibration configurations
│
└── scripts/
    ├── load_data.py               # Main script to load events into Awkward/Parquet and apply matching/gen cuts
    ├── matching_test.py           # Main script to apply matching 
    ├── run_new_performance_plots.py   # Main script to run performance plots
    ├── run_calibration_plots.py   # Main script to run the calibration plots
    └── derive_calibration.py      # Main script to derive calibration parameters 
       
```

## Installation

To run the performance plots and associated notebooks, create and configure a Conda environment:

```Bash
conda create --name <env_name> python=3.13

pip install matplotlib
pip install scipy
pip install awkward
pip install uproot
pip install mplhep
pip install tqdm
pip install pyarrow
pip install pandas
```

> **Note:** If you need a connection to `lxplus`, you will also need to install `xrootd`:

```Bash
conda install -c conda-forge xrootd
```

## Configuration

- **Plotting & General Config:** Since multiple triangle sizes are evaluated, global configuration is handled via `config.py`.
- **Data Paths:** Modify `PARQUET_BASE` in `config.py` to set the directory where your local Parquet files will be stored.
- **Data Loading Branches:** Data loading requires specific branch definitions. You must manually configure the targeted branches directly inside `load_data.py`.

## Loading Datasets

Because processing raw datasets can be time-consuming, the recommended workflow is to load and skim them **once**, saving the outputs as Parquet files for rapid subsequent plotting.

### Local Testing (Small Scale)

To load a small test subset locally, run:

```Bash
python3 load_data.py -n 100 --particles Photon --pileup PU200
```

### HTCondor Batch Submission (Full Scale)

Loading full datasets locally will exhaust disk space and memory. Instead, submit the jobs to HTCondor.

1. Open `submit_load.sub` and update your Grid proxy path (`ProxyPath`).
2. Adjust the specific parameters inside `submit_load.sub` as required:
    - `-n`: Total number of events (calculated as number of ROOT files × entries per file).
    - `--n_jobs`: Number of parallel jobs (must match the `queue` count at the bottom).
    - Update the target particles and Pileup (PU) configurations inside the parenthesis block.

#### Example `submit_load.sub` Configuration:

```Ini, TOML
# LLR T3 specific configuration
universe = vanilla
executable = run_load_data.sh

# Arguments: ProxyPath, followed by python flags
arguments = $(ProxyPath) -n 291824 --particles $(Part) --pileup $(PU) --job_id $(Process) --n_jobs 10

# Transfer everything needed
transfer_input_files = scripts/, data_handling/, configs/

# Resource Requests
request_memory = 8G
request_cpus = 1

# LLR T3 specific queueing logic
T3Queue = short
WNTag = el9
include : /opt/exp_soft/cms/t3/t3queue |

ProxyPath = /home/llr/cms/amella/.globus/user_proxy.pem

# Logging
output = logs/load_$(Part)_$(PU)_$(Process).out
error  = logs/load_$(Part)_$(PU)_$(Process).err
log    = logs/load_$(Part)_$(PU)_$(Process).log

# The Queue
queue 10 Part,PU from (
  Photon, PU200
)
```

3. Submit the script:

```Bash
condor_submit submit_load.sub
```

### Job Recovery & Merging

If any batch jobs fail, specify the missing IDs in `recovery.sub` and resubmit them via:

```Bash
condor_submit recovery.sub
```

Once all batch chunks have successfully finished, stitch the standalone Parquet chunks into a single unified list:

```Bash
python3 -m data_handling.stitch_parquets --particles Photon --pileup PU200
```

## Applying Matching

To execute DeltaR matching between generated particles and reconstructed clusters (with an optional generator-level pT cut), run:

```Bash
python3 -m scripts.matching_test --particles Photon --pileup PU0 --gen_pt_cut 20.0
```

To compute and output the **total efficiency** metric, simply append the `--total_efficiency` flag:

```Bash
python3 -m scripts.matching_test --particles Photon --pileup PU0 --gen_pt_cut 20.0 --total_efficiency
```

## Deriving and Applying Calibrations

You can evaluate multiple calibration configurations by defining them inside `CALIB_CONFIGS` within `config.py`.

The core calibration formula is defined as:

$$E_T^{calib​}=l∑​w_l​E_T^l​−(α∣η∣-1.5+β)$$

Currently, **4 configurations** are available for **3 distinct strategies**:

### Calibration Strategies

1. **PU0**: Derives $w_l$ calibration weights from PU0 samples to fully account for energy not captured in the signal.
2. **PU200_seq**: Derives $\alpha$ and $\beta$ as η correction to account for pileup effects, using the PU0 weights as fixed layer weights.
3. **PU200**: Derives all coefficients simultaneously using PU200 samples.

For every strategy, the optimization coefficients are extracted using `lsq_linear` from `scipy.optimize` .

### Structural Configurations

Each strategy can be parsed into one of four setup options:

- `bounds`
- `no bounds`
- `bounds no layer 1`
- `no bounds no layer 1`
-
> **Note on PU200_seq:** There is a configuration option `PU0_CONFIG_FOR_SEQ="bounds_0_20"` which fixes the explicit configuration schema applied to the sequential PU200 framework. If you prefer to apply the exact same strategy (`bounds` vs `no bounds`) to both the PU0 layer baseline and the PU200 α,β parameters, set `PU0_CONFIG_FOR_SEQ="None"`.

### Running Calibration

To derive the calibration factors based on your configurations, run:

```Bash
python3 -m scripts.derive_calibration --particles Photon --pileup PU200 --gen_pt_cut 20.0
```

All calculated weights are automatically stored within the configured `parquet_path` and can be evaluated elsewhere using the `apply_functions` in `data_handling/calibration_functions.py`
## Plotting

The primary plotting allows for algorithm and triangle-size comparison. The syntax below is the baseline to run comparisons:

```Bash
python3 -m scripts.run_new_performance_plots --particles Photon --pileup PU0
```

- bare events are the events after loading them with condor
-  filtered events are a subset con the bare events containing the events where at least 1 cluster was reconstructed
- matched events are a subset of the filtered events containing the events where a cluster and a gen particle were matched

For targeted performance figures, add the respective specialized flags:

### 1D Distributions

```Bash
python3 -m scripts.run_new_performance_plots --particles Photon --pileup PU0 --matched --distribution
python3 -m scripts.run_new_performance_plots --particles Photon --pileup PU0 --events --distribution
python3 -m scripts.run_new_performance_plots --particles Photon --pileup PU0 --filtered_events --distribution
```

### 2D Distributions


```Bash
python3 -m scripts.run_new_performance_plots --particles Photon --pileup PU0 --matched --two_d_dist
python3 -m scripts.run_new_performance_plots --particles Photon --pileup PU0 --events --two_d_dist
python3 -m scripts.run_new_performance_plots --particles Photon --pileup PU0 --filtered_events --two_d_dist
```

### Resolution & Scale

```Bash
python3 -m scripts.run_new_performance_plots --particles Photon --pileup PU0 --matched --scale_distribution
python3 -m scripts.run_new_performance_plots --particles Photon --pileup PU0 --matched --resolution_plots
```

### Efficiency & Cluster Profiles

```Bash
# Efficiency vs kinematics
python3 -m scripts.run_new_performance_plots --particles Photon --pileup PU0 --matched --events --efficiency

# Total integrated efficiency values
python3 -m scripts.matching_test --particles Photon --pileup PU0 --matched --events --total_efficiency 

# Binned kinematics & Cluster multiplicity
python3 -m scripts.run_new_performance_plots --particles Photon --pileup PU0 --matched --binned_distributions
python3 -m scripts.run_new_performance_plots --particles Photon --pileup PU0 --events --n_clusters_plots
```

### Calibration Performance

To visualize calibration performance (such as layer weights or eta residuals), ensure you have run the derivation script first.

To choose which strategies and configurations to plot only `COMPARISONS` in `config.py` needs to be changed. 

```python
COMPARISONS = {
"PU200_no_bounds": {"strategy": "PU200", "all": "no_bounds"},
"PU200_bounds": {"strategy": "PU200", "all": "bounds_0_20"},
"PU200_all_bounds_0_20_no_layer1": {"strategy": "PU200", "all": "bounds_0_20_no_layer1"},
"PU200_all_no_bounds_no_layer1": {"strategy": "PU200", "all": "no_bounds_no_layer1"},
}
```

(other examples in `config.py`)

A `--tag` can be added at the end of the plots in order to save the chosen comparison with the correct name

```Bash
# Response (mean) and resolution vs pt_gen / abs_eta_gen for triangle 0p03:

python -m scripts.run_calibration_plots --particles Photon --pileup PU200 --gen_pt_cut 20 --triangle 0p03 --resolution_plots --tag PU200_1p5

# Scale (pT/pT_gen distribution) for every triangle:

python -m scripts.run_calibration_plots --particles Photon --pileup PU200 
--gen_pt_cut 20 --scale_distribution
  
# 1-D calibrated-pT distribution for triangle 0p03:

python -m scripts.run_calibration_plots --particles Photon --pileup PU200 --gen_pt_cut 20 --triangle 0p03 --distribution --tag PU200_1p5

  
# Plotting weights for every strategy, triangle 0p03:
python -m scripts.run_calibration_plots --particles Photon --pileup PU200 --gen_pt_cut 20 --triangle 0p03 --weights --tag PU200_1p5
  

# Eta residual vs abs(eta) with fit curve overlay, for every strategy, triangle 0p03:
python -m scripts.run_calibration_plots --particles Photon --pileup PU200 --gen_pt_cut 20 --triangle 0p03 --eta_residual --tag PU200_seq_1p5


# Compare all triangles for one strategy (pass --all_triangles):
python -m scripts.run_calibration_plots --particles Photon --pileup PU200 --gen_pt_cut 20 --all_triangles --resolution_plots


For all options: python -m scripts.run_calibration_plots --help
```

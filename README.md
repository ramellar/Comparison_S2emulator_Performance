# Performance studies for different S2 emulation algorithms

This repository is based on [S2_emulator](https://github.com/mchiusi/S2_emulator) framework but is only focused on performances plots. Its goal is to compare different S2 emulator algorithms, each configured with varying triangle sizes, against the reference CMSSW simulation.
Currently, we are working with a semi-emulator setup. As such, this repository does not perform cluster reconstruction directly. Instead, it uses pre-produced Ntuples containing clusters generated both by specific triangle configurations and the CMSSW simulation. Finally, in addition to the [S2_emulator](https://github.com/mchiusi/S2_emulator) framework, these scrpits use awkward arrays for the matching to make it easier to handle a big number of events.

### Installation

To run the performance plots we need to install an environment

```
conda create --name <env_name> python=3.13

pip install cppyy
pip install matplotlib
pip install scipy
pip install awkward
pip install uproot
pip install mplhep

conda install gxx
```

This environment can also de used to run the notebooks

### Useful commands 

```
# Only loading the data
python run_performance_plots.py -n 1000 --pileup PU0 --particles photons

# Getting the total efficiency numbers
python run_performance_plots.py -n 1000 --pileup PU0 --particles photons --total_efficiency

# Plotting the response 
 python run_performance_plots.py -n 1000 --pileup PU0 --particles photons --resp --response
 
# Plotting the response 
 python run_performance_plots.py -n 1000 --pileup PU0 --particles photons --resp --resolution
```

To display other options use `python run_performance_plots.py --help`.


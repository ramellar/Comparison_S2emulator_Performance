import os
import argparse
import awkward as ak
import data_handle.plot_performances as plot
from configs.config import PARQUET_BASE, EVENT_NAMES, CALIB_CONFIGS, EMU_CONFIG,STRATEGIES, PLOT_VARS
from   data_handling.utils import build_parquet_dir
import data_handling.files as f
import data_handling.calibration_functions as calib
from plotting.run_plots import DistributionPlotter
import copy

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Stage-2 Emulator Parameters')

    parser.add_argument('-n',          type=int, default=1,         help='Provide the number of events')
    parser.add_argument('--particles', type=str, default='Photon', help='Choose the particle sample')
    parser.add_argument('--pileup',    type=str, default='PU0',     help='Choose the pileup - PU0 or PU200')
    parser.add_argument('--pt_cut',    type=float, default=0,         help='Provide the cut for the cluster pt')
    parser.add_argument('--gen_pt_cut',    type=float, default=0,         help='Provide the cut for the gen pt')
    args = parser.parse_args()

    parquet_dir= build_parquet_dir(args)
    parquet_dir_bare_events= f"{PARQUET_BASE}{args.particles}_{args.pileup}_new_branch/"

    #Load matched events for calibration plots ‚
    matched_events= f.load_matching_results(parquet_dir)

    #Preparing calibration
    PU200_calibrated = calib.CalibrationManager(
    matched_events,
    parquet_dir,
    CALIB_CONFIGS,
    args
    )

    # Prepare dictionnary for plotting 
    cluster_calib={}
    weights={}
    for strategy in STRATEGIES:
        cluster_calib[strategy] = {}
        weights[strategy] = {}
        for calib_config in CALIB_CONFIGS:
            cluster_calib[strategy][calib_config] = {}
            weights[strategy][calib_config] = {}
            for key in EMU_CONFIG: 
                cluster_calib[strategy][calib_config][key], gen, weights[strategy][calib_config][key] = PU200_calibrated.get_calibrated_cluster( 
                    strategy=strategy, config_name=calib_config, 
                    key=key, 
                    args= args,
                    name= f"{strategy}_{calib_config}" ) 
    

                
    plotter = DistributionPlotter(args, output_dir=f"plots_calibration/{args.pileup}")

    # # --- Calibrations (Fixed Tri and Strategy) ---
    plot_data_dist = []
    vars_to_plot = ["pt_calib", "eta_calib", "phi_calib"]

    for tri in EMU_CONFIG.keys():
        for strat in STRATEGIES:
            for var in vars_to_plot:
                # --- STEP 1: Add the Uncalibrated baseline ---
                # We just pick the cluster object from any existing calib_config 
                # (since they all contain the same raw .pt)
                raw_data_source = matched_events[tri]["pair_cluster"]

                plot_data_dist.append({
                    'data': raw_data_source,
                    'branch': PLOT_VARS[var]["branch"],         
                    'label': 'Uncalibrated',
                    'color': 'black'        
                })

                # --- STEP 2: Add the Calibrated configs as usual ---
                for calib_name in CALIB_CONFIGS.keys():
                    plot_data_dist.append({
                        'data': cluster_calib[strat][calib_name][tri],
                        'branch': f"Ecalib_{strat}_{calib_name}",
                        'label': calib_name
                    })

                # --- STEP 3: Plot ---
                plotter.plot(plot_data_dist, var,
                            f"Calib_Comparison_{tri}_distribution_{var}", title=f"{strat} Calibration")
        

import os
import sys
import argparse
import copy
import awkward as ak
import numpy as np
import data_handling.calibration_functions as calib
from configs.config import EMU_CONFIG, CALIB_CONFIGS, STRATEGIES, PU0_CONFIG_FOR_SEQ
from data_handling.utils import build_parquet_dir
import data_handling.files as io

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Stage-2 Emulator Calibration')

    parser.add_argument('--particles', type=str, default='photons', help='Particle sample')
    parser.add_argument('--pileup', type=str, default='PU0', help='PU0 or PU200')
    parser.add_argument('--tag', type=str, default='', help='Unique tag for json files')
    parser.add_argument('--gen_pt_cut', type=float, default=0, help='Provide the cut for the gen pt') 
    parser.add_argument('--pt_cut', type=float, default=0, help='Provide the cut for the gen pt')
    args = parser.parse_args()

    #--------------------
    # Load directories for PU0 and PU200 samples
    #--------------------
    args_PU0 = copy.deepcopy(args) 
    args_PU0.pileup = "PU0" 
    args_PU200 = copy.deepcopy(args) 
    args_PU200.pileup = "PU200"
    
    output_dir_PU0 = build_parquet_dir(args_PU0)
    print("PU0",output_dir_PU0)
    output_dir_PU200 = build_parquet_dir(args_PU200)
    print("PU200",output_dir_PU200)

    results_PU0 = io.load_matching_results(output_dir_PU0)
    results_PU200 = io.load_matching_results(output_dir_PU200)

    output_dir = build_parquet_dir(args)


    for strategy in STRATEGIES:  # ["PU0", "PU200", "PU200_seq"]

        for config_name, cfg in CALIB_CONFIGS.items(): # ["no_bounds", "bounds_0_20", "no_bounds_no_layer1", "bounds_0_20_no_layer1"]

            bounds = cfg["bounds"]
            remove_layer1 = cfg["remove_layer1"]

            for key in EMU_CONFIG:

                # =========================================
                # Derive layer weights with PU0 sample
                # =========================================
                if strategy == "PU0":
                    cluster = results_PU0[key]["pair_cluster"]
                    gen = results_PU0[key]["pair_gen"]

                    w_layer = calib.derive_calibration(
                        cluster, gen,
                        mode="PU0",
                        bounds=bounds,
                        remove_layer1=remove_layer1
                    )

                    calib.save_weights(
                        w_layer, output_dir,
                        f"PU0_wl_{config_name}_{key}.parquet"
                    )

                # =========================================
                # Derive layer weights + eta correction with PU200 sample
                # =========================================
                elif strategy == "PU200":
                    cluster = results_PU200[key]["pair_cluster"]
                    gen = results_PU200[key]["pair_gen"]

                    w_all = calib.derive_calibration(
                        cluster, gen,
                        mode="PU200",
                        bounds=bounds,
                        remove_layer1=remove_layer1
                    )

                    calib.save_weights(
                        w_all, output_dir,
                        f"PU200_all_{config_name}_{key}.parquet"
                    )

                # =========================================
                # PU200_seq: Apply PU0 w and derive eta correction with PU200 sample
                # =========================================
                elif strategy == "PU200_seq":

                    # Decide PU0 config for wl
                    if PU0_CONFIG_FOR_SEQ is None:
                        PU0_cfg = cfg  # same as current PU200 config
                        PU0_cfg_name = config_name
                    else:
                        PU0_cfg = CALIB_CONFIGS[PU0_CONFIG_FOR_SEQ]
                        PU0_cfg_name = PU0_CONFIG_FOR_SEQ

                    # ---- Derive wl from PU0 ----
                    w_layer = calib.derive_calibration(
                        results_PU0[key]["pair_cluster"],
                        results_PU0[key]["pair_gen"],
                        mode="PU0",
                        bounds=PU0_cfg["bounds"],
                        remove_layer1=PU0_cfg["remove_layer1"]
                    )

                    # ---- Apply wl on PU200 ----
                    cluster_calib = calib.apply_calibration(
                        results_PU200[key]["pair_cluster"],
                        weights_layer=w_layer,
                        remove_layer1=PU0_cfg["remove_layer1"],
                        name="tmp"
                    )

                    # ---- Derive eta on PU200 ----
                    w_eta = calib.derive_calibration(
                        cluster_calib,
                        results_PU200[key]["pair_gen"],
                        mode="PU_eta",
                        bounds=bounds,
                        name="tmp"
                    )

                    # ---- Save weights ----
                    # calib.save_weights(
                    #     w_layer, output_dir,
                    #     f"PU200_seq_wl_{PU0_cfg_name}_{config_name}_{key}.parquet"
                    # )
                    calib.save_weights(
                        w_eta, output_dir,
                        f"PU200_seq_ab_{config_name}_with_PU0_{PU0_cfg_name}_{key}.parquet"
                    )

    manager = calib.CalibrationManager(
    results_PU0,
    results_PU200,
    output_dir,
    CALIB_CONFIGS,
    args
    )

    for key in EMU_CONFIG: 
        cluster_calib, gen, weights = manager.get_calibrated_cluster( 
            strategy="PU200_seq", config_name="bounds_0_20", 
            key=key, 
            args= args,
            name="PU200_seq" ) 
        Ecalib = getattr(cluster_calib, "Ecalib_PU200_seq") 
        print("Calibrated:", ak.flatten(Ecalib)[:10]) 
        print("Gen:", ak.flatten(gen.pt)[:10])
        print("weights:", weights)

    


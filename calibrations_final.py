import os
import sys
import argparse
import cppyy
import data_handle.plot_performances as plot
from   data_handle.event_performances import provide_events_performaces, apply_matching
import matplotlib
import matplotlib.pyplot as plt
import mplhep
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import json
plt.style.use(mplhep.style.CMS)
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import awkward as ak
from scipy.optimize import lsq_linear
from pathlib import Path

colors = ["tab:olive", "tab:cyan", "darkorchid" , "darkorange", "deeppink", "coral", "royalblue", "orangered"]
# my_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", colors)
my_cmap = ListedColormap(colors)
# cmap = plt.get_cmap("Set3")
n_colors = len(colors)

output_dir="plots/"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Stage-2 Emulator Parameters')
    parser.add_argument('--particles', type=str, default='photons', help='Choose the particle sample')
    parser.add_argument('--pileup',    type=str, default='PU0',     help='Choose the pileup - PU0 or PU200')
    parser.add_argument('--tag',       type=str, default='',        help='Name to make unique json files')
    parser.add_argument('--pt_cut',    type=float, default=0,         help='Provide the cut for the cluster pt')
    parser.add_argument('--gen_pt_cut',    type=float, default=0,         help='Provide the cut for the gen pt')
    #plotting arguments
    parser.add_argument('--resolution',     action='store_true', help='Extract mean and std from selecting the shorted interval containing 68& events')
    parser.add_argument('--response',     action='store_true', help='Extract mean and std from selecting the shorted interval containing 68& events')
    parser.add_argument('--distribution',     action='store_true', help='Plot the distributions')
    parser.add_argument('--eta',     action='store_true', help='Plot the distributions')
    parser.add_argument('--scale',     action='store_true', help='Plot the response distribution of the emulator')
    parser.add_argument('--PU0calib',     action='store_true', help='Plot the response distribution of the emulator')
    parser.add_argument('--PU200calib',     action='store_true', help='Plot the response distribution of the emulator')
    parser.add_argument('--PileUpcalib',     action='store_true', help='Plot the response distribution of the emulator')
    args = parser.parse_args()

    if args.pt_cut:
        # output_dir += f"results_performance_plots_{args.particles}_{args.pileup}_cut_{args.pt_cut}_GeV"
        parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/cluster_pt_cut{args.pt_cut}_GeV/'
        # os.makedirs(output_dir, exist_ok=True)
    elif args.gen_pt_cut:
        # output_dir += f"results_performance_plots_{args.particles}_{args.pileup}_gen_cut_{args.gen_pt_cut}_GeV"
        # os.makedirs(output_dir, exist_ok=True)
        parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/'
    elif args.gen_pt_cut and args.pt_cut:
        # output_dir += f"results_performance_plots_{args.particles}_{args.pileup}_gen_cut_{args.gen_pt_cut}_GeV_cut_{args.pt_cut}_GeV"
        # os.makedirs(output_dir, exist_ok=True)
        parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/cluster_pt_cut{args.pt_cut}_GeV_gen_pt_cut{args.gen_pt_cut}_GeV/'
    else:
        parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/'
        # output_dir += f"results_performance_plots_{args.particles}_{args.pileup}"
        # os.makedirs(output_dir, exist_ok=True)

    # if args.tag:
    #     output_dir += f"_{args.tag}"
    #     os.makedirs(output_dir, exist_ok=True)
    # parquet_dir += f"{args.tag}/"

    print(output_dir)

    print("Loading cluster info")

    print("opening", parquet_dir)

    pair_cluster_0p0113_matched = ak.from_parquet(parquet_dir+"pair_cluster_0p0113_matched.parquet")
    pair_cluster_0p016_matched = ak.from_parquet(parquet_dir+"pair_cluster_0p016_matched.parquet")
    pair_cluster_0p03_matched = ak.from_parquet(parquet_dir+"pair_cluster_0p03_matched.parquet")
    pair_cluster_0p045_matched = ak.from_parquet(parquet_dir+"pair_cluster_0p045_matched.parquet")
    pair_cluster_Ref_matched = ak.from_parquet(parquet_dir+"pair_cluster_Ref_matched.parquet")

    print("Loading gen info")
    pair_gen_masked_0p0113 = ak.from_parquet(parquet_dir+"pair_gen_masked_0p0113.parquet")
    pair_gen_masked_0p016 = ak.from_parquet(parquet_dir+"pair_gen_masked_0p016.parquet")
    pair_gen_masked_0p03 = ak.from_parquet(parquet_dir+"pair_gen_masked_0p03.parquet")
    pair_gen_masked_0p045 = ak.from_parquet(parquet_dir+"pair_gen_masked_0p045.parquet")
    pair_gen_masked_Ref = ak.from_parquet(parquet_dir+"pair_gen_masked_Ref.parquet")

    clusters = {
    "0p0113": pair_cluster_0p0113_matched,
    "0p016": pair_cluster_0p016_matched,
    "0p03": pair_cluster_0p03_matched,
    "0p045": pair_cluster_0p045_matched,
    "Ref": pair_cluster_Ref_matched,
    }

    gen_masked = {
    "0p0113": pair_gen_masked_0p0113,
    "0p016": pair_gen_masked_0p016,
    "0p03": pair_gen_masked_0p03,
    "0p045": pair_gen_masked_0p045,
    "Ref": pair_gen_masked_Ref,
    }

    os.makedirs(f"{output_dir}/calibration_scale", exist_ok=True)

    triangles = ["0p0113", "0p016", "0p03", "0p045", "Ref"]
    weights_PU0 = {}

    parquet_dir_wl= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU0_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV'

    calibration_configs_PU0 = {
    "fit_bounds_0_20": "PU0_bounds_0_20"
    }

    for calib_name, folder in calibration_configs_PU0.items():
        weights_PU0[calib_name] = {}
        for tri in triangles:
            path = f"{parquet_dir_wl}/{folder}/weight_wl_{tri}.parquet"
            weights_PU0[calib_name][tri] = ak.from_parquet(path)

    for tri in triangles:
        for calib_name in calibration_configs_PU0.keys():
            clusters[tri] = plot.apply_calibration(
            clusters[tri],
            weights_PU0[calib_name][tri],
            calib_name
        )

    calibration_configs= {
    "a_b_no_bounds": "wlab-ab_no_bounds_final"
    }


    parquet_dir_wl= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU200_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV'

    weights_a = {}
    for calib_name, folder in calibration_configs.items():
        weights_a[calib_name] = {}
        for tri in triangles:
            path = f"{parquet_dir_wl}/{folder}/weight_a_{tri}.parquet"
            weights_a[calib_name][tri] = ak.from_parquet(path)
        
    quantity_calib="calibPUterm"


    calibration_configs_applying = {
    "fit_bounds_0_20": "a_b_no_bounds"
    }

    for tri in triangles:
        for calib_name, weight_name in calibration_configs_applying.items():
            clusters[tri] = plot.apply_calibration_eta(
            clusters[tri],
            weights_a[weight_name][tri],
            calib_name,
            weight_name
        )
            
    weights_combined = {"a_b_no_bounds": {}}

    weights_combined = {"a_b_no_bounds": {}}

    for tri in triangles:

        pu0 = weights_PU0["fit_bounds_0_20"][tri]
        ab = ak.flatten(weights_a["a_b_no_bounds"][tri])
        weights_combined["a_b_no_bounds"][tri] = ak.concatenate([pu0, ab])

    print("weights_combined", weights_combined)

    weights={}
    parquet_dir_wl= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU200_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV'

    calibration_configs= {
    "fit_bounds_0_20": "PU200_all_bounds_0_20"
    }

    for calib_name, folder in calibration_configs.items():
        weights[calib_name] = {}
        for tri in triangles:
            path = f"{parquet_dir_wl}/{folder}/weight_w_{tri}_PU200.parquet"
            weights[calib_name][tri] = ak.from_parquet(path)
    
    quantity_calib="calibPU200"

    for tri in triangles:
        for calib_name in calibration_configs.keys():
            clusters[tri] = plot.apply_calibration_all_weights(
            clusters[tri],
            weights[calib_name][tri],
            calib_name
        )
            
    print(weights["fit_bounds_0_20"])
    print(weights_a)


    # plt.figure(figsize=(10,10))
    # for tri_idx, tri in enumerate(triangles):

    #     plot.plot_weight_val(weights, tri, my_cmap(1), "fit_bounds_0_20", 0, args)
    #     plot.plot_weight_val(weights_combined, tri, my_cmap(3), "a_b_no_bounds", 1, args)

    #     plt.savefig(
    #         f"Weightval_{tri}_PU200_vs_PileUpCalib_{args.tag}_{args.pileup}.png",
    #         dpi=300
    #     )

    #     plt.savefig(
    #         f"Weightval_{tri}_PU200_vs_PileUpCalib_{args.tag}_{args.pileup}.pdf"
    #     )

    #     plt.close()

    # # ------------------------------------------------------------
    # # Plotting
    # # ------------------------------------------------------------
    range_eta = [1.6, 2.9]
    range_phi = [-3.14, 3.14]
    range_pt = [20, 100]
    bin_eta = 10
    bin_phi = 10
    bin_pt = 10

    n_colors = len(calibration_configs) + 1

    if args.eta:
        print("hi")
        for tri in triangles:
            fig, ax = plt.subplots(figsize=(10, 10))
            plot.plot_eta(clusters[tri], weights_a["a_b_no_bounds"][tri], "tomato", tri, "PU0 bounds + a_b no bounds")
            plot.plot_eta(clusters[tri], weights["fit_bounds_0_20"][tri], "yellowgreen", tri, "PU200 bounds", 13, 14)
            plt.savefig(
                f"{output_dir}/calibration_final_comparison/Eta_{tri}_calib_comparison_{args.tag}_{args.pileup}.png",
                dpi=300
                )
            plt.close()


    if args.response or args.resolution:

        for tri in triangles:

            for response_variable, bin_variable, bin_, range_ in zip(
                ["pt", "pt", "pt"],
                ["pt", "eta", "phi"],
                [bin_pt, bin_eta, bin_phi],
                [range_pt, range_eta, range_phi]
                ):

                fig, ax = plt.subplots(figsize=(10, 10))

                # Plot raw pT
                plot.plot_response_calib(
                clusters[tri],
                gen_masked[tri],
                args,
                ax,
                f"pT for {tri}",
                my_cmap(0),
                response_variable= response_variable,
                bin_variable= bin_variable,
                quantity="raw",
                calib_name = None,
                bin_n= bin_,
                range_= range_
                )

                # Plot all calibrations
                
                # print("i",calib_name) 
                plot.plot_response_calib(
                clusters[tri],
                gen_masked[tri],
                args,
                ax,
                f"a_b_no_bounds",
                my_cmap(3),
                response_variable,
                bin_variable,
                "calibPUterm",
                "a_b_no_bounds",
                bin_,
                range_
                )

                plot.plot_response_calib(
                clusters[tri],
                gen_masked[tri],
                args,
                ax,
                f"fit_bounds_0_20",
                my_cmap(1),
                response_variable,
                bin_variable,
                "calibPU200",
                "fit_bounds_0_20",
                bin_,
                range_
                )

                if args.response:
                    naming= "Response"
                elif args.resolution:
                    naming= "Resolution"
                plt.savefig(
                f"{output_dir}/calibration_final_comparison/{naming}_pT_{bin_variable}_differential_{tri}_calib_comparison_{args.tag}_{args.pileup}.png",
                dpi=300
                )
                plt.savefig(
                f"{output_dir}/calibration_final_comparison/{naming}_pT_{bin_variable}_differential_{tri}_calib_comparison_{args.tag}_{args.pileup}.pdf"
                )

                print(f"{output_dir}/calibration_final_comparison/{naming}_pT_{bin_variable}_differential_{tri}_calib_comparison_{args.tag}_{args.pileup}.png")

                plt.close()

    if args.scale:

        legend_handles = []
        range_pt= [0,2]
        bin_pt= 30
        for tri in triangles:

            if tri == "0p0113" or tri == "0p016":
                range_pt= [0,2]
            else:
                range_pt= [0,5]
        
            fig, ax = plt.subplots(figsize=(10, 10))
            plot.scale_distribution_calib(
                clusters[tri],
                gen_masked[tri],
                args,
                field= "pT",
                bin_n= bin_pt,
                range_= range_pt,
                label= f"pT for {tri}",
                color=my_cmap(0),
                legend_handles= legend_handles,
                ax=ax
            )

            plot.scale_distribution_calib(
                clusters[tri],
                gen_masked[tri],
                args,
                field= f"Ecalib_PU_term_a_b_no_bounds",
                bin_n= bin_pt,
                range_= range_pt,
                label= f"PU0 bounds + a_b bounds" +"\n"+ f"for {tri}",
                color=my_cmap(3),
                legend_handles= legend_handles,
                ax=ax
            )

            plot.scale_distribution_calib(
                clusters[tri],
                gen_masked[tri],
                args,
                field= f"Ecalib_all_fit_bounds_0_20",
                bin_n= bin_pt,
                range_= range_pt,
                label= f"PU200 bounds " +"\n"+ f" for {tri}",
                color=my_cmap(1),
                legend_handles= legend_handles,
                ax=ax
            )

            naming= "Scale"
            plt.savefig(
            f"{output_dir}/calibration_final_comparison/{naming}_pT_{tri}_calib_comparison_{args.tag}_{args.pileup}.png",
            dpi=300
            )
            plt.savefig(
            f"{output_dir}/calibration_final_comparison/{naming}_pT_{tri}_calib_comparison_{args.tag}_{args.pileup}.pdf"
            )

            print(f"{output_dir}/calibration_final_comparison/{naming}_pT_{tri}_calib_comparison_{args.tag}_{args.pileup}.png")
            legend_handles = []
            plt.close()


import os
import sys
import argparse
import cppyy
import data_handle.plot_performances as plot
from   data_handling.event_performances import provide_events_performaces, apply_matching
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
    parser.add_argument('--scale',     action='store_true', help='Plot the response distribution of the emulator')
    parser.add_argument('--eta',     action='store_true', help='Plot the response distribution of the emulator')
    parser.add_argument('--PU0calib',     action='store_true', help='Plot the response distribution of the emulator')
    parser.add_argument('--PU200calib',     action='store_true', help='Plot the response distribution of the emulator')
    parser.add_argument('--PileUpcalib',     action='store_true', help='Plot the response distribution of the emulator')
    args = parser.parse_args()

    if args.pt_cut:
        output_dir += f"results_performance_plots_{args.particles}_{args.pileup}_cut_{args.pt_cut}_GeV"
        parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/cluster_pt_cut{args.pt_cut}_GeV/'
        os.makedirs(output_dir, exist_ok=True)
    elif args.gen_pt_cut:
        output_dir += f"results_performance_plots_{args.particles}_{args.pileup}_gen_cut_{args.gen_pt_cut}_GeV"
        os.makedirs(output_dir, exist_ok=True)
        parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/'
    elif args.gen_pt_cut and args.pt_cut:
        output_dir += f"results_performance_plots_{args.particles}_{args.pileup}_gen_cut_{args.gen_pt_cut}_GeV_cut_{args.pt_cut}_GeV"
        os.makedirs(output_dir, exist_ok=True)
        parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/cluster_pt_cut{args.pt_cut}_GeV_gen_pt_cut{args.gen_pt_cut}_GeV/'
    else:
        parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/'
        output_dir += f"results_performance_plots_{args.particles}_{args.pileup}"
        os.makedirs(output_dir, exist_ok=True)

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
    weights = {}



    if args.PU0calib:
        parquet_dir_wl= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU0_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV'

        calibration_configs = {
        # "fit_bounds_0_2": "PU0_bounds_0_2",
        "fit_bounds_0_20": "PU0_bounds_0_20",
        # "fit_bounds_0_2_no_layer1": "PU0_bounds_0_2_no_layer1",
        "fit_bounds_0_20_no_layer1": "PU0_bounds_0_20_no_layer1",
        "no_bounds": "PU0_no_bounds",
        "no_bounds_nolayer1": "PU0_no_bounds_no_layer1",
        }
        for calib_name, folder in calibration_configs.items():
            weights[calib_name] = {}
            for tri in triangles:
                path = f"{parquet_dir_wl}/{folder}/weight_wl_{tri}.parquet"
                weights[calib_name][tri] = ak.from_parquet(path)
            
        quantity_calib="calibPU0"
        

        for tri in triangles:
            for calib_name in calibration_configs.keys():
                clusters[tri] = plot.apply_calibration(
                clusters[tri],
                weights[calib_name][tri],
                calib_name
                )
                print(f"weights_{calib_name}_{tri}", weights[calib_name][tri])

        
        for tri in triangles:
            print(tri)
            fit_bounds_0_20 = getattr(clusters[tri], f"Ecalib_fit_bounds_0_20")
            fit_bounds_0_20_no_layer1 = getattr(clusters[tri], f"Ecalib_fit_bounds_0_20_no_layer1")
            no_bounds = getattr(clusters[tri], f"Ecalib_no_bounds")
            no_bounds_no_layer1 = getattr(clusters[tri], f"Ecalib_no_bounds_nolayer1")

            print("between fit and no layer 1 fit ", np.max(np.abs(fit_bounds_0_20-fit_bounds_0_20_no_layer1)))
            print("between fit and no bound ", np.max(np.abs(fit_bounds_0_20-no_bounds)))
            print("between fit and no bound no l1 ", np.max(np.abs(fit_bounds_0_20-no_bounds_no_layer1)))

            print("between fit no layer and no bound ", np.max(np.abs(fit_bounds_0_20_no_layer1-no_bounds)))
            print("between fit no layer and no bound no layer ", np.max(np.abs(fit_bounds_0_20_no_layer1-no_bounds_no_layer1)))

            print("between no bound and no layer 1 ", np.max(np.abs(no_bounds_no_layer1- no_bounds)))
        # print(getattr(clusters["0p0113"], f"Ecalib_PU_term_a_b_fit_bounds_0_20"))
        # print(getattr(clusters["0p0113"], f"Ecalib_PU_term_a_b_fit_bounds_0_20_no_layer1"))

        plt.figure(figsize=(10,10))
        for tri_idx, tri in enumerate(triangles):

            for calib_idx, calib_name in enumerate(calibration_configs.keys()):
                plot.plot_weight_val(weights, tri , my_cmap(calib_idx+1), calib_name,calib_idx, args)
                
            plt.savefig(
                f"Weightval_{tri}_calib_comparison_{args.tag}_{args.pileup}.png",
                dpi=300
                )
            plt.savefig(f"Weightval_{tri}_calib_comparison_{args.tag}_{args.pileup}.pdf")
            plt.close()
                
    if args.PileUpcalib:
        parquet_dir_wl= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU0_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV'

        calibration_configs_PU0 = {
        "fit_bounds_0_20": "PU0_bounds_0_20",
        "fit_bounds_0_20_no_layer1": "PU0_bounds_0_20_no_layer1",
        "no_bounds": "PU0_no_bounds",
        "no_bounds_nolayer1": "PU0_no_bounds_no_layer1",
        }

        for calib_name, folder in calibration_configs_PU0.items():
            weights[calib_name] = {}
            for tri in triangles:
                path = f"{parquet_dir_wl}/{folder}/weight_wl_{tri}.parquet"
                weights[calib_name][tri] = ak.from_parquet(path)

        for tri in triangles:
            for calib_name in calibration_configs_PU0.keys():
                clusters[tri] = plot.apply_calibration(
                clusters[tri],
                weights[calib_name][tri],
                calib_name
            )

        calibration_configs= {
        "a_b_fit_bounds_0_20": "a_b_bounds_0_20",
        "a_b_fit_bounds_0_20_no_layer1": "a_b_bounds_0_20_no_layer_1_PU0",
        "a_b_no_bounds": "a_b_no_bounds",
        "a_b_no_bounds_nolayer1": "a_b_no_bounds_no_layer1",
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
        "fit_bounds_0_20": "a_b_fit_bounds_0_20",
        "fit_bounds_0_20_no_layer1": "a_b_fit_bounds_0_20_no_layer1",
        "no_bounds": "a_b_no_bounds",
        "no_bounds_nolayer1": "a_b_no_bounds_nolayer1",
        }

        for tri in triangles:
            for calib_name, weight_name in calibration_configs_applying.items():
                clusters[tri] = plot.apply_calibration_eta(
                clusters[tri],
                weights_a[weight_name][tri],
                calib_name,
                weight_name
            )
                
    
            # print(tri)
            # fit_bounds_0_20 = getattr(clusters[tri], f"Ecalib_PU_term_a_b_fit_bounds_0_20")
            # fit_bounds_0_20_no_layer1 = getattr(clusters[tri], f"Ecalib_PU_term_a_b_fit_bounds_0_20_no_layer1")
            # no_bounds = getattr(clusters[tri], f"Ecalib_PU_term_a_b_no_bounds")
            # no_bounds_no_layer1 = getattr(clusters[tri], f"Ecalib_PU_term_a_b_no_bounds_nolayer1")

            # print("between fit and no layer 1 fit ", np.max(np.abs((fit_bounds_0_20 - fit_bounds_0_20_no_layer1) / fit_bounds_0_20)))
            # print("between fit and no bound ", np.max(np.abs(fit_bounds_0_20-no_bounds)/ fit_bounds_0_20))
            # print("between fit and no bound no l1 ", np.max(np.abs(fit_bounds_0_20-no_bounds_no_layer1)/ fit_bounds_0_20))

            # print("between fit no layer and no bound ", np.max(np.abs(fit_bounds_0_20_no_layer1-no_bounds)/fit_bounds_0_20_no_layer1))
            # print("between fit no layer and no bound no layer ", np.max(np.abs(fit_bounds_0_20_no_layer1-no_bounds_no_layer1)/fit_bounds_0_20_no_layer1))

            # print("between no bound and no layer 1 ", np.max(np.abs(no_bounds_no_layer1- no_bounds)/no_bounds_no_layer1))

        plt.figure(figsize=(10,10))
        for tri_idx, tri in enumerate(triangles):

            for calib_idx, calib_name in enumerate(calibration_configs.keys()):
                plot.plot_weight_val(weights_a, tri , my_cmap(calib_idx+1), calib_name,calib_idx, args)
                
            plt.savefig(
                f"Weightval_{tri}_calib_comparison_{args.tag}_{args.pileup}.png",
                dpi=300
                )
            plt.savefig(f"Weightval_{tri}_calib_comparison_{args.tag}_{args.pileup}.pdf")

            plt.close()
            
    
    if args.PU200calib:
        parquet_dir_wl= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU200_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV'

        calibration_configs= {
        "fit_bounds_0_20": "PU200_all_bounds_0_20",
        "fit_bounds_0_20_no_layer1": "PU200_all_bounds_0_20_no_layer1",
        "no_bounds": "PU200_all_no_bounds",
        "no_bounds_nolayer1": "PU200_all_no_bounds_no_layer1",
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

       

        plt.figure(figsize=(10,10))
        for tri_idx, tri in enumerate(triangles):

            for calib_idx, calib_name in enumerate(calibration_configs.keys()):
                plot.plot_weight_val(weights, tri , my_cmap(calib_idx+1), calib_name,calib_idx, args)
                
            plt.savefig(
                f"Weightval_{tri}_calib_comparison_{args.tag}_{args.pileup}.png",
                dpi=300
                )
            plt.savefig(f"Weightval_{tri}_calib_comparison_{args.tag}_{args.pileup}.pdf")
            plt.close()

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
                for i, calib_name in enumerate(calibration_configs.keys()):
                    # print("i",calib_name) 
                    plot.plot_response_calib(
                    clusters[tri],
                    gen_masked[tri],
                    args,
                    ax,
                    f"{calib_name}",
                    my_cmap(i+1),
                    response_variable,
                    bin_variable,
                    quantity_calib,
                    calib_name,
                    bin_,
                    range_
                    )

                if args.response:
                    naming= "Response"
                elif args.resolution:
                    naming= "Resolution"
                plt.savefig(
                f"{output_dir}/calibration_final/{naming}_pT_{bin_variable}_differential_{tri}_calib_comparison_{args.tag}_{args.pileup}.png",
                dpi=300
                )
                plt.savefig(
                f"{output_dir}/calibration_final/{naming}_pT_{bin_variable}_differential_{tri}_calib_comparison_{args.tag}_{args.pileup}.pdf"
                )

                print(f"{output_dir}/calibration_final/{naming}_pT_{bin_variable}_differential_{tri}_calib_comparison_{args.tag}_{args.pileup}.png")

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
            for i, calib_name in enumerate(calibration_configs.keys()):
                # print("i",calib_name)
                # # print("weights", weights[calib_name][tri])
                # print("weights", weights[calib_name][tri])
                if args.PileUpcalib:
                    field = f"Ecalib_PU_term_{calib_name}"
                elif args.PU200calib:
                    field = f"Ecalib_all_{calib_name}"
                elif args.PU0calib:
                    field = f"Ecalib_{calib_name}"
                else:
                    raise ValueError("No valid calibration type selected")

                plot.scale_distribution_calib(
                clusters[tri],
                gen_masked[tri],
                args,
                field= field,
                bin_n= bin_pt,
                range_= range_pt,
                label= f"{calib_name}",
                color=my_cmap(i+1),
                legend_handles= legend_handles,
                ax=ax
                )
            naming= "Scale"
            plt.savefig(
            f"{output_dir}/calibration_final/{naming}_pT_{tri}_calib_comparison_{args.tag}_{args.pileup}.png",
            dpi=300
            )
            plt.savefig(
            f"{output_dir}/calibration_final/{naming}_pT_{tri}_calib_comparison_{args.tag}_{args.pileup}.pdf"
            )

            print(f"{output_dir}/calibration_final/{naming}_pT_{tri}_calib_comparison_{args.tag}_{args.pileup}.png")
            legend_handles = []
            plt.close()


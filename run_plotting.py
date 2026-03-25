import os
import argparse
import awkward as ak
import data_handle.plot_performances as plot
from configs.config import PARQUET_BASE, EMU_CONFIG, PLOT_VARS
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
    #Object to plot
    parser.add_argument('--matched',     action='store_true', help='Plot matched events')
    parser.add_argument('--events',     action='store_true', help='Plot matched events')
    parser.add_argument('--filtered_events',     action='store_true', help='Plot matched events')

    # Plotting arguments
    parser.add_argument('--distribution',     action='store_true', help='Plot the distributions')
    parser.add_argument('--scale_distribution',     action='store_true', help='Plot the response distribution')

    args = parser.parse_args()

    parquet_dir= build_parquet_dir(args)
    parquet_dir_bare_events= f"{PARQUET_BASE}{args.particles}_{args.pileup}_new_branch/"

    #Load events 
    if args.matched:
        matched_events= f.load_matching_results(parquet_dir)
        print(matched_events["0p0113"])
    if args.events:
        events, events_gen= f.load_events(parquet_dir_bare_events)
        print(events_gen)
    if args.filtered_events:
        filtered_events= f.load_filtered_events(parquet_dir)
        print(filtered_events["0p0113"])

                
    # Initialize once
    plotter = DistributionPlotter(args, output_dir=f"plots_test/{args.pileup}")

    if args.distribution:
        vars_to_plot = ["pt", "eta", "phi"]

        for var in vars_to_plot:
            conf = PLOT_VARS[var]
            if args.matched:
                # --- Matched Clusters (Comparing Triangles Sizes) ---
                plot_data_matched_clusters = []
                for tri_key in EMU_CONFIG.keys():
                    plot_data_matched_clusters.append({
                        'data': matched_events[tri_key]["pair_cluster"],
                        'branch': conf['branch'],
                        'label': f"{tri_key}"
                    })
                plotter.plot(plot_data_matched_clusters, var , "Matched_cluster_distribution")

            if args.events:
                # --- Initial clusters (Comparing Triangles Sizes) ---
                plot_data_clusters = []
                for tri_key, tri_val in EMU_CONFIG.items():
                    # Dynamically find the bare branch name (e.g., cl3d_p0113pt)
                    # print(len(events[tri_key].pt))
                    plot_data_clusters.append({
                        'data': events[tri_key],
                        'branch': conf['branch'],
                        'label': f"Tri {tri_key} (Bare)"
                    })
                plotter.plot(plot_data_clusters, var, "Cluster_distribution")

            if args.filtered_events:
                # --- Filtered Clusters (Comparing Triangles Sizes) ---
                plot_data_filtered_clusters = []
                for tri_key in EMU_CONFIG.keys():
                    plot_data_matched_clusters.append({
                        'data': filtered_events[tri_key]["events_cluster_filtered"],
                        'branch': conf['branch'],
                        'label': f"Tri {tri_key} (Filtered)"
                    })
                plotter.plot(plot_data_matched_clusters,var, f"Clusters_gen_pt_cut_{args.gen_pt_cut}_distribution")

    if args.scale_distribution:
        vars_to_plot = ["pt_response", "eta_response", "phi_response"]

        for var in vars_to_plot:
            if args.matched:
                # --- Matched Clusters (Comparing Triangles Sizes) ---
                plot_data_matched_clusters = []
                for tri_key in EMU_CONFIG.keys():
                    plot_data_matched_clusters.append({
                        'data': matched_events[tri_key]["pair_cluster"],
                        'gen': matched_events[tri_key]["pair_gen"],
                        'branch': conf['branch'],
                        'branch_gen': conf['branch_gen'],
                        'label': f"{tri_key}"
                    })
                plotter.plot(plot_data_matched_clusters, var , "Matched_cluster_distribution")

            if args.events:
                # --- Initial clusters (Comparing Triangles Sizes) ---
                plot_data_clusters = []
                for tri_key, tri_val in EMU_CONFIG.items():
                    # Dynamically find the bare branch name (e.g., cl3d_p0113pt)
                    plot_data_clusters.append({
                        'data': events[tri_key],
                        'gen': events_gen,
                        'branch': conf['branch'],
                        'branch_gen': conf['branch_gen'],
                        'label': f"Tri {tri_key} (Bare)"
                    })
                plotter.plot(plot_data_clusters, var, "Cluster_distribution")

            if args.filtered_events:
                # --- Filtered Clusters (Comparing Triangles Sizes) ---
                plot_data_filtered_clusters = []
                for tri_key in EMU_CONFIG.keys():
                    plot_data_matched_clusters.append({
                        'data': filtered_events[tri_key]["events_cluster_filtered"],
                        'gen': filtered_events[tri_key]["events_gen_filtered"],
                        'branch': conf['branch'],
                        'branch_gen': conf['branch_gen'],
                        'label': f"Tri {tri_key} (Filtered)"
                    })
                plotter.plot(plot_data_matched_clusters,var, f"Clusters_gen_pt_cut_{args.gen_pt_cut}_distribution")


   
        
        




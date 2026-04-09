import os
import argparse
import awkward as ak
from configs.config import PARQUET_BASE, EMU_CONFIG, PLOT_VARS
from   data_handling.utils import build_parquet_dir, build_plotting_dir
import data_handling.files as f
import data_handling.calibration_functions as calib
from plotting.run_plots import PerformancePlotter, get_triangle_comparison
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
    parser.add_argument('--resolution_plots',     action='store_true', help='Plot the response distribution')
    parser.add_argument('--two_d_dist',     action='store_true', help='Plot 2D distribution')
    parser.add_argument('--efficiency',     action='store_true', help='Plot the efficiency')
    parser.add_argument('--binned_distributions',     action='store_true', help='Plot binned distributions')
    parser.add_argument('--n_clusters_plots',     action='store_true', help='Plot binned distributions')

    args = parser.parse_args()

    parquet_dir= build_parquet_dir(args)
    parquet_dir_bare_events= f"{PARQUET_BASE}{args.particles}_{args.pileup}_new_branch/"

    output_dir= build_plotting_dir(args)

    #Load events 
    if args.matched:
        matched_events= f.load_matching_results(parquet_dir)
        TASKS = {
            "dist": ["pt", "eta", "abs_eta", "phi", "delta_r"],
            "resp": ["pt_response", "eta_response", "phi_response"],
            "corr": [("pt_gen", "pt"), ("eta", "phi"),("pt", "eta"), ("delta_r", "eta") ],
            "dif_resp": [("pt_gen", "pt"), ("eta_gen", "pt"),("phi_gen", "pt"),  ("eta_gen", "eta")],
            "eff":  ["pt_eff", "eta_eff", "abs_eta_eff"],
            "binned_resp": [("pt_gen", "pt_response"), ("eta_gen", "pt_response"), ("phi_gen", "pt_response"), ("eta_gen", "eta_response"), ("phi_gen", "phi_response")]
            }
        
    if args.events:
        events, events_gen= f.load_events(parquet_dir_bare_events)
        TASKS = {
            "dist": ["pt", "eta","abs_eta", "phi"],
            "corr": [("eta", "phi"),("pt", "eta")],
            }
        if args.gen_pt_cut > 0:
            event_mask = ak.any(events_gen.pt > args.gen_pt_cut, axis=-1)
            events_gen = events_gen[event_mask]
            for key in events.keys():
                events[key] = events[key][event_mask]
    if args.filtered_events:
        filtered_events= f.load_filtered_events(parquet_dir)
        TASKS = {
            "dist": ["pt", "eta","abs_eta", "phi"],
            "corr": [("eta", "phi"),("pt", "eta")],
            }

    # Initialize Plotter
    plotter = PerformancePlotter(args, output_dir=output_dir)

    #set title based on pt cuts

    if args.gen_pt_cut !=0 and args.pt_cut ==0:
        title= r"$p_T^{gen} >$" + f"{args.gen_pt_cut} GeV"
    elif args.pt_cut !=0 and args.gen_pt_cut ==0:
        title= r"$p_T^{cluster} >$" + f"{args.pt_cut} GeV"
    elif args.pt_cut !=0 and args.gen_pt_cut !=0:
        title= r"$p_T^{cluster} >$" + f"{args.pt_cut} GeV and " + r"$p_T^{gen} >$"+ f"{args.gen_pt_cut} GeV"
    else:
        title=""


    # --- Distributions ---
    if args.distribution:
        if args.matched:
            data = get_triangle_comparison(matched_events)
            plotter.plot_1d(data, "delta_r", f"Matched_Dist_delta_r", title)

        for var in TASKS["dist"]:
            if args.matched:
                data = get_triangle_comparison(matched_events)
                plotter.plot_1d(data, var, f"Matched_Dist_{var}_gen_pt_cut_{args.gen_pt_cut}", title)

            if args.events:
                data = get_triangle_comparison(events,events_gen)
                plotter.plot_1d(data, var, f"Event_distribution_{var}_gen_pt_cut_{args.gen_pt_cut}", title)

            if args.filtered_events:
                data = get_triangle_comparison(filtered_events)
                plotter.plot_1d(data, var, f"Filtered_Dist_{var}_with_gen_pt_cut_{args.gen_pt_cut}", title)

    # --- 2D Distributions ---
    if args.two_d_dist:
        if args.matched:
            data = get_triangle_comparison(matched_events)
        if args.events:
            data = get_triangle_comparison(events, events_gen)
        if args.filtered_events:
            data = get_triangle_comparison(filtered_events)
        plotter.plot_2d_batch(data, TASKS["corr"], title)


    # --- Response Distributions ---
    if args.scale_distribution:
        for var in TASKS["resp"]:
            if args.matched:
                data = get_triangle_comparison(matched_events)
                plotter.plot_1d(data, var, f"Response_Dist_{var}")

    if args.resolution_plots:
        data = get_triangle_comparison(matched_events)
        # Define the 'X' axes (The generator truth we bin by)
        # Note: Using 'abs_eta_gen' will trigger our new absolute value logic
        x_variables = ["pt_gen", "abs_eta_gen", "phi_gen"]
        x_variables = ["pt_gen", "abs_eta_gen", "phi_gen"]

        # Define the 'Y' axes (The performance metrics we want to see)
        # y_variables = ["pt_response", "eta_response", "phi_response"]
        y_variables = ["phi_response"]

        for x_var in x_variables:
            for y_var in y_variables:
                plotter.plot_profile(data, x_var, y_var, 
                                    filename=f"Profile_{y_var}_vs_{x_var}", 
                                    mode='mean', title=title)
                plotter.plot_profile(data, x_var, y_var, 
                                    filename=f"Profile_{y_var}_vs_{x_var}", 
                                    mode='resolution', title=title)
                plotter.plot_profile(data, x_var, y_var, 
                                    filename=f"Profile_{y_var}_vs_{x_var}", 
                                    mode='rms', title=title)
                                    

        
    if args.efficiency:
        eff_data = get_triangle_comparison(matched_events, total_gen=events_gen)
        plotter.plot_efficiency(eff_data, PLOT_VARS["pt_eff"])
        plotter.plot_efficiency(eff_data, PLOT_VARS["abs_eta_eff"])
        plotter.plot_efficiency(eff_data, PLOT_VARS["phi_eff"])

    if args.binned_distributions:
        data = get_triangle_comparison(matched_events)

        for var in TASKS["binned_resp"]:
            plotter.plot_distributions_per_bin(
                datasets=data, 
                var_key=var[1], 
                binning_var_key=var[0], 
                filename="Dist_Response",
                title=title
                )
            

    if args.n_clusters_plots:
        # 1. Prepare the data bundle as you were doing
        if args.matched:
            data = get_triangle_comparison(matched_events)
        elif args.events:
            data = get_triangle_comparison(events, events_gen)

        # --- Global Distribution (Total clusters in event) ---
        # This shows the detector activity for all events, including those with 1 or 2 gen particles in theinitial state
        plotter.plot_1d(data, "n_clusters", filename="Dist_NClusters_Global", 
                        title=title, gen_n=None)

        # --- Single Particle Distribution ---
        # This shows the "splitting" behavior for clean 1-gen events
        plotter.plot_1d(data, "n_clusters", filename="Dist_NClusters_Gen1", title=title, gen_n=1)

        # --- Distribution per pT Bin (for Gen=1) ---
        # This creates a plot for each pT bin to see if splitting changes with energy
        plotter.plot_nclusters_per_bin(data, "pt_gen", gen_n=1, title=title)

        # --- SCENARIO 4: The Profile Plot (Mean N_clusters vs Pt) ---
        # This is the "summary" plot showing the average splitting vs energy
        plotter.plot_profile(data, "pt_gen", "n_clusters", "N_clusters_pt_gen", mode='mean', gen_n=1, title=title)
        plotter.plot_profile(data, "pt_gen", "n_clusters", "N_clusters_pt_gen", mode='resolution', gen_n=1, title=title)

        # --- SCENARIO 5: Profile for Eta ---
        # Useful to see if detector geometry (Endcaps vs Barrel) affects splitting
        plotter.plot_profile(data, "abs_eta_gen", "n_clusters", "N_clusters_abs_eta_gen", mode='mean', gen_n=1, title=title)
        plotter.plot_profile(data, "abs_eta_gen", "n_clusters", "N_clusters_abs_eta_gen", mode='resolution', gen_n=1, title=title)
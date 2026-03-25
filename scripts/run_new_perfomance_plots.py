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
            "eff":  ["pt_eff", "eta_eff", "abs_eta_eff"]
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

    if args.gen_pt_cut !=0 and args.pt_cut ==0:
        title= r"p_T^{gen} >" + f"{args.gen_pt_cut}"
    elif args.pt_cut !=0 and args.gen_pt_cut ==0:
        title= r"p_T^{cluster} >" + f"{args.pt_cut}"
    elif args.pt_cut !=0 and args.gen_pt_cut !=0:
        title= r"p_T^{cluster} >" + f"{args.pt_cut} and " + r"p_T^{gen} >"+ f"{args.gen_pt_cut}"
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

        # Define the 'Y' axes (The performance metrics we want to see)
        y_variables = ["pt_response", "eta_response", "phi_response"]

        for x_var in x_variables:
            for y_var in y_variables:
                plotter.plot_profile(data, x_var, y_var, 
                                    filename=f"Profile_{y_var}_vs_{x_var}", 
                                    mode='mean')
                plotter.plot_profile(data, x_var, y_var, 
                                    filename=f"Profile_{y_var}_vs_{x_var}", 
                                    mode='resolution')
                
                plotter.plot_profile(data, x_var, y_var, 
                                    filename=f"Profile_{y_var}_vs_{x_var}", 
                                    mode='rms')
                                    

        
    if args.efficiency:
        eff_data = get_triangle_comparison(matched_events, total_gen=events_gen)
        plotter.plot_efficiency(eff_data, PLOT_VARS["pt_eff"])
        plotter.plot_efficiency(eff_data, PLOT_VARS["abs_eta_eff"])
        plotter.plot_efficiency(eff_data, PLOT_VARS["abs_phi_eff"])

    if args.binned_distributions:
        data = get_triangle_comparison(matched_events)

        plotter.plot_distributions_per_bin(
            datasets=data, 
            var_key="pt_response", 
            binning_var_key="pt_gen", 
            filename="Dist_Response"
        )
    
    if args.n_clusters_plots:
        # 1. Calculate how many Gen particles were in each event
        # This is done once on the global events_gen array
        gen_multiplicity = ak.num(events_gen.pt, axis=-1)
        print(events_gen.pt)
        print(ak.num(events_gen.pt, axis=-1))

        # 2. Create Masks
        mask_1gen = (gen_multiplicity == 1)
        mask_2gen = (gen_multiplicity == 2)

        # 3. Build Bundles for each scenario
        bundle_1gen = []
        bundle_2gen = []

        for tri_key in EMU_CONFIG.keys():
            # Scenario: 1 Gen particle
            bundle_1gen.append({
                'label': f"Tri {tri_key}",
                'data': events[tri_key][mask_1gen],
                'gen': events_gen[mask_1gen][:, 0] # The only gen particle
            })
            
            # Scenario: 2 Gen particles
            bundle_2gen.append({
                'label': f"Tri {tri_key}",
                'data': events[tri_key][mask_2gen],
                # For 2-gen events, we usually use the leading Pt for the X-axis
                'gen': events_gen[mask_2gen][:, 0] 
            })

            # 4. Plot them separately so the comparison is "Fair"
            # --- Single Gen Plots ---
            plotter.plot_multi_distribution(bundle_1gen, "n_clusters", "1Gen_Comparison", 
                                        title="Events with 1 Gen Particle")
            plotter.plot_profile(bundle_1gen, "pt_gen", "n_clusters", 
                                filename="NClusters_vs_Pt_1Gen", mode='mean',
                                title="Mean Clusters (1 Gen Particle)")

            # --- Double Gen Plots ---
            plotter.plot_multi_distribution(bundle_2gen, "n_clusters", "2Gen_Comparison",
                                        title="Events with 2 Gen Particles")
            plotter.plot_profile(bundle_2gen, "pt_gen", "n_clusters", 
                                filename="NClusters_vs_Pt_2Gen", mode='mean',
                                title="Mean Clusters (2 Gen Particles)")
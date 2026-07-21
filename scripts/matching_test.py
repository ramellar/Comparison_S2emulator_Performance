import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from configs.config import DEFAULT_COLORS
from   data_handling.event_performances import apply_matching, perform_antikt
from data_handling.utils import build_parquet_dir
from configs.config import PARQUET_BASE, EMU_CONFIG
import data_handling.files as io
from data_handling.efficiency import compute_efficiencies, compute_efficiencies_from_results

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Stage-2 Emulator Parameters')

    parser.add_argument('--particles', type=str, default='photons', help='Choose the particle sample')
    parser.add_argument('--pileup',    type=str, default='PU0',     help='Choose the pileup - PU0 or PU200')
    parser.add_argument('--pt_cut',    type=float, default=0,         help='Provide the cut for the cluster pt')
    parser.add_argument('--gen_pt_cut',    type=float, default=0,         help='Provide the cut for the gen pt')
    parser.add_argument('--deltaR',    type=float, default=0.2,         help='DeltaR')
    parser.add_argument('--total_efficiency',     action='store_true', help='Compute the total efficiency for each emulation test')
    parser.add_argument("--matching_type", type=str, choices=["gen_cluster", "antikt_jets"], default="gen_cluster", help="Choose the matching strategy: 1-to-1 gen-cluster or matching with anti-kt reconstructed jets.")
    parser.add_argument("--only_efficiency", action="store_true", help="Compute efficiency from saved matching results without rerunning matching.")

    args = parser.parse_args()

    base = f"{PARQUET_BASE}{args.particles}_{args.pileup}_new_branch/"
    print(base)
    events, events_gen = io.load_events(base)
    output_dir= build_parquet_dir(args)
    
    mask_gen_pt = events_gen.pt > args.gen_pt_cut
    events_gen_den = events_gen[mask_gen_pt]
    events_gen_den = events_gen_den[ak.num(events_gen_den.pt) > 0]
    
    results = {}
    
    if args.only_efficiency:
        results = io.load_matching_results(output_dir)
        compute_efficiencies_from_results(results)
        exit()
    
    mask_gen_pt = events_gen.pt > args.gen_pt_cut
    events_gen_den = events_gen[mask_gen_pt]

    mask_gen = ak.num(events_gen_den.pt, axis=-1) > 0
    events_gen_den = events_gen_den[mask_gen]
    

    #if args.total_efficiency:
    #    compute_efficiencies(events, events_gen, args)

    #Apply antikt algorithm for clusters definition with deltaR < 0.4
    #jets = {}
    
    #for key in EMU_CONFIG.keys():
    #    n_clusters_before = ak.num(events[key][:100].pt)
    #    mean_before = ak.mean(n_clusters_before)
    #    print(f"\n===== {key} =====")
    #    #print("Clusters before anti-kt:", ak.to_list(n_clusters_before))
    #    print(f"Mean before anti-kt: {mean_before:.2f}")



    # Choose the matched argument between reco jets and single clusters
    
    matched_objects = {}
    if args.matching_type == "antikt_jets":

        for key, ds in EMU_CONFIG.items():

            print(f">>> RECONSTRUCTING ANTI-KT JETS: {key} <<<")

            matched_objects[key] = perform_antikt(events[key][:5000])

            n_objects_per_event = ak.num(matched_objects[key].pt)
            mean_objects = ak.mean(n_objects_per_event)

            print(f"Mean number of jets per event ({key}): {mean_objects:.2f}")

        plot_dir = "plots_PU200/antikt_diagnostics"
        os.makedirs(plot_dir, exist_ok=True)

        print(f"Creating diagnostic plots in: {plot_dir}")

        # ---------------------------------------------------------
        # Rho distributions
        # ---------------------------------------------------------
        plt.figure(figsize=(7, 5))

        for i, key in enumerate(EMU_CONFIG):

            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

            rho_values = ak.drop_none(matched_objects[key].rho)
            rho_values = ak.to_numpy(rho_values)
            rho_values = rho_values[np.isfinite(rho_values)]

            plt.hist(
                rho_values,
                bins=25,
                histtype="step",
                linewidth=1.5,
                density=True,
                color=color,
                label=key
            )
            
            plt.hist(
                rho_values,
                bins=25,
                density=True,
                color=color,
                histtype="stepfilled",
                alpha=0.20
            )

        plt.xlabel(r"$\rho$ [GeV / unit area]")
        plt.ylabel("Normalized events")
        plt.title(r"Event $\rho$ distribution")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        rho_path = os.path.join(
            plot_dir,
            "rho_all_triangle_sizes.png"
        )

        plt.savefig(rho_path, dpi=300)
        plt.close()

        print(f"Saved: {rho_path}")

        # ---------------------------------------------------------
        # Jet-area distributions
        # ---------------------------------------------------------
        plt.figure(figsize=(7, 5))

        for i, key in enumerate(EMU_CONFIG):

            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

            area_values = ak.flatten(
                matched_objects[key].area,
                axis=None
            )

            area_values = ak.drop_none(area_values)
            area_values = ak.to_numpy(area_values)

            area_values = area_values[
                np.isfinite(area_values) & (area_values > 0)
            ]

            plt.hist(
                area_values,
                bins=25,
                histtype="step",
                linewidth=1.5,
                density=True,
                color=color,
                label=key
            )
            
            plt.hist(
                area_values,
                bins=25,
                density=True,
                color=color,
                histtype="stepfilled",
                alpha=0.20
            )

        plt.xlabel("Jet area")
        plt.ylabel("Normalized jets")
        plt.title(r"Anti-$k_{\mathrm{T}}$ jet-area distribution")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        area_path = os.path.join(
            plot_dir,
            "area_all_triangle_sizes.png"
        )

        plt.savefig(area_path, dpi=300)
        plt.close()

        print(f"Saved: {area_path}")

    elif args.matching_type == "gen_cluster":

        for key, branch in EMU_CONFIG.items():

            matched_objects[key] = events[key]

            n_objects_per_event = ak.num(matched_objects[key].pt)
            mean_objects = ak.mean(n_objects_per_event)

            #print(f"\n===== {key} =====")
            #print(f"Mean number of clusters: {mean_objects:.2f}")
            #print(f"Size {key}: using original clusters")
    # Print reconstructed jets properties for the first n events    
    #for key in all_jets.keys():
    #    print(f"\n===== {key} =====")
    #    for i in range(1):
    #        n_jets = len(all_jets[key].pt[i])
    #        print(f"Event {i}: {n_jets} jets")
    #        for j in range(n_jets):
    #            print(
    #                f"pt={all_jets[key].pt[i][j]:.2f}, "
    #                f"eta={all_jets[key].eta[i][j]:.2f}, "
    #                f"phi={all_jets[key].phi[i][j]:.2f}"
    #            )
        
    

    #for jet in all_clusters["0p0113"][0]:
        #print("Printing pt and number of reconstructed clusters with antikt")
        #print(jet.pt(), len(jet.constituents()))

    
    #Apply matching of clusters and gen partciles requiring DeltaR < 0.2 and taking the highest pt cluster
    for key, branch in EMU_CONFIG.items():
        pair_cluster, pair_gen, events_filtered, events_gen_filtered = apply_matching(
            matched_objects[key],
            events_gen,
            args,
            deltaR=args.deltaR
        )
    
        results[key] = {
            "pair_cluster": pair_cluster,
            "pair_gen": pair_gen,
            "events_filtered": events_filtered,
            "events_gen_filtered": events_gen_filtered,
            "events_gen_denominator": events_gen_den,
        }
        

    matched_clusters = [results[key]["pair_cluster"] for key in EMU_CONFIG]
    matched_gen = [results[key]["pair_gen"] for key in EMU_CONFIG]
    events_cl_gen_cut = [results[key]["events_filtered"] for key in EMU_CONFIG]
    events_gen_gen_cut = [results[key]["events_gen_filtered"] for key in EMU_CONFIG]
    


    #for key in EMU_CONFIG.keys():
#
    #    print(f"\n==================== {key} ====================")
#
    #    pair_cluster = results[key]["pair_cluster"]
    #    pair_gen = results[key]["pair_gen"]
#
    #    n_events = min(99900, len(pair_cluster.pt))
#
    #    for iev in range(10):
#
    #        n_matches = len(pair_cluster.pt[iev])
#
    #        if n_matches == 0:
    #            continue
#
    #        print(f"\nEvent {iev} ({n_matches} matched objects)")
#
    #        for iobj in range(n_matches):
#
    #            print(
    #                f"GEN     : "
    #                f"pt={pair_gen.pt[iev][iobj]:7.2f}  "
    #                f"eta={pair_gen.eta[iev][iobj]:6.3f}  "
    #                f"phi={pair_gen.phi[iev][iobj]:6.3f}"
    #            )
#
    #            print(
    #                f"CLUSTER : "
    #                f"pt={pair_cluster.pt[iev][iobj]:7.2f}  "
    #                f"eta={pair_cluster.eta[iev][iobj]:6.3f}  "
    #                f"phi={pair_cluster.phi[iev][iobj]:6.3f}"
    #            )

    #            print("-" * 50)

    io.save_matching_results(results, output_dir)
    
    if args.total_efficiency:
        compute_efficiencies(matched_objects, events_gen, args)
        
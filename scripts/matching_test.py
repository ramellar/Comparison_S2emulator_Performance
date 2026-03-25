import os
import argparse
import awkward as ak
from   data_handling.event_performances import apply_matching
from data_handling.utils import build_parquet_dir
from configs.config import PARQUET_BASE, EMU_CONFIG
import data_handling.files as io
from data_handling.efficiency import compute_efficiencies

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Stage-2 Emulator Parameters')

    parser.add_argument('--particles', type=str, default='photons', help='Choose the particle sample')
    parser.add_argument('--pileup',    type=str, default='PU0',     help='Choose the pileup - PU0 or PU200')
    parser.add_argument('--pt_cut',    type=float, default=0,         help='Provide the cut for the cluster pt')
    parser.add_argument('--gen_pt_cut',    type=float, default=0,         help='Provide the cut for the gen pt')
    parser.add_argument('--deltaR',    type=float, default=0.1,         help='DeltaR')
    parser.add_argument('--total_efficiency',     action='store_true', help='Compute the total efficiency for each emulation test')

    args = parser.parse_args()

    base = f"{PARQUET_BASE}{args.particles}_{args.pileup}_new_branch/"
    print(base)
    events, events_gen = io.load_events(base)
    output_dir= build_parquet_dir(args)

    results = {}

    if args.total_efficiency:
        compute_efficiencies(events, events_gen, args)

    #Apply matching of clusters and gen partciles requiring DeltaR <0.1 and taking the highest pt cluster
    for key, branch in EMU_CONFIG.items():
        pair_cluster, pair_gen, events_filtered, events_gen_filtered = apply_matching(
            events[key],
            events_gen,
            args,
            deltaR=args.deltaR
        )

        results[key] = {
            "pair_cluster": pair_cluster,
            "pair_gen": pair_gen,
            "events_filtered": events_filtered,
            "events_gen_filtered": events_gen_filtered,
        }


    matched_clusters = [results[key]["pair_cluster"] for key in EMU_CONFIG]
    matched_gen = [results[key]["pair_gen"] for key in EMU_CONFIG]
    events_cl_gen_cut = [results[key]["events_filtered"] for key in EMU_CONFIG]
    events_gen_gen_cut = [results[key]["events_gen_filtered"] for key in EMU_CONFIG]

    io.save_matching_results(results, output_dir)


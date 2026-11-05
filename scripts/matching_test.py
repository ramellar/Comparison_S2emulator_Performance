import os
import argparse
import awkward as ak
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
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument("--n_events", type=int, default=None, help="Use only the first N events before splitting them across jobs")

    args = parser.parse_args()
    
    #Flag the matching argument in the file name
    matching_object = {"gen_cluster": "_cluster", "antikt_jets": "_jet"}[args.matching_type]
    
    output_dir= build_parquet_dir(args)
    
    # ==============================================================        
    # COMPUTE MATCHING EFFICIENCY FROM ALREADY EXISTING DATA
    # ==============================================================            
    if args.only_efficiency:
        results = io.load_matching_results(output_dir)
        compute_efficiencies_from_results(results)
        exit()
        
        

    base = f"{PARQUET_BASE}{args.particles}_{args.pileup}_new_branch/"
    print(base)
    
    events, events_gen = io.load_events(base)
    if args.n_events is not None:
        if args.n_events <= 0:
            parser.error("--n_events must be a positive integer")

        n_events = min(args.n_events, len(events_gen))
        events_gen = events_gen[:n_events]
        events = {key: event[:n_events] for key, event in events.items()}
        print(f"Limiting input to the first {n_events} events")
    
    
    
    # ==============================================================            
    #DIVIDING IN CHUNKS AND APLLYING CUTS
    # ==============================================================        
    start = len(events_gen) * args.job_id // args.n_jobs
    stop = len(events_gen) * (args.job_id + 1) // args.n_jobs

    events_gen = events_gen[start:stop]
    events = {
        key: event[start:stop]
        for key, event in events.items()
    }

    print(
        f"Job {args.job_id}/{args.n_jobs}: "
        f"Global events [{start}:{stop}), local {len(events_gen)}"
    )
           
    
    parts_dir = os.path.join(output_dir, "parts")
    if args.n_jobs > 1:
        os.makedirs(parts_dir, exist_ok=True)
    
    mask_gen_pt = events_gen.pt > args.gen_pt_cut
    events_gen_den = events_gen[mask_gen_pt]
    events_gen_den = events_gen_den[ak.num(events_gen_den.pt) > 0]
    
    results = {}



    # ==============================================================        
    # CHOOSING THE MATCHING ARGUMENT BETWEEN RECO JETS AND SINGLE CLUSTERS
    # ==============================================================        
    matched_objects = {}
    antikt_diagnostics = {}
    if args.matching_type == "antikt_jets":

        for key, ds in EMU_CONFIG.items():

            print(f">>> RECONSTRUCTING ANTI-KT JETS: {key} <<<")

            matched_objects[key] = perform_antikt(events[key])
            antikt_diagnostics[key] = matched_objects[key][["rho", "area"]]

            n_objects_per_event = ak.num(matched_objects[key].pt)
            mean_objects = ak.mean(n_objects_per_event)

            print(f"Mean number of jets per event ({key}): {mean_objects:.2f}")

    elif args.matching_type == "gen_cluster":

        for key, branch in EMU_CONFIG.items():

            matched_objects[key] = events[key]

            n_objects_per_event = ak.num(matched_objects[key].pt)
            mean_objects = ak.mean(n_objects_per_event)
            
            
            
    # ==============================================================        
    # APPLY MATCHING
    # ==============================================================        
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
        if key in antikt_diagnostics:
            results[key]["antikt_diagnostics"] = antikt_diagnostics[key]
        

    matched_clusters = [results[key]["pair_cluster"] for key in EMU_CONFIG]
    matched_gen = [results[key]["pair_gen"] for key in EMU_CONFIG]
    events_cl_gen_cut = [results[key]["events_filtered"] for key in EMU_CONFIG]
    events_gen_gen_cut = [results[key]["events_gen_filtered"] for key in EMU_CONFIG]

        
        
    # ==============================================================            
    # SAVE MATCHING RESULTS IN CHUNKS
    # ==============================================================        
    if args.n_jobs > 1:
        io.save_matching_results(
            results,
            parts_dir + "/",
            n_chunk=args.job_id,
            matching_object=matching_object,
        )
    else:
        io.save_matching_results(
            results,
            output_dir,
            matching_object=matching_object,
        )



    # ==============================================================        
    # COMPUTE EFFICIENCY DIRECTLY AFTER RECONSTRUCTION AND MATCHING
    # ==============================================================            
    if args.total_efficiency:
        compute_efficiencies(matched_objects, events_gen, args)
        

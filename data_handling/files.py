import awkward as ak
import os
from configs.config import EMU_CONFIG


def load_events(parquet_dir):

    events = {}

    for key in EMU_CONFIG:
        events[key] = ak.from_parquet(parquet_dir + f"events_{key}.parquet")     #cambia f"events_{key}_part_0.parquet" se vuoi vedere come sono stati salvati i files in single chunk

    events_gen = ak.from_parquet(parquet_dir + "events_gen.parquet")         #cambia anche qua "events_gen_part_0.parquet"

    return events, events_gen


def save_matching_results(results, parquet_dir, n_chunk=None, matching_object=""):
    
    chunk_tag = "" if n_chunk is None else f"_part_{n_chunk}"
    
    for key, res in results.items():
        ak.to_parquet(results[key]["pair_cluster"], parquet_dir + f"pair_cluster_{key}_matched{matching_object}{chunk_tag}.parquet")
        ak.to_parquet(results[key]["pair_gen"], parquet_dir + f"pair_gen_masked_{key}{matching_object}{chunk_tag}.parquet")
        ak.to_parquet(results[key]["events_filtered"], parquet_dir + f"events_{key}_filtered{matching_object}{chunk_tag}.parquet")
        ak.to_parquet(results[key]["events_gen_filtered"], parquet_dir + f"events_gen_filtered_{key}{matching_object}{chunk_tag}.parquet")
        ak.to_parquet(results[key]["events_gen_denominator"], parquet_dir + f"events_gen_denominator_{key}{matching_object}{chunk_tag}.parquet")
        
        #Add the information about rho and jet Area (only for antikt reconstruction)
        if "antikt_diagnostics" in res:
            ak.to_parquet(res["antikt_diagnostics"], parquet_dir + f"antikt_diagnostics_{key}{matching_object}{chunk_tag}.parquet")




def load_filtered_events(parquet_dir, matching_object=""):

    results = {}
    for key in EMU_CONFIG:
        results[key] = {
            "events_cluster_filtered": ak.from_parquet(parquet_dir + f"events_{key}_filtered{matching_object}.parquet"), 
            "events_gen_filtered": ak.from_parquet(parquet_dir + f"events_gen_filtered_{key}{matching_object}.parquet")
        }

    return results


def load_antikt_diagnostics(parquet_dir, matching_object=""):
    """Load event-level rho and per-jet area arrays for every triangle."""
    diagnostics = {}
    for key in EMU_CONFIG:
        path = os.path.join(parquet_dir, f"antikt_diagnostics_{key}{matching_object}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Anti-kT diagnostic data not found for triangle {key}: {path}")
        diagnostics[key] = ak.from_parquet(path)
    return diagnostics




def load_matching_results(parquet_dir, pt_type="raw", matching_object=""):

    results = {}

    for key in EMU_CONFIG:

        cluster_path = parquet_dir + f"pair_cluster_{key}_matched{matching_object}.parquet"
        pair_cluster = ak.from_parquet(cluster_path)
        pt_fields = {"raw": None, "offset": "pt_corrected", "mc": "pt_corr_tot",}
        pt_field = pt_fields[pt_type]

        if pt_field is not None:
            if pt_field not in ak.fields(pair_cluster):
                raise ValueError(f"Missing {pt_field} field in {cluster_path}. " "Run scripts.apply_calib with the required calibration first.")
            pair_cluster = ak.with_field(pair_cluster, pair_cluster[pt_field], "pt")

        results[key] = {
            "pair_cluster": pair_cluster,
            "pair_gen": ak.from_parquet(parquet_dir + f"pair_gen_masked_{key}{matching_object}.parquet"),
            "events_filtered": ak.from_parquet(parquet_dir + f"events_{key}_filtered{matching_object}.parquet"),
            "events_gen_filtered": ak.from_parquet(parquet_dir + f"events_gen_filtered_{key}{matching_object}.parquet"),
            "events_gen_denominator": ak.from_parquet(parquet_dir + f"events_gen_denominator_{key}{matching_object}.parquet")
        }
        taucalib_fit_path = os.path.join(parquet_dir, f"taucalib_fit_{key}.parquet")
        if os.path.exists(taucalib_fit_path):
            results[key]["taucalib_fit"] = ak.from_parquet(taucalib_fit_path)

    return results




def load_taucalib_mc_response(parquet_dir):
    response_maps = {}

    for key in EMU_CONFIG:
        path = os.path.join(parquet_dir, f"taucalib_mc_response_{key}.parquet")

        if not os.path.exists(path):
            raise FileNotFoundError(f"MC response data not found: {path}." "Run scripts.apply_calib with --calibration MC first.")
        
        response_maps[key] = ak.from_parquet(path)

    return response_maps

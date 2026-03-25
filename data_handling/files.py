import awkward as ak
from configs.config import EMU_CONFIG


def load_events(parquet_dir):

    events = {}

    for key in EMU_CONFIG:
        events[key] = ak.from_parquet(parquet_dir + f"events_{key}.parquet")

    events_gen = ak.from_parquet(parquet_dir + "events_gen.parquet")

    return events, events_gen


def save_matching_results(results, parquet_dir):

    for key, res in results.items():
        ak.to_parquet(
            results[key]["pair_cluster"],
            parquet_dir + f"pair_cluster_{key}_matched.parquet"
        )

        ak.to_parquet(
            results[key]["pair_gen"],
            parquet_dir + f"pair_gen_masked_{key}.parquet"
        )

        ak.to_parquet(
            results[key]["events_filtered"],
            parquet_dir + f"events_{key}_filtered.parquet"
        )

        ak.to_parquet(
            results[key]["events_gen_filtered"],
            parquet_dir + f"events_gen_filtered_{key}.parquet"
        )


def load_matching_results(parquet_dir):

    results = {}

    for key in EMU_CONFIG:

        results[key] = {
            "pair_cluster": ak.from_parquet(parquet_dir + f"pair_cluster_{key}_matched.parquet"),
            "pair_gen": ak.from_parquet(parquet_dir + f"pair_gen_masked_{key}.parquet"),
        }

    return results

def load_filtered_events(parquet_dir):

    results = {}

    for key in EMU_CONFIG:

        results[key] = {
            "events_cluster_filtered": ak.from_parquet(parquet_dir + f"events_{key}_filtered.parquet"),
            "events_gen_filtered": ak.from_parquet(parquet_dir + f"events_gen_filtered_{key}.parquet"),
        }

    return results

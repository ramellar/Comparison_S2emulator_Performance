import numpy as np

EMU_CONFIG = {
    "0p0113": "cl3d_p0113Tri",
    "0p016": "cl3d_p016Tri",
    "0p03": "cl3d_p03Tri",
    "0p045": "cl3d_p045Tri",
    "Ref": "cl3d_Ref",
}

CALIB_CONFIGS = {
    "no_bounds": {
        "bounds": (-np.inf, np.inf),
        "remove_layer1": False,
    },
    "bounds_0_20": {
        "bounds": (0, 20),
        "remove_layer1": False,
    },
    "no_bounds_no_layer1": {
        "bounds": (-np.inf, np.inf),
        "remove_layer1": True,
    },
    "bounds_0_20_no_layer1": {
        "bounds": (0, 20),
        "remove_layer1": True,
    },
}

# PU0_CONFIG_FOR_SEQ=None
PU0_CONFIG_FOR_SEQ="bounds_0_20"

COMPARISONS = {
    "PU0_bounds": {
        "strategy": "PU0",
        "wl": "bounds",
    },

    "PU200_seq_mixed": {
        "strategy": "PU200_seq",
        "wl": "bounds",
        "eta": "no_bounds",
    },

    "PU200_bounds": {
        "strategy": "PU200",
        "all": "bounds",
    },
}

STRATEGIES = ["PU0", "PU200", "PU200_seq"]

PARQUET_BASE = "/data_CMS/cms/amella/HGCAL_samples/parquet_files_ready/"

EVENT_NAMES = ["events_gen", "events_0p0113", "events_0p016", "events_0p03", "events_0p045", "events_Ref"]
CLUSTER_MATCHED_EVENTS_NAMES=["pair_cluster_0p0113_matched", "pair_cluster_0p016_matched", "pair_cluster_0p03_matched", "pair_cluster_0p045_matched", "pair_cluster_Ref_matched"]
GEN_MATCHED_EVENTS_NAMES=["pair_gen_masked_0p0113.parquet", "pair_gen_masked_0p016.parquet", "pair_gen_masked_0p03.parquet", "pair_gen_masked_0p045.parquet", "pair_gen_masked_Ref.parquet"]
FILTERED_CL_EVENTS_NAMES=["events_0p0113_filtered", "events_0p016_filtered", "events_0p03_filtered", "events_0p045_filtered", "events_Ref_filtered"]
FILTERED_GEN_EVENTS_NAMES = ["events_gen_fileterd_0p0113", "events_gen_fileterd_0p016", "events_gen_fileterd_0p03", "events_gen_fileterd_0p045", "events_gen_fileterd_Ref"]
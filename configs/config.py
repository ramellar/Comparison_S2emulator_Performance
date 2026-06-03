import numpy as np

EMU_CONFIG = {
    "0p0113": "cl3d_p0113Tri",
    "0p016": "cl3d_p016Tri",
    "0p03": "cl3d_p03Tri",
    "0p045": "cl3d_p045Tri",
    "Ref": "cl3d_Ref",
}

PARQUET_BASE = "/data_CMS/cms/amella/HGCAL_samples/parquet_files_final/"

EVENT_NAMES = ["events_gen", "events_0p0113", "events_0p016", "events_0p03", "events_0p045", "events_Ref"]
CLUSTER_MATCHED_EVENTS_NAMES=["pair_cluster_0p0113_matched", "pair_cluster_0p016_matched", "pair_cluster_0p03_matched", "pair_cluster_0p045_matched", "pair_cluster_Ref_matched"]
GEN_MATCHED_EVENTS_NAMES=["pair_gen_masked_0p0113.parquet", "pair_gen_masked_0p016.parquet", "pair_gen_masked_0p03.parquet", "pair_gen_masked_0p045.parquet", "pair_gen_masked_Ref.parquet"]
FILTERED_CL_EVENTS_NAMES=["events_0p0113_filtered", "events_0p016_filtered", "events_0p03_filtered", "events_0p045_filtered", "events_Ref_filtered"]
FILTERED_GEN_EVENTS_NAMES = ["events_gen_fileterd_0p0113", "events_gen_fileterd_0p016", "events_gen_fileterd_0p03", "events_gen_fileterd_0p045", "events_gen_fileterd_Ref"]

#--------------------------
# CALIBRATION
#--------------------------

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

# Choose the PU0 config to be used to derive the eta correction factors
# If "None" then the same method type will be applied to PU0 and PU200
# If fixed string, then that method will be applied for the layer weights

# PU0_CONFIG_FOR_SEQ=None
PU0_CONFIG_FOR_SEQ="bounds_0_20"

# COMPARISONS = {
#     "PU0_bounds": {
#         "strategy": "PU0",
#         "wl": "bounds",
#     },

#     "PU200_seq_mixed": {
#         "strategy": "PU200_seq_b_nb",
#         "wl": "bounds",
#         "eta": "no_bounds",
#     },

#     "PU200_bounds": {
#         "strategy": "PU200",
#         "all": "bounds",
#     },
# }
# COMPARISONS = {
#     # PU0 only
#     "PU0_bounds":            {"strategy": "PU0",       "wl": "bounds_0_20"},
#     "PU0_no_bounds":         {"strategy": "PU0",       "wl": "no_bounds"},

#     # PU200 sequential: PU0 layer weights + PU200 eta correction
#     "PU200_seq_b_nb":        {"strategy": "PU200_seq", "wl": "bounds", "eta": "no_bounds"},
#     "PU200_seq_b_b":         {"strategy": "PU200_seq", "wl": "bounds", "eta": "bounds_0_20"},

#     # PU200 joint
#     "PU200_no_bounds":       {"strategy": "PU200",     "all": "no_bounds"},
#     "PU200_bounds":          {"strategy": "PU200",     "all": "bounds_0_20"},
# }
# COMPARISONS = {
#     "PU200_no_bounds":       {"strategy": "PU200",     "all": "no_bounds"},
#     "PU200_bounds":          {"strategy": "PU200",     "all": "bounds_0_20"},
#     "PU200_all_bounds_0_20_no_layer1":          {"strategy": "PU200",     "all": "bounds_0_20_no_layer1"},
#     "PU200_all_no_bounds_no_layer1":          {"strategy": "PU200",     "all": "no_bounds_no_layer1"},  
# }
# COMPARISONS = {
#     # "PU0_bounds":            {"strategy": "PU0",       "wl": "bounds_0_20"},
#     # "PU0_no_bounds":         {"strategy": "PU0",       "wl": "no_bounds"},
#     # "PU0_bounds_0_20_no_layer1": {"strategy": "PU0",     "wl": "bounds_0_20_no_layer1"},
#     # "PU0_no_bounds_no_layer1":  {"strategy": "PU0",     "wl": "no_bounds_no_layer1"},  
#     "PU200_seq_b_nb":       {"strategy": "PU200_seq",     "wl": "bounds", "eta": "no_bounds"},
#     "PU200_seq_b_b":          {"strategy": "PU200_seq",     "wl": "bounds", "eta": "bounds_0_20"},
# }


#Final configuration for photons

COMPARISONS = {
    "PU200_seq_b_nb":           {"strategy": "PU200_seq", "wl": "bounds", "eta": "no_bounds", "offset": 0},
    "PU200_bounds_offset2p4":   {"strategy": "PU200",     "all": "bounds_0_20", "offset": 2.4},
    # "PU200_bounds_no_offset":   {"strategy": "PU200",     "all": "bounds_0_20", "offset": 0},
    # "PU200_all_no_bounds_no_layer1":          {"strategy": "PU200",     "all": "no_bounds_no_layer1", "offset": 0},
    # "PU200_all_no_bounds_no_layer1_offset2p4":          {"strategy": "PU200",     "all": "no_bounds_no_layer1","offset": 2.4}
}

#only used in derive_calibrations
STRATEGIES = ["PU0", "PU200", "PU200_seq"]
# STRATEGIES = ["PU0","PU200_seq"]

#----------------------------
# PLOTTING
#----------------------------

DEFAULT_COLORS = [
    "tab:olive",
    "tab:cyan",
    "deeppink",
    "darkorange",
    "darkorchid",
    "gold",
    "limegreen",
    "royalblue",
    "lightseagreen",
    "steelblue",
    "mediumslateblue",
    "coral",
]

PLOT_VARS = {
    "pt": {"branch": "pt","label": r"$p_T^{cluster}$ [GeV]","bins": 40,"range": [0, 200],"is_log": True},
    "eta": {"branch": "eta","label": r"$\eta^{cluster}$","bins": 40,"range": [-2.9, 2.9], "is_log": True},
    "abs_eta": {"branch": "eta","label": r"|$\eta^{cluster}$|","bins": 40,"range": [1.6, 2.9], "is_log": True},
    "phi": {"branch": "phi","label": r"$\phi^{cluster}$","bins": 40,"range": [-np.pi, np.pi],"is_log": True},
    "delta_r": {"branch": "delta_r","label": r"$\Delta R$(cluster,gen)","bins": 40,"range": [0,0.1],"is_log": True},
    "n_clusters": {"branch": "pt","label": r"$N_{clusters}$", "bins": 20, "range": [0, 20], "is_log": True},

    "pt_calib": {"branch": "pt","label": r"$p_T^{cluster}$ [GeV]","bins": 40,"range": [0, 100],"is_log": True},
    "eta_calib": {"branch": "eta","label": r"$\eta^{cluster}$","bins": 40,"range": [-2.9, 2.9], "is_log": False},
    "abs_eta_calib": {"branch": "eta","label": r"|$\eta^{cluster}$|","bins": 40,"range": [1.6, 2.9], "is_log": False},
    "phi_calib": {"branch": "phi","label": r"$\phi^{cluster}$","bins": 40,"range": [-np.pi, np.pi],"is_log": False},

    "pt_response": {"branch": "pt","label": r"$p_T^{cluster}$/$p_T^{gen}$","bins": 30,"range": [0.25, 2],"is_log": True},
    "eta_response": {"branch": "eta","label": r"$\eta^{cluster}-\eta^{gen}$","bins": 30,"range": [-0.05, 0.05], "is_log": False},
    "phi_response": {"branch": "phi","label": r"$\phi^{cluster}-\phi^{gen}$","bins": 30,"range": [-0.05, 0.05],"is_log": False},

    "pt_gen": {"branch": "pt", "label": r"$p_T^{gen}$ [GeV]", "bins": 10, "range": [0, 100], "is_log": False},
    "eta_gen": {"branch": "eta","label": r"$\eta^{gen}$","bins": 10,"range": [-2.9, 2.9], "is_log": False},
    "abs_eta_gen": {"branch": "eta","label": r"|$\eta^{gen}$|","bins": 10,"range": [1.6, 2.9], "is_log": False},
    "phi_gen": {"branch": "phi","label": r"$\phi^{gen}$","bins": 10,"range": [-3.14, 3.14],"is_log": False},

    "pt_eff": {"branch": "pt", "label": r"$p_T^{gen}$ [GeV]", "bins": 10,  "range":[0, 100], "is_log": False},
    "eta_eff": {"branch": "eta","label": r"$\eta^{gen}$", "bins": 10, "range": [-2.9, 2.9], "is_log": False},
    "abs_eta_eff": {"branch": "eta","label": r"|$\eta^{gen}$|", "bins": 10, "range": [1.6, 2.9], "is_log": False},
    "phi_eff": {"branch": "phi","label": r"$\phi^{gen}$", "bins": 10, "range": [-3.14, 3.14], "is_log": False}
}

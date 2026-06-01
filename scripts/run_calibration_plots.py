"""
scripts/run_calibration_plots.py
---------------------------------
Compare calibration strategies for a given triangle size (or all sizes).
Parallel to scripts/run_new_performance_plots.py — reuses the same
PerformancePlotter, CalibrationManager, PLOT_VARS, COMPARISONS, etc.

Calibration strategies are driven entirely by configs/config.py::COMPARISONS:
    "PU0_bounds"      -> layer weights only, derived on PU0
    "PU200_seq_mixed" -> PU0 layer weights  +  α/β derived on PU200
    "PU200_bounds"    -> all weights derived jointly on PU200

Usage examples
--------------
  # Response (mean) and resolution vs pt_gen / abs_eta_gen for triangle 0p03:
  python -m  scripts.run_calibration_plots --particles Photon --pileup PU200 --gen_pt_cut 20 --triangle 0p03  --resolution_plots --tag PU200_1p5

  # Scale (pT/pT_gen distribution) for every triangle:
  python -m  scripts.run_calibration_plots --particles Photon --pileup PU200 \\
         --gen_pt_cut 20 --scale_distribution

  # 1-D calibrated-pT distribution for triangle 0p03:
    python -m  scripts.run_calibration_plots --particles Photon --pileup PU200 --gen_pt_cut 20 --triangle 0p03  --distribution --tag PU200_1p5

  # Plotting weights for every strategy, triangle 0p03:
    python -m  scripts.run_calibration_plots --particles Photon --pileup PU200 --gen_pt_cut 20 --triangle 0p03  --weights --tag PU200_1p5

  # Eta residual vs abs(eta) with fit curve overlay, for every strategy, triangle 0p03:
  python -m  scripts.run_calibration_plots --particles Photon --pileup PU200 --gen_pt_cut 20 --triangle 0p03  --eta_residual --tag PU200_seq_1p5

  # Compare all triangles for one strategy (pass --all_triangles):
  python -m  scripts.run_calibration_plots --particles Photon --pileup PU200 \\
         --gen_pt_cut 20 --all_triangles --resolution_plots

For all options: python -m scripts.run_calibration_plots --help
"""

import os
import argparse
import awkward as ak

from configs.config import (
    PARQUET_BASE, EMU_CONFIG, CALIB_CONFIGS, COMPARISONS, PLOT_VARS, PU0_CONFIG_FOR_SEQ  
)
from data_handling.utils import build_parquet_dir, build_plotting_dir
import data_handling.files as f
import data_handling.calibration_functions as calib
from plotting.run_plots import PerformancePlotter, get_triangle_comparison
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_calib_plotting_dir(args, extra="calibration_comparison"):
    """Output directory, mirrors build_plotting_dir but with a calib suffix."""
    base = build_plotting_dir(args)
    out = os.path.join(base, extra)
    os.makedirs(out, exist_ok=True)
    return out


def load_and_calibrate(args, triangles, calib_dir):
    """
    Load matched events for *triangles* and apply every strategy in COMPARISONS.

    Returns
    -------
    results : dict
        results[tri_key] = {
            "raw":          {"pair_cluster": ..., "pair_gen": ...},
            <comparison_key>: {"pair_cluster": ..., "pair_gen": ...},
            ...
        }
    """
    parquet_dir = build_parquet_dir(args)
    raw_results = f.load_matching_results(parquet_dir)

    # CalibrationManager expects results keyed by triangle key
    manager = calib.CalibrationManager(
        results=raw_results,
        output_dir=calib_dir,
        configs=CALIB_CONFIGS,
        args=args,
    )

    results = {}
    for tri_key in triangles:
        results[tri_key] = {}

        # Store raw (uncalibrated) pair
        raw_data = manager.get_data(args, tri_key)
        results[tri_key]["raw"] = {
            "pair_cluster": raw_data["pair_cluster"],
            "pair_gen":     raw_data["pair_gen"],
        }

        # Apply every comparison strategy defined in config.py
        for comp_name, comp_cfg in COMPARISONS.items():
            strategy = comp_cfg["strategy"]
            print(f"\nApplying strategy {strategy} for comparison {comp_name} on triangle {tri_key}...")

            # Determine CALIB_CONFIGS key from comp_cfg
            if strategy == "PU0":
                config_name = f"{comp_cfg['wl']}" 
            elif strategy == "PU200":
                config_name = f"{comp_cfg['all']}"
            elif strategy == "PU200_seq":
                config_name = f"{comp_cfg['eta']}"  # wl config handled internally
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

       
            print(f"config_name: {config_name}")


            cluster_calib, gen, _ = manager.get_calibrated_cluster(
                strategy=strategy,
                config_name=config_name,
                key=tri_key,
                args=args,
                name=comp_name,
            )

            results[tri_key][comp_name] = {
                "pair_cluster": cluster_calib,
                "pair_gen":     gen,
            }

    return results , manager, raw_results
       

def build_weight_bundles(manager, tri_key, strategies_to_plot):
    """Extract weights for each strategy and format for plot_weights."""
    default_colors = ["tab:olive","tab:cyan", "darkorchid", "darkorange", "deeppink", "royalblue", "limegreen"]
    LABELS = {
        "PU0_bounds":      r"$w_l$ from PU0 bounds",
        "PU0_no_bounds":   r"$w_l$ from PU0 no bounds",
        "PU200_seq_b_nb": r"$w_l$ (PU0) b + $\alpha\beta$ (PU200) nb",
        "PU200_seq_b_b": r"$w_l$ (PU0) b + $\alpha\beta$ (PU200) b",
        "PU200_bounds":    r"PU200 w/ bounds",
        "PU200_no_bounds": r"PU200 w/o bounds",
    }

    bundles = []
    for i, strat_key in enumerate(strategies_to_plot):
        if strat_key == "raw":
            continue   # no weights to show for raw
        comp_cfg    = COMPARISONS[strat_key]
        # config_name = comp_cfg["config"]
        strategy    = comp_cfg["strategy"]

        if strategy == "PU0":
            config_name = comp_cfg["wl"]
            layer_config = config_name 
        elif strategy == "PU200_seq":
            config_name = comp_cfg["eta"]   # this is what varies for load()
            layer_config = PU0_CONFIG_FOR_SEQ
        elif strategy == "PU200":
            config_name = comp_cfg["all"]
            layer_config = config_name 

        weights = manager.load(strategy, config_name, tri_key)

        if strategy == "PU0":
            print(weights.keys())
            w = weights["layer"]
            w_layer = w[:13] if len(w) == 13 else w[:12]
            eta, bias = None, None

        elif strategy == "PU200":
            w = weights["all"]
            w_layer = w[:13] if len(w) == 15 else w[:12]
            eta, bias = w[-2], w[-1]

        elif strategy == "PU200_seq":
            w_layer = weights["layer"]
            eta, bias = weights["eta"][0], weights["eta"][1]

        bundles.append({
            "label": LABELS.get(strat_key, strat_key),
            "color": default_colors[i % len(default_colors)],
            "layer": w_layer,
            "eta":   eta,
            "bias":  bias,
            "remove_layer1": CALIB_CONFIGS[layer_config]["remove_layer1"],
        })

        weights = manager.load(strategy, config_name, tri_key)
    return bundles


def build_residual_bundles(manager, raw_results, tri_key, strategies_to_plot):
    default_colors = ["tab:cyan", "darkorchid", "darkorange", "deeppink", "royalblue", "limegreen"]
    LABELS = {
        "PU0_bounds":      r"$w_l$ PU0 bounds",
        "PU0_no_bounds":   r"$w_l$ PU0 no bounds",
        "PU200_seq_b_nb":  r"$w_l$ (PU0)b + $\alpha\beta$ (PU200)nb",
        "PU200_seq_b_b":   r"$w_l$ (PU0)b + $\alpha\beta$ (PU200)b",
        "PU200_bounds":    r"PU200 bounds",
        "PU200_no_bounds": r"PU200 no bounds",
    }
    bundles       = []
    color_idx     = 0
    shared_data   = None   # computed once, reused for all PU200_seq strategies

    for strat_key in strategies_to_plot:
        if strat_key == "raw":
            continue

        comp_cfg = COMPARISONS[strat_key]
        strategy = comp_cfg["strategy"]
        if strategy == "PU0":
            continue

        if strategy == "PU200_seq":
            config_name  = comp_cfg["eta"]
            layer_config = PU0_CONFIG_FOR_SEQ
        elif strategy == "PU200":
            config_name  = comp_cfg["all"]
            layer_config = config_name

        weights    = manager.load(strategy, config_name, tri_key)
        cfg        = CALIB_CONFIGS[layer_config]
        remove_l1  = cfg["remove_layer1"]

        raw_cluster = raw_results[tri_key]["pair_cluster"]
        layer_pt    = ak.to_numpy(ak.flatten(raw_cluster.layer_pt, axis=1))
        layer_np    = layer_pt[:, 1:13] if remove_l1 else layer_pt[:, :13]
        abs_eta     = np.abs(ak.to_numpy(ak.flatten(raw_cluster.eta, axis=1)))

        if strategy == "PU200_seq":
            w_layer = weights["layer"]
            alpha   = weights["eta"][0]
            beta    = weights["eta"][1]
        elif strategy == "PU200":
            w       = weights["all"]
            w_layer = w[:13] if len(w) == 15 else w[:12]
            alpha   = w[-2]
            beta    = w[-1]

        E_wl = np.sum(layer_np * np.asarray(w_layer), axis=1)

        # For PU200_seq: compute the data residual only once
        # (all seq strategies share the same layer weights)
        if strategy == "PU200_seq" and shared_data is None:
            gen_pt   = ak.to_numpy(ak.flatten(raw_results[tri_key]["pair_gen"].pt, axis=-1))
            shared_data = {
                "abs_eta":  abs_eta ,
                "residual": E_wl - gen_pt,   # actual data: gen - sum(wl*El)
            }

        # For PU200 joint each strategy has different layer weights so residual differs
        if strategy == "PU200":
            gen_pt   = ak.to_numpy(ak.flatten(raw_results[tri_key]["pair_gen"].pt, axis=-1))
            data_residual =  E_wl - gen_pt
            # data_residual = gen_pt - E_wl
            data_abs_eta  = abs_eta
        else:
            data_residual = shared_data["residual"]
            data_abs_eta  = shared_data["abs_eta"]

        bundles.append({
            "label":    LABELS.get(strat_key, strat_key),
            "color":    default_colors[color_idx % len(default_colors)],
            "eta":      data_abs_eta,
            "residual": data_residual,   # real data scatter, same for all PU200_seq
            "alpha":    alpha,
            "beta":     beta,
        })
        color_idx += 1

    return bundles

def build_comparison_bundle(results, tri_key, strategies_to_plot):
    """
    Build the list-of-dicts that PerformancePlotter expects,
    for ONE triangle, overlaying multiple calibration strategies.

    Each entry has:
        label, data (pair_cluster), gen (pair_gen), color (optional)
    """
    default_colors = [
        "tab:olive",    # raw
        "tab:cyan",     # PU0_bounds
        "darkorchid",   # PU200_seq_mixed
        "darkorange",   # PU200_bounds
        "deeppink",
        "gold",
        "limegreen",
    ]

    LABELS = {
        "raw":            r"raw $p_T$",
        "PU0_bounds":      r"$w_l$ from PU0 bounds",
        "PU0_no_bounds":   r"$w_l$ from PU0 no bounds",
        "PU200_seq_b_nb": r"$w_l$ (PU0) b + $\alpha\beta$ (PU200) nb",
        "PU200_seq_b_b": r"$w_l$ (PU0) b + $\alpha\beta$ (PU200) b",
        "PU200_bounds":    r"PU200 w/ bounds",
        "PU200_no_bounds": r"PU200 w/o bounds",
    }

    bundle = []

    for i, strat_key in enumerate(strategies_to_plot):
        data = results[tri_key][strat_key]
        cluster = data["pair_cluster"]
        gen     = data["pair_gen"]

        # Inject calibrated pt into the .pt field so pt_response works correctly
        if strat_key == "raw":
            # raw already has .pt, nothing to do
            pass
        else:
            # Ecalib_<strat_key> → overwrite .pt so _get_values picks it up
            ecalib = getattr(cluster, f"Ecalib_{strat_key}")
            cluster = ak.with_field(cluster, ecalib, "pt")

        bundle.append({
            "label":  LABELS.get(strat_key, strat_key) + f"  [{tri_key}]",
            "data":   cluster,
            "gen":    gen,
            "color":  default_colors[i % len(default_colors)],
        })
    return bundle


def build_triangle_bundle(results, strat_key):
    """
    Build the list-of-dicts for ONE strategy, overlaying all triangle sizes.
    Mirrors get_triangle_comparison() from run_plots.py.
    """
    default_colors = [
        "tab:olive", "tab:cyan", "darkorchid", "darkorange", "deeppink"
    ]
    bundle = []
    for i, tri_key in enumerate(EMU_CONFIG.keys()):
        if tri_key not in results:
            continue
        data = results[tri_key][strat_key]
        bundle.append({
            "label":  f"Tri {tri_key}",
            "data":   data["pair_cluster"],
            "gen":    data["pair_gen"],
            "color":  default_colors[i % len(default_colors)],
        })
    return bundle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare calibration strategies for a given triangle size"
    )

    # -- data selection (same flags as run_new_performance_plots.py) --
    parser.add_argument('-n',           type=int,   default=1,
                        help='Number of events')
    parser.add_argument('--particles',  type=str,   default='Photon',
                        help='Particle sample (Photon / Electron)')
    parser.add_argument('--pileup',     type=str,   default='PU0',
                        help='Pileup scenario: PU0 or PU200')
    parser.add_argument('--pt_cut',     type=float, default=0,
                        help='Cluster pT cut in GeV')
    parser.add_argument('--gen_pt_cut', type=float, default=0,
                        help='Gen pT cut in GeV')

    # -- triangle selection --
    parser.add_argument('--triangle',   type=str,   default=None,
                        choices=list(EMU_CONFIG.keys()),
                        help='Single triangle to study (default: all)')
    parser.add_argument('--all_triangles', action='store_true',
                        help='One canvas per strategy, all triangles overlaid')

    # -- which calibration strategies to include --
    parser.add_argument('--strategies', nargs='+',
                        choices=['raw'] + list(COMPARISONS.keys()),
                        default=['raw'] + list(COMPARISONS.keys()),
                        help='Calibration strategies to overlay (default: all)')

    # -- calibration weights directory --
    parser.add_argument('--calib_dir',  type=str,   default=None,
                        help='Directory where weight .parquet files are stored '
                             '(default: same as parquet_dir)')

    # -- plot types (same names as run_new_performance_plots.py) --
    parser.add_argument('--distribution',        action='store_true',
                        help='1-D pT distribution (raw vs calibrated)')
    parser.add_argument('--scale_distribution',  action='store_true',
                        help='pT / pT_gen response distribution')
    parser.add_argument('--resolution_plots',    action='store_true',
                        help='Profile plots: mean response and resolution vs gen variables')
    parser.add_argument('--binned_distributions', action='store_true',
                        help='Response distribution per bin of gen variable')
    
    # --data set types
    parser.add_argument('--matched', action='store_true',
                        help='matched dataset')
    parser.add_argument('--events', action='store_true',
                        help='full dataset')
    parser.add_argument('--filtered_events', action='store_true',
                        help='filtered dataset')
    parser.add_argument('--weights', action='store_true',
                    help='Plot layer weights per strategy')
    parser.add_argument('--eta_residual', action='store_true',
                    help='Plot eta correction residual vs |eta| with fit curve overlay')
    parser.add_argument('--tag', type=str,  default=None,
                    help='Add tag to the name of the plots')

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve triangle list
    if args.all_triangles:
        triangles = list(EMU_CONFIG.keys())
    elif args.triangle:
        triangles = [args.triangle]
    else:
        triangles = list(EMU_CONFIG.keys())

    # Directories
    parquet_dir = build_parquet_dir(args)
    calib_dir   = args.calib_dir if args.calib_dir else parquet_dir
    output_dir  = build_calib_plotting_dir(args)

    print(f"Parquet dir : {parquet_dir}")
    print(f"Calib dir   : {calib_dir}")
    print(f"Output dir  : {output_dir}")

    # ----------------------------------------------------------------
    # Load data & apply all calibrations
    # ----------------------------------------------------------------
    results, manager, raw_results = load_and_calibrate(args, triangles, calib_dir)

    strategies_to_plot = args.strategies   # user-chosen subset

    # ----------------------------------------------------------------
    # Title string (same logic as run_new_performance_plots.py)
    # ----------------------------------------------------------------
    if args.gen_pt_cut != 0 and args.pt_cut == 0:
        title = r"$p_T^{gen} >$" + f"{args.gen_pt_cut} GeV"
    elif args.pt_cut != 0 and args.gen_pt_cut == 0:
        title = r"$p_T^{cluster} >$" + f"{args.pt_cut} GeV"
    elif args.pt_cut != 0 and args.gen_pt_cut != 0:
        title = (r"$p_T^{cluster} >$" + f"{args.pt_cut} GeV and "
                 + r"$p_T^{gen} >$" + f"{args.gen_pt_cut} GeV")
    else:
        title = ""

    # ----------------------------------------------------------------
    # MODE A: one canvas per strategy, all triangles overlaid
    # ----------------------------------------------------------------
    if args.all_triangles:
        for strat_key in strategies_to_plot:
            strat_out = os.path.join(output_dir, strat_key)
            plotter = PerformancePlotter(args, output_dir=strat_out)
            bundle = build_triangle_bundle(results, strat_key)

            if args.distribution:
                for var in ["pt_calib", "eta_calib", "abs_eta_calib", "phi_calib"]:
                    plotter.plot_1d(bundle, var,
                                    filename=f"Dist_{var}_{strat_key}",
                                    title=title)

            if args.scale_distribution:
                plotter.plot_1d(bundle, "pt_response",
                                filename=f"Scale_pt_response_{strat_key}",
                                title=title)

            if args.resolution_plots:
                x_vars = ["pt_gen", "abs_eta_gen", "phi_gen"]
                y_vars = ["pt_response", "eta_response", "phi_response"]
                for x_var in x_vars:
                    for y_var in y_vars:
                        for mode in ['mean', 'resolution', 'rms']:
                            plotter.plot_profile(
                                bundle, x_var, y_var,
                                filename=f"Profile_{y_var}_vs_{x_var}_{strat_key}",
                                mode=mode, title=title
                            )

            if args.binned_distributions:
                for x_var, y_var in [("pt_gen", "pt_response"),
                                     ("abs_eta_gen", "pt_response"),
                                     ("phi_gen", "phi_response")]:
                    plotter.plot_distributions_per_bin(
                        datasets=bundle,
                        var_key=y_var,
                        binning_var_key=x_var,
                        filename=f"Dist_Response_{strat_key}",
                    )

    # ----------------------------------------------------------------
    # MODE B: one canvas per triangle, all strategies overlaid
    # ----------------------------------------------------------------
    else:
        for tri_key in triangles:
            tri_out = os.path.join(output_dir, tri_key)
            plotter = PerformancePlotter(args, output_dir=tri_out)
            print(strategies_to_plot)
            bundle  = build_comparison_bundle(results, tri_key, strategies_to_plot)
            tri_title = (title + f"\n Triangle {tri_key}").strip()

            # --- 1-D calibrated distributions ---
            if args.distribution:
                for var in ["pt_calib", "eta_calib", "abs_eta_calib", "phi_calib"]:
                    plotter.plot_1d(bundle, var,
                                    filename=f"Dist_{var}_{tri_key}_calib_comparison",
                                    title=tri_title)

            # --- Scale: full pT/pT_gen distribution ---
            if args.scale_distribution:
                plotter.plot_1d(bundle, "pt_response",
                                filename=f"Scale_pt_response_{tri_key}_calib_comparison",
                                title=tri_title)

            # --- Response & resolution profiles ---
            if args.resolution_plots:
                x_vars = ["pt_gen", "abs_eta_gen", "phi_gen"]
                y_vars = ["pt_response", "eta_response", "phi_response"]
                for x_var in x_vars:
                    for y_var in y_vars:
                        for mode in ['mean', 'resolution', 'rms']:
                            plotter.plot_profile(
                                bundle, x_var, y_var,
                                filename=f"Profile_{y_var}_vs_{x_var}_{tri_key}",
                                mode=mode, title=tri_title
                            )

            # --- Response distribution per gen-variable bin ---
            if args.binned_distributions:
                for x_var, y_var in [("pt_gen",      "pt_response"),
                                     ("abs_eta_gen",  "pt_response"),
                                     ("phi_gen",      "phi_response")]:
                    plotter.plot_distributions_per_bin(
                        datasets=bundle,
                        var_key=y_var,
                        binning_var_key=x_var,
                        filename=f"Dist_Response_{tri_key}",
                    )

            if args.weights:
                weight_bundles = build_weight_bundles(manager, tri_key, strategies_to_plot)
                plotter.plot_weights(weight_bundles,
                                    filename=f"Weights_{tri_key}",
                                    args=args,
                                    title=tri_title)
                
            if args.eta_residual:
                residual_bundles = build_residual_bundles(
                    manager, raw_results, tri_key, strategies_to_plot
                )
                plotter.plot_eta_residual(
                    residual_bundles,
                    filename=f"EtaResidual_{tri_key}",
                    args=args,
                    title=tri_title,
                )
            
    if not (args.distribution or args.scale_distribution
            or args.resolution_plots or args.binned_distributions):
        print(
            "[INFO] No plot type selected. Add one or more of:\n"
            "  --distribution  --scale_distribution  "
            "--resolution_plots  --binned_distributions"
        )


if __name__ == "__main__":
    main()

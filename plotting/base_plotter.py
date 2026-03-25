import matplotlib.pyplot as plt
import mplhep
import numpy as np
import awkward as ak
from matplotlib.patches import Rectangle

'''PlotManagerjob is:

Standardization: Every plot gets the same CMS headers, legend styles, and save logic automatically.

Logic Separation: It will have specific methods for "Distributions", "Profiles" (Mean/Resolution vs X), and "2D Maps".

Config-Driven: It reads a dictionary of what you want to compare (e.g., Triangle sizes or Calib vs Uncalib).'''

class PlotManager:
    def __init__(self, args, output_dir):
        self.args = args
        self.output_dir = output_dir
        plt.style.use(mplhep.style.CMS)

    def _setup_canvas(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        return fig, ax

    def _finalize_plot(self, ax, xlabel, ylabel, title, filename, is_log=False):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if is_log: ax.set_yscale('log')
        
        # Standard CMS Decorations
        mplhep.cms.label("Preliminary", data=True, 
                         rlabel=f"{self.args.pileup} {self.args.particles}")
        
        # Legend with automatic Title (Cuts)
        legend_title = ""
        if self.args.gen_pt_cut: legend_title += fr"$p_T^{{gen}} > {self.args.gen_pt_cut}$ GeV"
        ax.legend(title=legend_title, frameon=True, facecolor='white', edgecolor='black')
        
        ax.grid(linestyle=":")
        plt.tight_layout()
        
        # Save logic
        out_path = f"{self.output_dir}/{filename}"
        plt.savefig(f"{out_path}.png", dpi=300)
        plt.savefig(f"{out_path}.pdf")
        print(f"--- Saved: {out_path}.png")
        plt.close()

    # --- TASK: Distributions (1D Hists) ---
    def plot_distributions(self, datasets, var_key, bins, range_, filename):
        """
        datasets: list of dicts [{'data': ak_array, 'label': str, 'color': str}]
        """
        fig, ax = self._setup_canvas()
        for ds in datasets:
            data = ak.to_numpy(ak.flatten(ds['data'], axis=-1))
            ax.hist(data, bins=bins, range=range_, color=ds['color'], 
                    alpha=0.2, histtype='stepfilled')
            ax.hist(data, bins=bins, range=range_, color=ds['color'], 
                    label=ds['label'], histtype='step', linewidth=2.5)
        
        self._finalize_plot(ax, var_key, "Counts", "", filename, is_log=True)

    # --- TASK: Performance Profiles (Mean/Resolution vs Variable) ---
    def plot_performance_vs_x(self, datasets, x_var, y_mode, bins, range_, filename):
        """
        y_mode: 'response' or 'resolution'
        """
        fig, ax = self._setup_canvas()
        for ds in datasets:
            # This calls your existing math logic (compute_response_calib)
            # but inside the loop for comparisons
            x_pts, resp, resp_err, resol, resol_err = compute_response_calib(
                ds['matched'], ds['gen'], self.args, 
                response_variable=ds['resp_var'], bin_variable=x_var, 
                quantity=ds['qty'], bin_n=len(bins)-1, range_=range_
            )
            
            y_val = resp if y_mode == 'response' else resol
            y_err = resp_err if y_mode == 'response' else resol_err
            
            ax.errorbar(x_pts, y_val, yerr=y_err, xerr=(x_pts[1]-x_pts[0])/2,
                        marker="s", ls="None", label=ds['label'], color=ds['color'])

        ylabel = "Response" if y_mode == 'response' else "Resolution"
        self._finalize_plot(ax, x_var, ylabel, "", filename)

import matplotlib.pyplot as plt
import mplhep as hep
import os

class BasePlotter:
    def __init__(self, args, output_dir):
        self.args = args
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        hep.style.use(hep.style.CMS)

    def _apply_decorations(self, ax, xlabel, ylabel, is_log=False, title=None):
        hep.cms.label("Preliminary", data=True, 
                      rlabel=f"{self.args.pileup} {self.args.particles}", ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if is_log: ax.set_yscale('log')
        ax.grid(linestyle=":", alpha=0.6)
        
        # Legend with automatic Title
        legend_title = title if title else ""
        if self.args.gen_pt_cut > 0: 
            legend_title += f"\n$p_T^{{gen}} > {self.args.gen_pt_cut}$ GeV"
        
        ax.legend(title=legend_title, frameon=True, facecolor='white', edgecolor='black')
        plt.tight_layout()

    def save(self, filename):
        for ext in ['png', 'pdf']:
            plt.savefig(os.path.join(self.output_dir, f"{filename}.{ext}"), dpi=300)
        plt.close()


class DistributionPlotter(BasePlotter):
    def _get_flat_data(self, dataset, var_name):
        """
        Internal helper to find the correct field and flatten it.
        """
        # 1. Determine which field to look for
        if "qty" in dataset and dataset["qty"] == "calib":
            # Look for the Ecalib field created by your CalibrationManager
            field = f"Ecalib_{dataset['calib_name']}"
            data = dataset["matched"][field]
        else:
            # Fallback to standard pt, eta, phi
            data = dataset["matched"][var_name]

        # 2. Flatten from (Events, Clusters) -> (Total Clusters)
        # axis=-1 ensures we flatten the inner nested dimension
        return ak.to_numpy(ak.flatten(data, axis=-1))

    def plot_comparison(self, datasets, var_name, bins, range_, filename, xlabel, title=""):
        """
        datasets: list of dicts [{'matched': ak_array, 'label': str, 'color': str, ...}]
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for ds in datasets:
            # The class now handles the flattening logic!
            flat_array = self._get_flat_data(ds, var_name)
            
            # Draw the histogram
            ax.hist(flat_array, bins=bins, range=range_, color=ds['color'], 
                    alpha=0.15, histtype='stepfilled')
            ax.hist(flat_array, bins=bins, range=range_, color=ds['color'], 
                    label=ds['label'], histtype='step', linewidth=2.5)

        self._apply_decorations(ax, xlabel, "Counts", is_log=True, title=title)
        self.save(filename)
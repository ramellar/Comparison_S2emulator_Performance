import matplotlib.pyplot as plt
from configs.config import PLOT_VARS, EMU_CONFIG
import matplotlib.colors as colors
import mplhep as hep
import numpy as np
import awkward as ak
import os
from scipy.stats import binned_statistic


# def get_triangle_comparison(data_source, total_gen=None):
#     bundle = []
#     for tri_key in EMU_CONFIG.keys():
#         content = data_source[tri_key]
#         entry = {'label': f"Tri {tri_key}"}
        
#         # Case A: Content is a dictionary (matched_events, filtered_events)
#         if isinstance(content, dict):
#             cl_key = next((k for k in content.keys() if "cluster" in k), None)
#             if cl_key: entry['data'] = content[cl_key]
            
#             gen_key = next((k for k in content.keys() if "gen" in k), None)
#             if gen_key: entry['gen'] = content[gen_key]
        
#         # Case B: Content is the array itself (bare events)
#         else:
#             entry['data'] = content
        
#         # Attach total_gen if provided (usually for efficiency or multiplicity)
#         if total_gen is not None:
#             entry['total_gen'] = total_gen
            
#         bundle.append(entry)
#     return bundle


def get_triangle_comparison(data_source, total_gen=None):
    bundle = []
    for tri_key in EMU_CONFIG.keys():
        content = data_source[tri_key]
        entry = {'label': f"Tri {tri_key}"}
        
        if isinstance(content, dict):
            # For matched/filtered data
            cl_key = next((k for k in content.keys() if "cluster" in k), None)
            if cl_key: entry['data'] = content[cl_key]
            gen_key = next((k for k in content.keys() if "gen" in k), None)
            if gen_key: entry['gen'] = content[gen_key]
        else:
            # For bare events
            entry['data'] = content
            # CRITICAL: Attach the gen data so _get_values can find it!
            if total_gen is not None:
                entry['gen'] = total_gen
        
        if total_gen is not None:
            entry['total_gen'] = total_gen
            
        bundle.append(entry)
    return bundle

class PerformancePlotter:
    def __init__(self, args, output_dir="plots"):
        """
        args: The argparse object (used for labels like pileup/particles)
        output_dir: Where to save the images
        """
        self.args = args
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        hep.style.use(hep.style.CMS)
    

    def _get_values(self, ds, var_key):
        # Determine if we need absolute value based on the REQUESTED key
        use_abs = var_key.startswith("abs_")
        
        # Handle Gen-specific logic (e.g., abs_eta_eff or pt_gen)
        if var_key.endswith("_gen") or "_eff" in var_key:
            # Use the branch defined in PLOT_VARS (e.g., "eta")
            branch = PLOT_VARS[var_key]["branch"]
            vals = self._extract_array(ds.get('gen', ds.get('total_gen')), branch)
            return np.abs(vals) if use_abs else vals

        # Handle Response variables
        if "_response" in var_key:
            base_var = var_key.replace("_response", "")
            # Response is usually (reco - gen) or (reco / gen)
            cl_vals = self._extract_array(ds['data'], base_var)
            gen_vals = self._extract_array(ds['gen'], base_var)
            if base_var == "pt":
                return np.divide(cl_vals, gen_vals, out=np.zeros_like(cl_vals), where=gen_vals!=0)
            return cl_vals - gen_vals
        
        #  Handle n_clusters
        if var_key == "n_clusters":
            return ak.num(ds['data'].pt, axis=-1)

        branch = PLOT_VARS[var_key]["branch"]
        vals = self._extract_array(ds['data'], branch)
        return np.abs(vals) if use_abs else vals
    
    def _extract_array(self, data, branch):
        """
        Internal helper: Handles .attribute, ['key'], and flattening.
        """
        # 1. Try to get the branch (handle attribute vs dictionary key)
        if hasattr(data, branch):
            arr = getattr(data, branch)
        else:
            arr = data[branch]
        
        # 2. Flatten the Awkward array to 1D numpy
        return ak.to_numpy(ak.flatten(arr, axis=-1))

    def plot_1d(self, datasets, var_key, filename, title=""):
        # This now handles BOTH distributions AND responses
        conf = PLOT_VARS[var_key]
        fig, ax = plt.subplots(figsize=(10, 10))
        default_colors = ["tab:olive", "tab:cyan", "darkorchid" , "darkorange", "deeppink", "lightseagreen", "steelblue", "gold", "mediumslateblue", "coral"]

        for i, ds in enumerate(datasets):
            values = self._get_values(ds, var_key)
            color = ds.get('color', default_colors[i % len(default_colors)])
            
            ax.hist(values, bins=conf["bins"], range=conf["range"], color=color,
                    label=ds['label'], histtype='step', linewidth=2.5)
            ax.hist(values, bins=conf["bins"], range=conf["range"], color=color, 
                    alpha=0.2, histtype='stepfilled')

        # Apply CMS Style
        hep.cms.label("Preliminary", data=True, 
                      rlabel=f"{self.args.pileup} {self.args.particles}", ax=ax)
        
        ax.set_xlabel(conf["label"])
        ax.set_ylabel("Counts")
        if conf["is_log"] == True :
            ax.set_yscale('log')
        ax.grid(linestyle=":", alpha=0.6)
        ax.legend(title=title, frameon=True, facecolor='white', edgecolor='black', fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"--- Plot Saved: {save_path}")
    
    def plot_multi_distribution(self, datasets, var_key, filename, title="", is_log=True):
        conf = PLOT_VARS[var_key]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        default_colors = ["tab:olive", "tab:cyan", "darkorchid", "darkorange", "deeppink"]

        for i, ds in enumerate(datasets):
            color = ds.get('color', default_colors[i % len(default_colors)])
            vals = self._get_values(ds, conf["branch"])
            
            # Use hist for the stepped-filled look from your old code
            ax.hist(vals, bins=conf['bins'], range=conf['range'], 
                    color=color, alpha=0.2, histtype='stepfilled')
            ax.hist(vals, bins=conf['bins'], range=conf['range'], 
                    color=color, label=ds['label'], histtype='step', linewidth=2.5)

        if is_log: ax.set_yscale('log')
        
        ax.set_xlabel(conf['label'])
        ax.set_ylabel("Counts")
        hep.cms.label("Preliminary", data=True, rlabel=f"{self.args.pileup}", ax=ax)
        ax.legend(title=title)

        save_path = os.path.join(self.output_dir, f"Dist_{var_key}_{filename}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"--- Plot Saved: {save_path}")
        
    def plot_2d(self, dataset, x_var_key, y_var_key, filename, title=""):
        """
        Plots a 2D histogram of two variables for a SINGLE dataset.
        """
        x_conf = PLOT_VARS[x_var_key]
        y_conf = PLOT_VARS[y_var_key]

        print('x_var_key',x_var_key)
        
        x_vals = self._get_values(dataset, x_var_key)
        y_vals = self._get_values(dataset, y_var_key)
        
        fig, ax = plt.subplots(figsize=(12, 10))

        # Use LogNorm to see the full range of density
        h = ax.hist2d(x_vals, y_vals, 
                      bins=[x_conf['bins'], y_conf['bins']], 
                      range=[x_conf['range'], y_conf['range']],
                      cmap='RdPu')
        
        fig.colorbar(h[3], ax=ax, label='Counts')
        tag = dataset['label'].replace("Tri ", "").replace(" ", "_").replace("p", ".")
        ax.legend(title=title, fontsize=16)
        ax.grid(linestyle=":")
        hep.cms.label("Preliminary", data=True, rlabel=f"{tag}-{self.args.particles}-{self.args.pileup}", ax=ax)

        ax.set_xlabel(x_conf['label'])
        ax.set_ylabel(y_conf['label'])
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{filename}_{x_var_key}_vs_{y_var_key}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"--- Plot Saved: {save_path}")

    def plot_2d_batch(self, bundle, correlations, title):
        """
        bundle: List of datasets from get_triangle_comparison
        correlations: List of tuples, e.g., [("pt_gen", "pt"), ("eta", "phi")]
        """
        for ds in bundle:
            # Extract the raw name from the label (e.g., "Tri 0p03" -> "0p03")
            tag = ds['label'].replace("Tri ", "").replace(" ", "_")
            
            for x_var, y_var in correlations:
                filename = f"2D_{tag}"
                # title = f"Triangle Size: {tag}"
                
                # Call the existing 2D method
                self.plot_2d(ds, x_var, y_var, filename=filename, title=title)
  

    def plot_efficiency(self, datasets, conf, title=""):
        """
        Plots Efficiency (Matched Gen / Total Gen) as a function of x_var_key.
        """
        x_conf = conf
        x_var_key = x_conf["branch"]
        fig, ax = plt.subplots(figsize=(10, 10))
        
        #Define binning
        bin_edges = np.linspace(x_conf['range'][0], x_conf['range'][1], x_conf['bins'] + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        default_colors = ["tab:olive", "tab:cyan", "darkorchid", "darkorange", "deeppink"]


        for i, ds in enumerate(datasets):
            color = ds.get('color', default_colors[i % len(default_colors)])

            # Efficiency = (Gen particles that were matched) / (All Gen particles)
            matched_gen_x = self._get_values(ds, f"{x_var_key}_gen")
            total_gen_x = self._extract_array(ds['total_gen'], x_var_key)

            h_matched, _ = np.histogram(matched_gen_x, bins=bin_edges)
            h_total, _ = np.histogram(total_gen_x, bins=bin_edges)
            
            eff = np.divide(h_matched, h_total, out=np.zeros_like(h_matched, dtype=float), where=h_total!=0)
            
            # Simple error calculation (binomial)
            err = np.sqrt(eff * (1 - eff) / h_total, out=np.zeros_like(eff), where=h_total!=0)

            ax.errorbar(bin_centers, eff, xerr=(bin_edges[1]-bin_edges[0])/2, yerr=err, label=ds['label'], fmt='o', markersize=6, color=color)

        # ax.set_ylim(0, 1.1)
        ax.set_ylabel("Efficiency")
        ax.set_xlabel(x_conf['label'])
        ax.legend(title=title)
        ax.grid(linestyle=":")
        hep.cms.label("Preliminary", data=True, rlabel=f"{self.args.particles}-{self.args.pileup}", ax=ax)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"Efficiency_vs_{x_var_key}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"--- Plot Saved: {save_path}")

    def effrms(self, x, c=0.68):
        """ Computes half-width of the smallest interval containing c% of the distribution. """
        # if len(x) < 5: return 0 # Need enough points to find an interval
        x_sorted = np.sort(x)
        m = int(c * len(x_sorted))
        # Find the width of all intervals containing 'm' points
        widths = x_sorted[m:] - x_sorted[:-m]
        return np.min(widths) / 2.0

    def plot_profile(self, datasets, x_var_key, y_var_key, filename, mode='mean', title=""):
        x_conf = PLOT_VARS[x_var_key]
        y_conf = PLOT_VARS[y_var_key]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        default_colors = ["tab:olive", "tab:cyan", "darkorchid" , "darkorange", "deeppink"]

        bin_edges = np.linspace(x_conf['range'][0], x_conf['range'][1], x_conf['bins'] + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        for i, ds in enumerate(datasets):
            color = ds.get('color', default_colors[i % len(default_colors)])
            x_vals = self._get_values(ds, x_var_key)
            y_vals = self._get_values(ds, y_var_key)

            # Get counts per bin for error calculation
            counts, _, _ = binned_statistic(x_vals, y_vals, statistic='count', bins=bin_edges)

            if mode == 'mean':
                stat, _, _ = binned_statistic(x_vals, y_vals, statistic='mean', bins=bin_edges)
                # Error on mean: sigma / sqrt(N)
                stds, _, _ = binned_statistic(x_vals, y_vals, statistic=lambda x: np.std(x), bins=bin_edges)
                y_err = np.divide(stds, np.sqrt(counts), out=np.zeros_like(stds), where=counts>0)
                ylabel = f"<{y_conf['label']}>"

            elif mode == 'resolution':
                means, _, _ = binned_statistic(x_vals, y_vals, statistic='mean', bins=bin_edges)
                stds, _, _ = binned_statistic(x_vals, y_vals, statistic=lambda x: np.std(x), bins=bin_edges)
                
                if 'pt' in y_var_key:
                    stat = np.divide(stds, means, out=np.zeros_like(stds), where=means!=0)
                    ylabel = r"$\sigma_{cluster} / \mu_{cluster}$"
                else:
                    stat = stds # For angles, resolution is just the width
                    ylabel = r"$\sigma_{cluster}$"
                
                # Statistical Error on Resolution: Resolution / sqrt(2N - 2)
                y_err = np.divide(stat, np.sqrt(2*counts - 2), out=np.zeros_like(stat), where=counts>1)

            elif mode == 'rms':
                means, _, _ = binned_statistic(x_vals, y_vals, statistic='mean', bins=bin_edges)
                eff_stds, _, _ = binned_statistic(x_vals, y_vals, 
                                                 statistic=lambda x: self.effrms(x), 
                                                 bins=bin_edges)
                
                if 'pt' in y_var_key:
                    stat = np.divide(eff_stds, means, out=np.zeros_like(eff_stds), where=means!=0)
                    ylabel = r"$\sigma^{eff-RMS}_{cluster} / \mu_{cluster}$"
                else:
                    stat = eff_stds # For angles, resolution is just the width
                    ylabel = r"$\sigma^{eff-RMS}_{cluster}$"
                
                # Statistical Error on Resolution: Resolution / sqrt(2N - 2)
                y_err = np.divide(stat, np.sqrt(2*counts - 2), out=np.zeros_like(stat), where=counts>1)

            # Masking in case there are few stats
            mask = ~np.isnan(stat) & (counts > 2) # Require at least 3 points to plot
            
            if np.any(mask):
                ax.errorbar(bin_centers[mask], stat[mask], yerr=y_err[mask],
                            xerr=(bin_edges[1]-bin_edges[0])/2,
                            label=ds['label'], color=color, fmt='o', 
                            markersize=8)

        hep.cms.label("Preliminary", data=True, rlabel=f"{self.args.particles}-{self.args.pileup}", ax=ax)
        ax.set_xlabel(x_conf['label'])
        ax.set_ylabel(ylabel)
        ax.grid(linestyle=":", alpha=0.6)
        ax.legend(title=title, fontsize=15)
        
        save_path = os.path.join(self.output_dir, f"{filename}_{mode}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"--- Plot Saved: {save_path}")

    def plot_distributions_per_bin(self, datasets, var_key, binning_var_key, filename):
        """
        Creates a separate plot for EVERY bin defined in PLOT_VARS[binning_var_key].
        Shows the distribution of var_key (e.g., pt_response) for that bin.
        """
        var_conf = PLOT_VARS[var_key]
        bin_conf = PLOT_VARS[binning_var_key]
        
        # 1. Define the bin edges exactly as the Profile Plot does
        bin_edges = np.linspace(bin_conf['range'][0], bin_conf['range'][1], bin_conf['bins'] + 1)
        
        default_colors = ["tab:olive", "tab:cyan", "darkorchid" , "darkorange", "deeppink"]

        # 2. Loop over every bin interval
        for j in range(len(bin_edges) - 1):
            low, high = bin_edges[j], bin_edges[j+1]
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            for i, ds in enumerate(datasets):
                color = ds.get('color', default_colors[i % len(default_colors)])
                
                # Extract aligned arrays
                vals_to_plot = self._get_values(ds, var_key)
                bin_vals = self._get_values(ds, binning_var_key)
                
                # Mask for the CURRENT bin
                mask = (bin_vals >= low) & (bin_vals < high)
                slice_data = vals_to_plot[mask]
                
                # if len(slice_data) < 5: continue
                # has_data = True

                # Use density=True if you want to compare SHAPES (fractions) 
                # rather than raw entry counts between different triangle sizes
                ax.hist(slice_data, bins=var_conf['bins'], range=var_conf['range'], 
                        color=color, label=f"{ds['label']} (N={len(slice_data)})", 
                        histtype='step', linewidth=2.5)


            # Styling to match the Profile Plots
            bin_label = f"{low:.1f} < {bin_conf['label']} < {high:.1f}"
            hep.cms.label("Preliminary", data=True, rlabel=f"{self.args.particles} {self.args.pileup}", ax=ax)
            
            ax.set_xlabel(var_conf['label'])
            ax.set_ylabel("Counts")
            ax.legend(title=bin_label, fontsize=14)
            ax.grid(linestyle=":", alpha=0.6)
            
            # Save with the bin index in the filename
            save_name = f"{filename}_{var_key}_in_{binning_var_key}_bin{j}.png"
            save_path = os.path.join(self.output_dir, "bin_distributions", save_name)
            
            # Ensure the sub-directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"--- Bin Distribution Saved: {save_path}")

    def plot_compare_bins(self, datasets, var_key, binning_var_key, slices, filename):
        """
        For each slice in 'slices', creates a plot comparing all datasets.
        slices: List of tuples, e.g., [(20, 40), (40, 60), (80, 100)]
        """
        from configs.config import PLOT_VARS
        var_conf = PLOT_VARS[var_key]
        bin_conf = PLOT_VARS[binning_var_key]
        
        default_colors = ["tab:olive", "tab:cyan", "darkorchid" , "darkorange", "deeppink"]

        for low, high in slices:
            fig, ax = plt.subplots(figsize=(10, 10))
            
            for i, ds in enumerate(datasets):
                color = ds.get('color', default_colors[i % len(default_colors)])
                
                # Get raw values for both variables
                all_var = self._get_values(ds, var_key)
                all_bin = self._get_values(ds, binning_var_key)
                
                # Filter for this specific slice
                mask = (all_bin >= low) & (all_bin < high)
                slice_data = all_var[mask]
                
                if len(slice_data) < 5: continue

                # Plot
                ax.hist(slice_data, bins=var_conf['bins'], range=var_conf['range'], 
                        color=color, label=ds['label'], histtype='step', linewidth=2.5)

            # Labels and Styling
            title = f"{low} < {bin_conf['label']} < {high}"
            hep.cms.label("Preliminary", data=True, rlabel=f"{self.args.pileup}", ax=ax)
            ax.set_xlabel(var_conf['label'])
            ax.set_ylabel("Counts")
            ax.legend(title=title, frameon=True)
            ax.grid(linestyle=":", alpha=0.6)
            
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f"{filename}_{var_key}_slice_{low}_{high}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"--- Plot Saved: {save_path}")

import matplotlib.pyplot as plt
from configs.config import PLOT_VARS, EMU_CONFIG
import matplotlib.colors as colors
import mplhep as hep
import numpy as np
import awkward as ak
import os
from scipy.stats import binned_statistic

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
            cl_vals = self._extract_array(ds['data'], base_var)
            gen_vals = self._extract_array(ds['gen'], base_var)
            
            if base_var == "pt":
                # Ratio for energy/momentum
                return np.divide(cl_vals, gen_vals, out=np.zeros_like(cl_vals), where=gen_vals!=0)
            
            # Absolute difference for coordinates (eta, phi)
            res = cl_vals - gen_vals
            
            if base_var == "phi":
                # Shortest path on the circle
                res = (res + np.pi) % (2 * np.pi) - np.pi
                
            return res

        branch = PLOT_VARS[var_key]["branch"]
        vals = self._extract_array(ds['data'], branch)
        return np.abs(vals) if use_abs else vals
    

    def _get_ncluster_values(self, ds,x_var_key='pt', gen_n=None):
        """
        Returns (x_vals, y_vals) for n_clusters at the event level.
        If gen_n=1: Only 1-particle events.
        If gen_n=None: All events (1 or 2 particles).
        """
        gen_ref = ds.get('total_gen', ds.get('gen'))
        
        # 1. Apply Filter
        if gen_n is not None:
            # print("GEN FLAGS NUM:", ak.num(gen_ref.pt, axis=-1) )
            mask = (ak.num(gen_ref.pt, axis=-1) == gen_n)
        else:
            # No filter: use every event in the file
            mask = np.ones(len(gen_ref), dtype=bool)

        data_masked = ds['data'][mask]
        data = ds['data']
        gen_masked = gen_ref[mask]

        # 2. Extract Values
        # Y = Total count of clusters in each event
        # print("DATA MASKED PT:", ak.num(data_masked.pt, axis=-1))
        # print("DATA PT:", ak.num(data.pt, axis=-1))
        y_vals = ak.to_numpy(ak.num(data_masked.pt, axis=-1))
        branch = PLOT_VARS[x_var_key]["branch"]
        
        # X = The pT of the leading particle (to define the bin on the x-axis)
        if "abs" in x_var_key:
            x_vals = np.abs(ak.to_numpy(ak.flatten(gen_masked[branch], axis=-1)))
        else:
            x_vals = ak.to_numpy(ak.flatten(gen_masked[branch], axis=-1))

        return x_vals, y_vals

    def plot_nclusters_per_bin(self, datasets, binning_var_key, title="", gen_n=1):
        """
        Replaces your plot_clusters_per_bin. 
        Plots N_clusters distribution in bins of binning_var_key (pt_gen or abs_eta_gen).
        """
        bin_conf = PLOT_VARS[binning_var_key]
        ncl_conf = PLOT_VARS["n_clusters"] 
        
        bin_edges = np.linspace(bin_conf['range'][0], bin_conf['range'][1], bin_conf['bins'] + 1)
        
        for j in range(len(bin_edges) - 1):
            low, high = bin_edges[j], bin_edges[j+1]
            fig, ax = plt.subplots(figsize=(10, 10))
            
            for i, ds in enumerate(datasets):
                default_colors = ["tab:olive", "tab:cyan", "darkorchid" , "darkorange", "deeppink", "lightseagreen", "steelblue", "gold", "mediumslateblue", "coral"]
                color = ds.get('color', default_colors[i % len(default_colors)])
                bin_vals, n_clusters = self._get_ncluster_values(ds, gen_n=gen_n)
                
                mask = (bin_vals >= low) & (bin_vals < high)
                slice_data = n_clusters[mask]
                
                # print(f"Bin {j}: {low:.1f} <= {bin_conf['label']} < {high:.1f} -> {len(mask)} entries")
                # print(len(n_clusters))
         
                if len(slice_data) == 0: continue

                ax.hist(slice_data, bins=ncl_conf['bins'], range=ncl_conf['range'], 
                        color=color, label=ds['label'], histtype='step', linewidth=2.5)
                ax.hist(slice_data, bins=ncl_conf['bins'], range=ncl_conf['range'], 
                        color=color, histtype='stepfilled', alpha=0.2)

            bin_label = f"{low:.1f} < {bin_conf['label']} < {high:.1f}"
            cuts = []
            if getattr(self.args, 'gen_pt_cut', 0) > 0:
                cuts.append(fr"$p_T^{{\mathrm{{gen}}}} > {self.args.gen_pt_cut}$ GeV")
            if getattr(self.args, 'pt_cut', 0) > 0:
                cuts.append(fr"$p_T^{{\mathrm{{cluster}}}} > {self.args.pt_cut}$ GeV")

            if title=="":
                full_title = bin_label + ("\n" + " & ".join(cuts) if cuts else "")
            else:
                full_title = title + "\n" + bin_label + ("\n" + " & ".join(cuts) if cuts else "")

            ax.set_yscale('log')
            ax.set_xlabel(r"$N_{clusters}$")
            ax.set_ylabel("Counts")
            ax.grid(linestyle=":")
            ax.legend(title=full_title, fontsize=16, title_fontsize=15)
            
            hep.cms.label("Preliminary", data=True, 
                          rlabel=f"{self.args.pileup} {self.args.particles} - {gen_n} gen part.", ax=ax)
            
            save_name = f"NClusters_{binning_var_key}_bin{j}_gen{gen_n}.png"
            save_name_pdf = f"NClusters_{binning_var_key}_bin{j}_gen{gen_n}.pdf"
            os.makedirs(os.path.join(self.output_dir), exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, save_name))
            plt.savefig(os.path.join(self.output_dir, save_name_pdf))
            print(f"--- Plot Saved: {os.path.join(self.output_dir, save_name)}")
    
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
    

    def plot_1d(self, datasets, var_key, filename, title="", gen_n=None):
        # This now handles BOTH distributions AND responses
        conf = PLOT_VARS[var_key]
        fig, ax = plt.subplots(figsize=(10, 10))
        default_colors = ["tab:olive", "tab:cyan", "darkorchid" , "darkorange", "deeppink", "lightseagreen", "steelblue", "gold", "mediumslateblue", "coral"]

        for i, ds in enumerate(datasets):
            # values = self._get_values(ds, var_key)
            if var_key == "n_clusters":
                _, values = self._get_ncluster_values(ds, gen_n=gen_n)
            else:
                values = self._get_values(ds, var_key)

            color = ds.get('color', default_colors[i % len(default_colors)])
            
            ax.hist(values, bins=conf["bins"], range=conf["range"], color=color,
                    label=ds['label'], histtype='step', linewidth=2.5)
            ax.hist(values, bins=conf["bins"], range=conf["range"], color=color, 
                    alpha=0.2, histtype='stepfilled')

        if gen_n is not None:
            hep.cms.label("Preliminary", data=True, 
                        rlabel=f"{self.args.pileup} {self.args.particles}- {gen_n} gen part.", ax=ax)
        else:
            hep.cms.label("Preliminary", data=True, 
                        rlabel=f"{self.args.pileup} {self.args.particles}", ax=ax)
        
        ax.set_xlabel(conf["label"])
        ax.set_ylabel("Counts")
        if conf["is_log"] == True :
            ax.set_yscale('log')
        ax.grid(linestyle=":")
        ax.legend(title=title, frameon=True, facecolor='white', edgecolor='black', fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{filename}.png")
        save_path_pdf = os.path.join(self.output_dir, f"{filename}.pdf")
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path_pdf)
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
        save_path_pdf = os.path.join(self.output_dir, f"{filename}_{x_var_key}_vs_{y_var_key}.pdf")
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path_pdf)
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
        save_path_pdf = os.path.join(self.output_dir, f"Efficiency_vs_{x_var_key}.pdf")
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path_pdf)
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

    def plot_profile(self, datasets, x_var_key, y_var_key, filename, mode='mean', title="", gen_n=1):
        x_conf = PLOT_VARS[x_var_key]
        y_conf = PLOT_VARS[y_var_key]
        
        fig, ax = plt.subplots(figsize=(12, 12  ))
        default_colors = ["tab:olive", "tab:cyan", "darkorchid" , "darkorange", "deeppink"]

        bin_edges = np.linspace(x_conf['range'][0], x_conf['range'][1], x_conf['bins'] + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        for i, ds in enumerate(datasets):
            color = ds.get('color', default_colors[i % len(default_colors)])
            if y_var_key == "n_clusters":
                x_vals, y_vals = self._get_ncluster_values(ds, x_var_key, gen_n=gen_n)
            else:
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
        ax.grid(linestyle=":")
        ax.legend(title=title, fontsize=15)
        
        save_path = os.path.join(self.output_dir, f"{filename}_{mode}.png")
        save_path_pdf = os.path.join(self.output_dir, f"{filename}_{mode}.pdf")
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path_pdf, dpi=300)
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
        
                ax.hist(slice_data, bins=var_conf['bins'], range=var_conf['range'], 
                        color=color, label=f"{ds['label']}", 
                        histtype='step', linewidth=2.5)
                ax.hist(slice_data, bins=var_conf['bins'], range=var_conf['range'], 
                        color=color, histtype='stepfilled', alpha=0.2)


            # Styling to match the Profile Plots
            bin_label = f"{low:.1f} < {bin_conf['label']} < {high:.1f}"
            hep.cms.label("Preliminary", data=True, rlabel=f"{self.args.particles} {self.args.pileup}", ax=ax)
            
            ax.set_xlabel(var_conf['label'])
            ax.set_ylabel("Counts")
            ax.legend(title=bin_label, fontsize=18, loc='upper left', title_fontsize=18)
            ax.grid(linestyle=":", alpha=0.6)
            
            # Save with the bin index in the filename
            save_name = f"{filename}_{var_key}_in_{binning_var_key}_bin{j}.png"
            save_path = os.path.join(self.output_dir, "bin_distributions", save_name)
            save_path_pdf = os.path.join(self.output_dir, "bin_distributions", f"{filename}_{var_key}_in_{binning_var_key}_bin{j}.pdf")
            
            # Ensure the sub-directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
            plt.savefig(save_path, dpi=300)
            plt.savefig(save_path_pdf)
            plt.close()
            print(f"--- Bin Distribution Saved: {save_path}")

    
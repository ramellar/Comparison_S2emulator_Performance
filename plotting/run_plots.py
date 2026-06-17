import matplotlib.pyplot as plt
from configs.config import PLOT_VARS, EMU_CONFIG
import matplotlib.colors as colors
import mplhep as hep
import numpy as np
import awkward as ak
import os
from scipy.stats import binned_statistic
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt


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
        use_abs = var_key.startswith("abs_")
        
        if var_key.endswith("_gen") or "_eff" in var_key:
            branch = PLOT_VARS[var_key]["branch"]
            vals = self._extract_array(ds.get('gen', ds.get('total_gen')), branch)
            return np.abs(vals) if use_abs else vals
    
        if "_response" in var_key:
            base_var = var_key.replace("_response", "")
            cl_vals = self._extract_array(ds['data'], base_var)
            gen_vals = self._extract_array(ds['gen'], base_var)
            
            if base_var == "pt":
                return np.divide(cl_vals, gen_vals, out=np.zeros_like(cl_vals), where=gen_vals!=0)
            
            res = cl_vals - gen_vals
            
            if base_var == "phi":
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
                #default_colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3", "#1b9e77", "#d95f02"]
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
        cmap = plt.cm.RdPu.copy()
        cmap.set_under('white')
#        h = ax.hist2d(x_vals, y_vals, 
#                      bins=[x_conf['bins'], y_conf['bins']], 
#                      range=[x_conf['range'], y_conf['range']],
#                      cmap=cmap,
#                      norm=LogNorm(vmin=1))
        
        h = ax.hist2d(x_vals, y_vals, 
                      bins=[x_conf['bins'], y_conf['bins']], 
                      range=[x_conf['range'], y_conf['range']],
                      cmap=cmap
                      )
        
        fig.colorbar(h[3], ax=ax, label='Counts')
        tag = dataset['label'].replace("Tri ", "").replace(" ", "_").replace("p", ".")
        ax.legend(title=title, fontsize=16)
        ax.grid(linestyle=":")
        hep.cms.label("Preliminary", data=True, rlabel=f"{tag}-{self.args.particles}-{self.args.pileup}", ax=ax)

        ax.set_xlabel(x_conf['label'])
        ax.set_ylabel(y_conf['label'])
        
        plt.tight_layout()

        save_dir = os.path.join(self.output_dir, "2D_distributions")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{filename}_{x_var_key}_vs_{y_var_key}.png")
        save_path_pdf = os.path.join(save_dir, f"{filename}_{x_var_key}_vs_{y_var_key}.pdf")
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


#        for i, ds in enumerate(datasets):
#            color = ds.get('color', default_colors[i % len(default_colors)])
#
#            # Efficiency = (Gen particles that were matched) / (All Gen particles)
#            matched_gen_x = self._get_values(ds, f"{x_var_key}_gen")
#            total_gen_x = self._extract_array(ds['total_gen'], x_var_key)
#
#            h_matched, _ = np.histogram(matched_gen_x, bins=bin_edges)
#            h_total, _ = np.histogram(total_gen_x, bins=bin_edges)
#            
#            eff = np.divide(h_matched, h_total, out=np.zeros_like(h_matched, dtype=float), where=h_total!=0)
#            
#            # Simple error calculation (binomial)
#            err = np.sqrt(eff * (1 - eff) / h_total, out=np.zeros_like(eff), where=h_total!=0)
#
#            ax.errorbar(bin_centers, eff, xerr=(bin_edges[1]-bin_edges[0])/2, yerr=err, label=ds['label'], fmt='o', markersize=6, color=color)
            
            
        for i, ds in enumerate(datasets):
            color = ds.get('color', default_colors[i % len(default_colors)])

            matched_gen = ds['gen']
            total_gen = ds['total_gen']

            # Apply the same gen pt cut to numerator and denominator
            if hasattr(self.args, "gen_pt_cut") and self.args.gen_pt_cut is not None:
                matched_gen = matched_gen[matched_gen["pt"] >= self.args.gen_pt_cut]
                total_gen = total_gen[total_gen["pt"] >= self.args.gen_pt_cut]
        
            # Efficiency = (Gen particles that were matched) / (All Gen particles)
            matched_gen_x = self._extract_array(matched_gen, x_var_key)
            total_gen_x = self._extract_array(total_gen, x_var_key)
        
            h_matched, _ = np.histogram(matched_gen_x, bins=bin_edges)
            h_total, _ = np.histogram(total_gen_x, bins=bin_edges)

            eff = np.divide(h_matched, h_total, out=np.zeros_like(h_matched, dtype=float), where=h_total != 0)

            # Simple error calculation (binomial)
            err = np.sqrt( eff * (1 - eff) / h_total, out=np.zeros_like(eff), where=h_total != 0)

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




    #def effrms(self, x, c=0.68):
    #    """ Computes half-width of the smallest interval containing c% of the distribution. """
    #    # if len(x) < 5: return 0 # Need enough points to find an interval
    #    x_sorted = np.sort(x)
    #    m = int(c * len(x_sorted))
    #    # Find the width of all intervals containing 'm' points
    #    widths = x_sorted[m:] - x_sorted[:-m]
    #    return np.min(widths) / 2.0
    
    def effrms(self, x, c=0.68):
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        if len(x) < 2:
            return np.nan
        x_sorted = np.sort(x)
        m = int(np.ceil(c * len(x_sorted)))
        if m < 1 or m >= len(x_sorted):
            return np.nan
        widths = x_sorted[m:] - x_sorted[:-m]
        if len(widths) == 0:
            return np.nan
        return np.min(widths) / 2.0




    def plot_profile(self, datasets, x_var_key, y_var_key, filename, mode='mean', title="", gen_n=1, save_dir=None):
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
        
        profile_dir = save_dir if save_dir is not None else os.path.join(self.output_dir, "profile_distributions")
        os.makedirs(profile_dir, exist_ok=True)
        
        save_path     = os.path.join(profile_dir, f"{filename}_{mode}.png")
        save_path_pdf = os.path.join(profile_dir, f"{filename}_{mode}.pdf")
        
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path_pdf, dpi=300)
        plt.close()
        print(f"--- Plot Saved: {save_path}")




    def plot_distributions_per_bin(self, datasets, var_key, binning_var_key, filename,  title_="", combined=True):
        """
        Shows the distribution of var_key (e.g., pt_response) for each bin of binning_var_key.
        With combined=True (default) all bins are laid out in a single figure saved as one PNG/PDF.
        """
        var_conf = PLOT_VARS[var_key]
        bin_conf = PLOT_VARS[binning_var_key]

        bin_edges = np.linspace(bin_conf['range'][0], bin_conf['range'][1], bin_conf['bins'] + 1)
        n_bins = len(bin_edges) - 1
        
        default_colors = ["tab:olive", "tab:cyan", "darkorchid", "darkorange", "deeppink"]

        tag = f"_{self.args.tag}" if self.args.tag is not None else ""
        base_name = f"{filename}_{var_key}_in_{binning_var_key}{tag}"

        if combined:
            n_rows = 2
            n_cols = int(np.ceil(n_bins / n_rows))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
            axes = np.array(axes).flatten()

            for j in range(n_bins):
                low, high = bin_edges[j], bin_edges[j + 1]
                ax = axes[j]
                
                for i, ds in enumerate(datasets):
                    color = ds.get('color', default_colors[i % len(default_colors)])
                    vals_to_plot = self._get_values(ds, var_key)
                    bin_vals     = self._get_values(ds, binning_var_key)
                    
                    if self.args.gen_pt_cut > 0:
                        pt_gen_vals = self._get_values(ds, "pt_gen")
                        gen_mask = pt_gen_vals > self.args.gen_pt_cut

                        vals_to_plot = vals_to_plot[gen_mask]
                        bin_vals = bin_vals[gen_mask]
                    
                    mask         = (bin_vals >= low) & (bin_vals < high)
                    slice_data   = vals_to_plot[mask]

                    ax.hist(slice_data, bins=var_conf['bins'], range=var_conf['range'],
                            color=color, label=ds['label'], histtype='step', linewidth=2.5)
                    ax.hist(slice_data, bins=var_conf['bins'], range=var_conf['range'],
                            color=color, histtype='stepfilled', alpha=0.2)

                bin_label = f"{low:.2f} < {bin_conf['label']} < {high:.2f}"
                ax.set_xlabel(var_conf['label'], fontsize=13)
                ax.set_ylabel("Counts", fontsize=13)
                ax.legend(title=bin_label + title_, fontsize=11, loc='upper left', title_fontsize=11)
                ax.grid(linestyle=":", alpha=0.6)

            # Hide unused subplots
            for j in range(n_bins, len(axes)):
                axes[j].set_visible(False)
                
            fig.suptitle(f"{var_conf['label']} per {bin_conf['label']} bin  |  {self.args.particles} {self.args.pileup}",
                         fontsize=16, y=1.01)
            fig.tight_layout()

            save_dir = os.path.join(self.output_dir, "bin_distributions")                
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{base_name}_bin{j}.png")
            save_path_pdf = os.path.join(save_dir, f"{base_name}_bin{j}.pdf") 
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.savefig(save_path_pdf, bbox_inches='tight')
            plt.close()
            print(f"--- Bin Distribution Saved: {save_path}")

        else:
            for j in range(n_bins):
                low, high = bin_edges[j], bin_edges[j + 1]
                fig, ax = plt.subplots(figsize=(10, 10))

                for i, ds in enumerate(datasets):
                    color = ds.get('color', default_colors[i % len(default_colors)])
                    vals_to_plot = self._get_values(ds, var_key)
                    bin_vals     = self._get_values(ds, binning_var_key)
                    
                    if self.args.gen_pt_cut > 0:
                        pt_gen_vals = self._get_values(ds, "pt_gen")
                        gen_mask = pt_gen_vals > self.args.gen_pt_cut
                    
                        vals_to_plot = vals_to_plot[gen_mask]
                        bin_vals = bin_vals[gen_mask]

                    mask         = (bin_vals >= low) & (bin_vals < high)
                    slice_data   = vals_to_plot[mask]

                    ax.hist(slice_data, bins=var_conf['bins'], range=var_conf['range'],
                            color=color, label=ds['label'], histtype='step', linewidth=2.5)
                    ax.hist(slice_data, bins=var_conf['bins'], range=var_conf['range'],

                            color=color, histtype='stepfilled', alpha=0.2)

                bin_label = f"{low:.1f} < {bin_conf['label']} < {high:.1f}"
                hep.cms.label("Preliminary", data=True, rlabel=f"{self.args.particles} {self.args.pileup}", ax=ax)
                ax.set_xlabel(var_conf['label'])
                ax.set_ylabel("Counts")
                ax.legend(title=bin_label, fontsize=18, loc='upper left', title_fontsize=18)
                ax.grid(linestyle=":", alpha=0.6)

                save_dir = os.path.join(self.output_dir, "bin_distributions")                
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{base_name}_bin{j}.png")
                save_path_pdf = os.path.join(save_dir, f"{base_name}_bin{j}.pdf")
                plt.savefig(save_path, dpi=300)
                plt.savefig(save_path_pdf)
                plt.close()
                print(f"--- Bin Distribution Saved: {save_path}")
                
        
    def plot_multiplicity_per_decaymode(self, datasets, decaymode_key="gen_decayMode", filename="Multiplicity_per_decaymode", title="", make_raw=True,):
        valid_dm = np.array([0, 1, 4, 5])
        bins = np.arange(-0.5, 6.5, 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        valid_mask = np.isin(bin_centers, valid_dm)

        default_colors = ["tab:olive", "tab:cyan", "darkorchid", "darkorange", "deeppink"]

        save_dir = os.path.join(self.output_dir, "decaymode_distributions")
        os.makedirs(save_dir, exist_ok=True)

        # ============================================================
        # Plot 1: normalized distribution + TRatio
        # ============================================================

        fig, (ax, rax) = plt.subplots(
            2, 1,
            figsize=(10, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )

        # riferimento = ultimo dataset
        ref_dm = ak.to_numpy(ak.flatten(datasets[-1]["gen"][decaymode_key], axis=None))
        ref_counts, _ = np.histogram(ref_dm, bins=bins)
        ref_norm = np.divide(ref_counts, np.sum(ref_counts), out=np.zeros_like(ref_counts, dtype=float), where=np.sum(ref_counts) > 0,)

        for i, ds in enumerate(datasets):
            color = ds.get("color", default_colors[i % len(default_colors)])

            dm_vals = ak.to_numpy(ak.flatten(ds["gen"][decaymode_key], axis=None))

            counts, _ = np.histogram(dm_vals, bins=bins)
            norm = np.divide(
                counts,
                np.sum(counts),
                out=np.zeros_like(counts, dtype=float),
                where=np.sum(counts) > 0,
            )

            weights = np.ones_like(dm_vals, dtype=float) / len(dm_vals)

            ax.hist(
                dm_vals,
                bins=bins,
                histtype="stepfilled",
                alpha=0.22,
                color=color,
                weights=weights,
            )

            ax.hist(
                dm_vals,
                bins=bins,
                histtype="step",
                linewidth=2.5,
                color=color,
                label=ds["label"],
                weights=weights,
            )

            ratio = np.divide(
                norm,
                ref_norm,
                out=np.zeros_like(norm, dtype=float),
                where=ref_norm > 0,
            )

            rax.plot(
                bin_centers[valid_mask],
                ratio[valid_mask],
                marker="o",
                linestyle="none",
                markersize=6,
                color=color,
                label=ds["label"],
            )

        hep.cms.label(
            "Preliminary",
            data=True,
            rlabel=f"{self.args.particles}-{self.args.pileup}",
            ax=ax,
        )

        ax.set_ylabel("Normalized events", fontsize=18)
        ax.legend(title=title, fontsize=13)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        rax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        rax.set_xlabel("gen decayMode", fontsize=18)
        rax.set_ylabel("Ratio / Ref", fontsize=14)
        rax.set_xticks(valid_dm)
        rax.set_xlim(-0.5, 5.5)
        rax.set_ylim(0.8, 1.2)
        rax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300)
        plt.savefig(os.path.join(save_dir, f"{filename}.pdf"), dpi=300)
        plt.close()

        print(f"--- Normalized multiplicity per DM saved: {os.path.join(save_dir, filename)}.png")

        # ============================================================
        # Plot 2: raw distribution, no normalization, no TRatio
        # ============================================================

        if make_raw:
            fig_raw, ax_raw = plt.subplots(figsize=(10, 7))

            for i, ds in enumerate(datasets):
                color = ds.get("color", default_colors[i % len(default_colors)])

                dm_vals = ak.to_numpy(
                    ak.flatten(ds["gen"][decaymode_key], axis=None)
                )

                ax_raw.hist(dm_vals, bins=bins, histtype="stepfilled", alpha=0.22, color=color,)

                ax_raw.hist(dm_vals, bins=bins, histtype="step", linewidth=2.5, color=color, label=ds["label"],)

            hep.cms.label(
                "Preliminary",
                data=True,
                rlabel=f"{self.args.particles}-{self.args.pileup}",
                ax=ax_raw,
            )

            ax_raw.set_xlabel("gen decayMode", fontsize=18)
            ax_raw.set_ylabel("Events", fontsize=18)
            ax_raw.set_xticks(valid_dm)
            ax_raw.set_xlim(-0.5, 5.5)
            ax_raw.legend(title=title, fontsize=13)
            ax_raw.grid(axis="y", linestyle=":", alpha=0.6)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{filename}_raw.png"), dpi=300)
            plt.savefig(os.path.join(save_dir, f"{filename}_raw.pdf"), dpi=300)
            plt.close()

            print(f"--- Raw multiplicity per DM saved: {os.path.join(save_dir, filename)}_raw.png")
        
        
    def plot_distributions_per_decaymode(self, datasets, var_key, decaymode_key, filename):
        var_conf = PLOT_VARS[var_key]

        decay_modes = [0, 1, 4, 5]
        default_colors = ["tab:olive", "tab:cyan", "darkorchid", "darkorange", "deeppink"]

        ncols = 2
        nrows = 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for j, dm in enumerate(decay_modes):
            ax = axes[j]

            for i, ds in enumerate(datasets):
                color = ds.get("color", default_colors[i % len(default_colors)])

                if var_key == "abs_eta":
                    vals_to_plot = np.abs(self._get_values(ds, "eta"))
                else:
                    vals_to_plot = self._get_values(ds, var_key)

                dm_vals = ak.to_numpy(ak.flatten(ds["gen"][decaymode_key], axis=None))

                vals_to_plot = ak.to_numpy(ak.flatten(vals_to_plot, axis=None))
                dm_vals = ak.to_numpy(ak.flatten(dm_vals, axis=None))

                mask = dm_vals == dm
                slice_data = vals_to_plot[mask]
                slice_data = slice_data[np.isfinite(slice_data)]

                if len(slice_data) == 0:
                    continue

                weights = np.ones_like(slice_data, dtype=float) / len(slice_data)

                ax.hist(
                    slice_data,
                    bins=var_conf["bins"],
                    range=var_conf["range"],
                    histtype="stepfilled",
                    alpha=0.22,
                    color=color,
                    weights=weights,
                )

                ax.hist(
                    slice_data,
                    bins=var_conf["bins"],
                    range=var_conf["range"],
                    histtype="step",
                    linewidth=2.2,
                    color=color,
                    label=ds["label"],
                    weights=weights,
                )

            ax.set_title(f"DecayMode = {dm}", fontsize=18)
            ax.grid(linestyle=":", alpha=0.6)

        fig.supxlabel(var_conf["label"], fontsize=22)
        fig.supylabel("Normalized entries", fontsize=22)

        handles, labels = axes[0].get_legend_handles_labels()
        if labels:
            fig.legend(handles, labels, loc="upper right", fontsize=14)

        hep.cms.label(
            "Preliminary",
            data=True,
            rlabel=f"{self.args.particles} {self.args.pileup}",
            ax=axes[0],
            loc=0,
            fontsize=16
        )

        save_dir = os.path.join(self.output_dir, "decaymode_distributions")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{filename}_{var_key}_per_{decaymode_key}.png")
        save_path_pdf = os.path.join(save_dir, f"{filename}_{var_key}_per_{decaymode_key}.pdf")

        plt.tight_layout(rect=[0, 0, 0.92, 0.90])
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path_pdf, dpi=300)
        plt.close()

        print(f"--- DecayMode Distribution Saved: {save_path}")
        
               
        
    
    # Response plot per diecaymode
    def plot_response_distribution_per_decaymode(self, datasets, var_key, decaymode_key, filename):
        conf = PLOT_VARS[f"{var_key}_response"]

        decay_modes = [0, 1, 4, 5]
        default_colors = ["tab:olive", "tab:cyan", "darkorchid", "darkorange", "deeppink"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for j, dm in enumerate(decay_modes):
            ax = axes[j]

            for i, ds in enumerate(datasets):
                color = ds.get("color", default_colors[i % len(default_colors)])

                gen_vals = self._extract_array(ds["gen"], var_key)
                #cl_vals = self._extract_array(ds["cluster"], var_key)
                cl_vals = self._extract_array(ds.get("cluster", ds.get("data")), var_key)
                dm_vals = self._extract_array(ds["gen"], decaymode_key)

                # Prima seleziona per decayMode sugli awkward array,
                # così gen/cluster/dm restano allineati.
                mask = dm_vals == dm

                gen_sel = ak.to_numpy(ak.flatten(gen_vals[mask], axis=None))
                cl_sel = ak.to_numpy(ak.flatten(cl_vals[mask], axis=None))

                if len(gen_sel) == 0 or len(cl_sel) == 0:
                    continue

                # Safety: se dopo flatten le lunghezze non coincidono, salta il dataset.
                if len(gen_sel) != len(cl_sel):
                    print(
                        f"[WARNING] Skipping {ds['label']} DM={dm} var={var_key}: "
                        f"len(gen_sel)={len(gen_sel)} != len(cl_sel)={len(cl_sel)}"
                    )
                    continue

                # Definizione response:
                # pt  -> cluster / gen
                # eta/phi -> cluster - gen
                if var_key == "pt":
                    slice_data = np.divide(
                        cl_sel,
                        gen_sel,
                        out=np.zeros_like(cl_sel, dtype=float),
                        where=gen_sel != 0,
                    )
                else:
                    slice_data = cl_sel - gen_sel

                slice_data = slice_data[np.isfinite(slice_data)]

                if len(slice_data) == 0:
                    continue
                
                weights = np.ones_like(slice_data, dtype=float) / len(slice_data)

                ax.hist(
                    slice_data,
                    bins=conf["bins"],
                    range=conf["range"],
                    histtype="stepfilled",
                    alpha=0.22,
                    color=color,
                    weights=weights,
                )
                ax.hist(
                    slice_data,
                    bins=conf["bins"],
                    range=conf["range"],
                    histtype="step",
                    linewidth=2.2,
                    color=color,
                    label=ds["label"],
                    weights=weights,
                )

            ax.set_title(f"DecayMode = {dm}", fontsize=18)
            ax.grid(linestyle=":", alpha=0.6)

        fig.supxlabel(conf["label"], fontsize=22)
        fig.supylabel("Normalized counts", fontsize=22)

        handles, labels = axes[0].get_legend_handles_labels()
        if labels:
            fig.legend(handles, labels, loc="upper right", fontsize=14)

        hep.cms.label(
            "Preliminary",
            data=True,
            rlabel=f"{self.args.pileup} {self.args.particles}",
            ax=axes[0],
            loc=0,
            fontsize=16,
        )

        save_dir = os.path.join(self.output_dir, "decaymode_distributions")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{filename}_{var_key}_response_per_{decaymode_key}.png")
        save_path_pdf = os.path.join(save_dir, f"{filename}_{var_key}_response_per_{decaymode_key}.pdf")

        plt.tight_layout(rect=[0, 0, 0.92, 0.90])
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path_pdf, dpi=300)
        plt.close()

        print(f"--- DecayMode Response Distribution Saved: {save_path}")
        
        
    
    
    # Response, resolution and efficiency plots for different decaymodes
    def plot_profile_per_decaymode(self, datasets, x_var_key, y_var_key, filename, mode='mean', title=""):
        decay_modes = [0, 1, 4, 5]
        save_dir = os.path.join(self.output_dir, "decaymode_distributions", "profile_distributions_per_decaymode")
        os.makedirs(save_dir, exist_ok=True)
    
        x_conf = PLOT_VARS[x_var_key]
        y_conf = PLOT_VARS[y_var_key]
    
        default_colors = ["tab:olive", "tab:cyan", "darkorchid", "darkorange", "deeppink"]
        bin_edges  = np.linspace(x_conf['range'][0], x_conf['range'][1], x_conf['bins'] + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True)
        axes = axes.flatten()
    
        ylabel = ""
    
        for j, dm in enumerate(decay_modes):
            ax = axes[j]
    
            # costruisci dataset mascherati per questo DM
            dm_datasets = []
            for ds in datasets:
                gen    = ds['gen']
                dm_raw = gen['gen_decayMode']
                mask   = ak.to_numpy(ak.any(dm_raw == dm, axis=-1))
    
                ds_dm          = dict(ds)
                ds_dm['gen']   = gen[mask]
                ds_dm['data']  = ds.get('cluster', ds.get('data'))[mask]
                if 'cluster' in ds:
                    ds_dm['cluster'] = ds['cluster'][mask]
                dm_datasets.append(ds_dm)
    
            for i, ds in enumerate(dm_datasets):
                color  = ds.get('color', default_colors[i % len(default_colors)])
                x_vals = self._get_values(ds, x_var_key)
                y_vals = self._get_values(ds, y_var_key)
    
                counts, _, _ = binned_statistic(x_vals, y_vals, statistic='count', bins=bin_edges)
    
                if mode == 'mean':
                    stat, _, _ = binned_statistic(x_vals, y_vals, statistic='mean', bins=bin_edges)
                    stds, _, _ = binned_statistic(x_vals, y_vals, statistic=lambda x: np.std(x), bins=bin_edges)
                    y_err  = np.divide(stds, np.sqrt(counts), out=np.zeros_like(stds), where=counts > 0)
                    ylabel = f"<{y_conf['label']}>"
    
                elif mode == 'resolution':
                    means, _, _ = binned_statistic(x_vals, y_vals, statistic='mean', bins=bin_edges)
                    stds,  _, _ = binned_statistic(x_vals, y_vals, statistic=lambda x: np.std(x), bins=bin_edges)
                    if 'pt' in y_var_key:
                        stat   = np.divide(stds, means, out=np.zeros_like(stds), where=means != 0)
                        ylabel = r"$\sigma_{cluster} / \mu_{cluster}$"
                    else:
                        stat   = stds
                        ylabel = r"$\sigma_{cluster}$"
                    y_err = np.divide(stat, np.sqrt(2 * counts - 2), out=np.zeros_like(stat), where=counts > 1)
    
                elif mode == 'rms':
                    means,    _, _ = binned_statistic(x_vals, y_vals, statistic='mean', bins=bin_edges)
                    eff_stds, _, _ = binned_statistic(x_vals, y_vals,
                                                      statistic=lambda x: self.effrms(x),
                                                      bins=bin_edges)
                    if 'pt' in y_var_key:
                        stat   = np.divide(eff_stds, means, out=np.zeros_like(eff_stds), where=means != 0)
                        ylabel = r"$\sigma^{eff-RMS}_{cluster} / \mu_{cluster}$"
                    else:
                        stat   = eff_stds
                        ylabel = r"$\sigma^{eff-RMS}_{cluster}$"
                    y_err = np.divide(stat, np.sqrt(2 * counts - 2), out=np.zeros_like(stat), where=counts > 1)
    
                mask_plot = ~np.isnan(stat) & (counts > 2)
                if np.any(mask_plot):
                    ax.errorbar(
                        bin_centers[mask_plot], stat[mask_plot],
                        yerr=y_err[mask_plot],
                        xerr=(bin_edges[1] - bin_edges[0]) / 2,
                        label=ds['label'], color=color,
                        fmt='o', markersize=8,
                    )
    
            ax.set_title(f"Decay Mode {dm}", fontsize=16)
            ax.grid(linestyle=":")
            ax.legend(title=f"{title}" if title else "", fontsize=11)
    
        # assi condivisi
        for ax in axes:
            ax.set_xlabel(x_conf['label'], fontsize=13)
        for ax in axes:
            ax.set_ylabel(ylabel, fontsize=13)
    
        hep.cms.label("Preliminary", data=True,
                      rlabel=f"{self.args.particles}-{self.args.pileup}",
                      ax=axes[0])
    
        fig.tight_layout()
    
        base = f"{filename}_{mode}"
        plt.savefig(os.path.join(save_dir, f"{base}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f"{base}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"--- Profile per DM saved: {os.path.join(save_dir, base)}.png")
        
        
        
        
    def plot_response_in_eta_bin_per_decaymode(self, datasets, dm, eta_low, eta_high,
                                            var_key="pt_response", title=""):
        """
        Istogramma di var_key per un singolo decay mode e un bin di |eta_gen|.
        Utile per investigare anomalie nei resolution plots.
        """
        conf     = PLOT_VARS[var_key]
        eta_conf = PLOT_VARS["abs_eta_gen"]

        default_colors = ["tab:olive", "tab:cyan", "darkorchid", "darkorange", "deeppink"]

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, ds in enumerate(datasets):
            color = ds.get("color", default_colors[i % len(default_colors)])

            # decay mode mask (a livello evento)
            dm_raw   = ds['gen']['gen_decayMode']
            dm_mask  = ak.to_numpy(ak.any(dm_raw == dm, axis=-1))

            ds_dm          = dict(ds)
            ds_dm['gen']   = ds['gen'][dm_mask]
            ds_dm['data']  = ds.get('cluster', ds.get('data'))[dm_mask]

            # eta mask (a livello particella, dopo flatten)
            eta_vals      = self._get_values(ds_dm, "abs_eta_gen")  # già flatten 1D
            response_vals = self._get_values(ds_dm, var_key)        # già flatten 1D

            eta_mask   = (eta_vals >= eta_low) & (eta_vals < eta_high)
            slice_data = response_vals[eta_mask]
            slice_data = slice_data[np.isfinite(slice_data)]

            outliers = slice_data[slice_data > 2.5]

            if len(outliers) > 0:
                print(
                    f"{ds['label']} DM={dm} eta=[{eta_low},{eta_high}] "
                    f"values > 2.5:"
                )
                print(outliers)
            else:
                print(
                    f"{ds['label']} DM={dm} eta=[{eta_low},{eta_high}] : no events"
                )

            if len(slice_data) == 0:
                print(f"[WARNING] {ds['label']} DM={dm} eta=[{eta_low},{eta_high}]: no entries")
                continue

            weights = np.ones_like(slice_data) / len(slice_data)

            ax.hist(slice_data, bins=conf["bins"], range=conf["range"],
                    histtype="stepfilled", alpha=0.22, color=color, weights=weights)
            ax.hist(slice_data, bins=conf["bins"], range=conf["range"],
                    histtype="step", linewidth=2.2, color=color,
                    label=f"{ds['label']} (N={len(slice_data)})", weights=weights)

        hep.cms.label("Preliminary", data=True,
                      rlabel=f"{self.args.particles}-{self.args.pileup}", ax=ax)

        eta_label = rf"${eta_low:.2f} < |\eta^{{gen}}| < {eta_high:.2f}$,  DM={dm}"
        ax.set_xlabel(conf["label"], fontsize=16)
        ax.set_ylabel("Normalized entries", fontsize=16)
        ax.set_title(eta_label, fontsize=16)
        ax.grid(linestyle=":", alpha=0.6)
        ax.legend(title=title, fontsize=12)

        save_dir = os.path.join(self.output_dir, "decaymode_distributions", "anomaly_investigation")
        os.makedirs(save_dir, exist_ok=True)

        tag = f"{var_key}_DM{dm}_eta{eta_low:.2f}-{eta_high:.2f}".replace(".", "p")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{tag}.png"), dpi=300)
        plt.savefig(os.path.join(save_dir, f"{tag}.pdf"), dpi=300)
        plt.close()
        print(f"--- Anomaly plot saved: {tag}.png")
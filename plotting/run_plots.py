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
        var_conf = PLOT_VARS[var_key]
        bin_conf = PLOT_VARS[binning_var_key]
        
        if "eta" in binning_var_key.lower():
            bin_edges = np.array([1.5, 1.8, 2.1, 2.4, 2.7, 3.0])
        else:
            bin_edges = np.linspace(
            bin_conf['range'][0],
            bin_conf['range'][1],
            bin_conf['bins'] + 1
            )


        default_colors = ["tab:olive", "tab:cyan", "darkorchid", "darkorange", "deeppink"]

        n_bins = len(bin_edges) - 1
        ncols = 5
        nrows = int(np.ceil(n_bins / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 4.5 * nrows),
            sharex=True,
            sharey=True
        )

        axes = np.array(axes).reshape(-1)

        for j in range(n_bins):
            low, high = bin_edges[j], bin_edges[j + 1]
            ax = axes[j]

            for i, ds in enumerate(datasets):
                color = ds.get('color', default_colors[i % len(default_colors)])

                vals_to_plot = self._get_values(ds, var_key)
                bin_vals = self._get_values(ds, binning_var_key)
                
                if "eta" in binning_var_key.lower():
                    bin_vals = np.abs(bin_vals)

                mask = (bin_vals >= low) & (bin_vals < high)
                slice_data = vals_to_plot[mask]

                ax.hist(
                    slice_data,
                    bins=var_conf['bins'],
                    range=var_conf['range'],
                    color=color,
                    histtype='step',
                    linewidth=2.0,
                    label=ds['label']
                )

                ax.hist(
                    slice_data,
                    bins=var_conf['bins'],
                    range=var_conf['range'],
                    color=color,
                    histtype='stepfilled',
                    alpha=0.2
                )

            ax.set_title(f"{low:.1f} < {bin_conf['label']} < {high:.1f}", fontsize=14)
            ax.grid(linestyle=":", alpha=0.6)

        for ax in axes[n_bins:]:
            ax.axis("off")

        fig.supxlabel(var_conf['label'], fontsize=22)
        fig.supylabel("Counts", fontsize=22)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=16)

        #hep.cms.label("Preliminary", data=True, rlabel=f"{self.args.particles} {self.args.pileup}", ax=axes[0])
        hep.cms.label("Preliminary", data=True, rlabel=f"{self.args.particles} {self.args.pileup}", ax=axes[0], pad=30)

        save_dir = os.path.join(self.output_dir, "bin_distributions")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            save_dir,
            f"{filename}_{var_key}_in_{binning_var_key}_all_bins.png"
        )
        save_path_pdf = os.path.join(
            save_dir,
            f"{filename}_{var_key}_in_{binning_var_key}_all_bins.pdf"
        )

        plt.tight_layout(rect=[0, 0, 0.92, 0.95])
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path_pdf)
        plt.close()

        print(f"--- All-bin Distribution Saved: {save_path}")
        
        
    def plot_decay_modes(self, datasets, filename="DecayMode_counts", title="", normalize=False):
        fig, ax = plt.subplots(figsize=(10, 8))
    
        default_colors = ["tab:olive", "tab:cyan", "darkorchid", "darkorange", "deeppink"]
    
        # bin centrati su 0, 1, 2, 3, 4, 5
        bins = np.arange(-0.5, 6.5, 1)
    
        for i, ds in enumerate(datasets):
            color = ds.get("color", default_colors[i % len(default_colors)])
    
            decay_modes = self._get_values(ds, "gen_decayMode")
            decay_modes = ak.to_numpy(ak.flatten(decay_modes, axis=None))
            
            ax.hist(decay_modes, bins=bins, histtype="stepfilled", alpha=0.22, color=color, density=normalize)
            ax.hist(decay_modes, bins=bins, histtype="step", linewidth=2.5, color=color, label=ds["label"], density=normalize)
        
        
        ax.set_xlabel("gen decayMode", fontsize=22)
        if normalize:
            ax.set_ylabel("Normalized events", fontsize=22)
        else:
            ax.set_ylabel("Number of events", fontsize=22)
            
        ax.set_title(title, fontsize=24)
        ax.set_xticks([0, 1, 4, 5])
        ax.legend(fontsize=16)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
    
        save_path = os.path.join(self.output_dir, f"{filename}.png")
        save_path_pdf = os.path.join(self.output_dir, f"{filename}.pdf")
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path_pdf)
        plt.close()
        
        
    def plot_decay_modes_ratio(self, datasets, filename="DecayMode_counts_ratio", title=""):
        fig, (ax, rax) = plt.subplots(
            2, 1,
            figsize=(10, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
        )

        default_colors = ["tab:olive", "tab:cyan", "darkorchid", "darkorange", "deeppink"]
        bins = np.arange(-0.5, 6.5, 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # usa ultimo dataset come riferimento, es. Tri Ref
        ref_ds = datasets[-1]
        ref_dm = self._get_values(ref_ds, "gen_decayMode")
        ref_dm = ak.to_numpy(ak.flatten(ref_dm, axis=None))

        ref_counts, _ = np.histogram(ref_dm, bins=bins)
        ref_norm = ref_counts / np.sum(ref_counts)

        for i, ds in enumerate(datasets):
            color = ds.get("color", default_colors[i % len(default_colors)])

            decay_modes = self._get_values(ds, "gen_decayMode")
            decay_modes = ak.to_numpy(ak.flatten(decay_modes, axis=None))

            counts, _ = np.histogram(decay_modes, bins=bins)
            norm = counts / np.sum(counts)

            ax.hist(
                decay_modes,
                bins=bins,
                histtype="stepfilled",
                alpha=0.22,
                color=color,
                weights=np.ones_like(decay_modes) / len(decay_modes)
            )

            ax.hist(
                decay_modes,
                bins=bins,
                histtype="step",
                linewidth=2.5,
                color=color,
                label=ds["label"],
                weights=np.ones_like(decay_modes) / len(decay_modes)
            )

            ratio = np.divide(
                norm,
                ref_norm,
                out=np.zeros_like(norm, dtype=float),
                where=ref_norm > 0
            )

            valid_dm = np.array([0, 1, 4, 5])
            valid_mask = np.isin(bin_centers, valid_dm)

            rax.plot(
                bin_centers[valid_mask],
                ratio[valid_mask],
                marker="o",
                linestyle="none",
                markersize=6,
                color=color,
                label=ds["label"]
            )

        ax.set_ylabel("Normalized events", fontsize=20)
        ax.set_title(title, fontsize=22)
        ax.legend(fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        rax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        rax.set_xlabel("gen decayMode", fontsize=20)
        rax.set_ylabel("Ratio / Ref", fontsize=16)
        rax.set_xticks([0, 1, 4, 5])
        rax.grid(axis="y", linestyle=":", alpha=0.6)
        rax.set_xlim(-0.5, 5.5)
        rax.set_ylim(0.95, 1.05)

        save_path = os.path.join(self.output_dir, f"{filename}.png")
        save_path_pdf = os.path.join(self.output_dir, f"{filename}.pdf")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path_pdf)
        plt.close()

        print(f"--- Ratio Plot Saved: {save_path}")
        
        
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
                    vals_to_plot = abs(self._get_values(ds, "eta"))
                else:
                    vals_to_plot = self._get_values(ds, var_key)
    
                dm_vals = self._get_values(ds, decaymode_key)
    
                vals_to_plot = ak.to_numpy(ak.flatten(vals_to_plot, axis=None))
                dm_vals = ak.to_numpy(ak.flatten(dm_vals, axis=None))
    
                mask = (dm_vals == dm)
                slice_data = vals_to_plot[mask]
    
                ax.hist(
                    slice_data,
                    bins=var_conf["bins"],
                    range=var_conf["range"],
                    histtype="stepfilled",
                    alpha=0.22,
                    color=color,
                    density=True
                )
                ax.hist(
                    slice_data,
                    bins=var_conf["bins"],
                    range=var_conf["range"],
                    histtype="step",
                    linewidth=2.2,
                    color=color,
                    label=ds["label"],
                    density=True
                )
    
            ax.set_title(f"DecayMode = {dm}", fontsize=18)
            ax.grid(linestyle=":", alpha=0.6)
    
        fig.supxlabel(var_conf["label"], fontsize=22)
        fig.supylabel("Normalized counts", fontsize=22)
    
        handles, labels = axes[0].get_legend_handles_labels()
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
        plt.savefig(save_path_pdf)
        plt.close()
    
        print(f"--- DecayMode Distribution Saved: {save_path}")
        
        
        
    def plot_stat_per_decaymode(self, datasets, y_var_key, decaymode_key, filename, mode="response", title=""):
        y_conf = PLOT_VARS[y_var_key]

        decay_modes = [0, 1, 4, 5]
        default_colors = ["tab:olive", "tab:cyan", "darkorchid", "darkorange", "deeppink"]

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, ds in enumerate(datasets):
            color = ds.get("color", default_colors[i % len(default_colors)])

            y_vals = self._get_values(ds, y_var_key)
            dm_vals = self._get_values(ds, decaymode_key)

            y_vals = ak.to_numpy(ak.flatten(y_vals, axis=None))
            dm_vals = ak.to_numpy(ak.flatten(dm_vals, axis=None))

            stats = []
            errors = []

            for dm in decay_modes:
                mask_dm = (dm_vals == dm)
                slice_data = y_vals[mask_dm]

                n = len(slice_data)

                if n > 2:
                    mean = np.mean(slice_data)
                    std = np.std(slice_data)

                    if mode == "response":
                        stat = mean
                        err = std / np.sqrt(n)
                        ylabel = f"<{y_conf['label']}>"

                    elif mode == "resolution":
                        if "pt" in y_var_key:
                            stat = std / mean if mean != 0 else np.nan
                            ylabel = r"$\sigma_{cluster} / \mu_{cluster}$"
                        else:
                            stat = std
                            ylabel = r"$\sigma_{cluster}$"

                        err = stat / np.sqrt(2 * n - 2)

                    else:
                        raise ValueError("mode must be 'response' or 'resolution'")

                else:
                    stat = np.nan
                    err = np.nan

                stats.append(stat)
                errors.append(err)

            stats = np.array(stats)
            errors = np.array(errors)

            mask = ~np.isnan(stats)

            ax.errorbar(
                np.array(decay_modes)[mask],
                stats[mask],
                yerr=errors[mask],
                label=ds["label"],
                color=color,
                fmt="o",
                markersize=8
            )

        hep.cms.label(
            "Preliminary",
            data=True,
            rlabel=f"{self.args.particles}-{self.args.pileup}",
            ax=ax
        )

        ax.set_xlabel("gen decayMode")
        ax.set_ylabel(ylabel)
        ax.set_xticks(decay_modes)
        ax.grid(linestyle=":")
        ax.legend(title=title, fontsize=15)

        save_dir = os.path.join(self.output_dir, "decaymode_distributions")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{filename}_{y_var_key}_{mode}_per_{decaymode_key}.png")
        save_path_pdf = os.path.join(save_dir, f"{filename}_{y_var_key}_{mode}_per_{decaymode_key}.pdf")

        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path_pdf, dpi=300)
        plt.close()

        print(f"--- DecayMode {mode} Saved: {save_path}")
        
        
        
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
                cl_vals = self._extract_array(ds["cluster"], var_key)
                dm_vals = self._extract_array(ds["gen"], decaymode_key)

                gen_vals = ak.to_numpy(ak.flatten(gen_vals, axis=None))
                cl_vals = ak.to_numpy(ak.flatten(cl_vals, axis=None))
                dm_vals = ak.to_numpy(ak.flatten(dm_vals, axis=None))

                response = cl_vals / gen_vals

                mask = (dm_vals == dm)
                slice_data = response[mask]

                ax.hist(
                    slice_data,
                    bins=conf["bins"],
                    range=conf["range"],
                    histtype="stepfilled",
                    alpha=0.22,
                    color=color,
                    density=True
                )
                ax.hist(
                    slice_data,
                    bins=conf["bins"],
                    range=conf["range"],
                    histtype="step",
                    linewidth=2.2,
                    color=color,
                    label=ds["label"],
                    density=True
                )

            ax.set_title(f"DecayMode = {dm}", fontsize=18)
            ax.grid(linestyle=":", alpha=0.6)

        fig.supxlabel(conf["label"], fontsize=22)
        fig.supylabel("Normalized counts", fontsize=22)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=14)

        hep.cms.label(
            "Preliminary",
            data=True,
            rlabel=f"{self.args.pileup} {self.args.particles}",
            ax=axes[0],
            loc=0,
            fontsize=16
        )

        save_dir = os.path.join(self.output_dir, "decaymode_distributions")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{filename}_{var_key}_response_per_{decaymode_key}.png")
        save_path_pdf = os.path.join(save_dir, f"{filename}_{var_key}_response_per_{decaymode_key}.pdf")

        plt.tight_layout(rect=[0, 0, 0.92, 0.90])
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path_pdf)
        plt.close()

        print(f"--- DecayMode Response Distribution Saved: {save_path}")

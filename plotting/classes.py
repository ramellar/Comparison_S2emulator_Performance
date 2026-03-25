import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep
from matplotlib.patches import Rectangle

class EventProcessor:
    def __init__(self, events, gen_events=None):
        self.events = events
        self.gen_events = gen_events

    def get_flat_info(self, matched=False):
        if matched:
            event_info = ak.zip({
                "eta": self.events.eta,
                "phi": self.events.phi,
                "pt": self.events.pt,
                "delta_r": self.events.delta_r,
                "Ecalib": self.events.Ecalib,
            })
        else:
            event_info = ak.zip({
                "eta": getattr(self.events, "eta"),
                "phi": getattr(self.events,"phi"),
                "pt": getattr(self.events, "pt"),
            })
        
        flat = ak.flatten(event_info, axis=-1)
        return ak.flatten(flat, axis=-1) if matched else flat


class PerformancePlotter:
    def __init__(self, args):
        self.args = args
        self.legend_handles = []
        plt.style.use(mplhep.style.CMS)

    def _apply_cms_style(self, var):
        """Internal helper to apply common styling."""
        mplhep.cms.label('Preliminary', data=True, 
                         rlabel=f"{self.args.pileup} {self.args.particles}")
        if self.args.pileup == 'PU200' or var == "delta_r":
            plt.yscale('log')

    def plot_distribution(self, data_array, bin_edges, color, label, var_name):
        """Refactored version of comparison_histo_performance."""
        plt.hist(data_array, bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(data_array, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
        
        self.legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, 
                                             edgecolor=color, linewidth=4, alpha=0.2, label=label))
        
        plt.xlabel(var_name)
        plt.ylabel('Counts')
        self._apply_cms_style(var_name)
        plt.legend(handles=self.legend_handles)
        plt.grid(linestyle=":")
    def compute_efficiency(self, matched_gen, all_gen, x_var, bins, range_):
        """
        Calculates efficiency: Count(matched) / Count(all_gen)
        """
        bin_edges = np.linspace(range_[0], range_[1], bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        def get_x(data):
            # Helper to get the correct branch and flatten
            branch = data.pt if x_var == 'pt' else np.abs(data.eta) if x_var == 'eta' else data.phi
            return ak.to_numpy(ak.flatten(branch, axis=-1))

        x_matched = get_x(matched_gen)
        x_all = get_x(all_gen)

        n_matched, _ = np.histogram(x_matched, bins=bin_edges)
        n_all, _ = np.histogram(x_all, bins=bin_edges)

        # Efficiency calculation with error bars (Clopper-Pearson or simple Poisson)
        eff = np.divide(n_matched, n_all, out=np.zeros_like(n_matched, dtype=float), where=n_all!=0)
        eff_err = np.sqrt(eff * (1 - eff) / n_all, out=np.zeros_like(eff), where=n_all!=0)

        return bin_centers, eff, eff_err

    def plot_efficiency(self, datasets, x_var, bins, range_, filename):
        fig, ax = plt.subplots(figsize=(10, 10))
        x_label = r"$p_T^{gen}$ [GeV]" if x_var == 'pt' else r"$|\eta^{gen}|$"
        
        for ds in datasets:
            x, eff, err = self.compute_efficiency(ds['matched_gen'], ds['all_gen'], x_var, bins, range_)
            ax.errorbar(x, eff, yerr=err, xerr=(x[1]-x[0])/2, marker="o", ls="None", 
                        label=ds['label'], color=ds['color'])

        ax.set_ylabel("Efficiency")
        ax.set_xlabel(x_label)
        ax.set_ylim(0, 1.1)
        self._apply_cms_decor(ax)
        self._save_fig(filename)
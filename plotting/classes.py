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
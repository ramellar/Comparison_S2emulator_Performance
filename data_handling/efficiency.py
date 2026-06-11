import awkward as ak
from configs.config import EMU_CONFIG
import data_handling.event_performances as ev
import numpy as np
import matplotlib.pyplot as plt
import mplhep
from scipy.stats import beta
plt.style.use(mplhep.style.CMS)

def calcDeltaR(eta_cl, phi_cl, gen):

    deta = np.abs(eta_cl - gen.eta)
    dphi = np.abs(phi_cl - gen.phi)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    # If condition is true choose 2 * np.pi - dphi, else choose dphi
    
    deltaR = np.sqrt(deta**2 + dphi**2)
    return deltaR


def compute_efficiencies(events, events_gen, args):

    print("------------------------------------------")
    print("Total efficiency for each emulation test")
    print("------------------------------------------\n")

    print(
        "After selections:",
        len(events_gen),
        "events and",
        len(ak.flatten(events_gen.pt)),
        "particles",
    )
    results = {}
    for key, branch in EMU_CONFIG.items():
        results[key] = compute_total_efficiency(
            key,
            events[key],
            events_gen,
            args
        )
    return results




def get_efficiency_with_error(k, n, alpha=0.3173):   # alpha is for the definition of 1-sigma confidence interval
    if n == 0:
        return 0.0, 0.0, 0.0
    eff = k / n
    low = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    up = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return eff, eff - low, up - eff

    #Plotting total efficinecy

def compute_total_efficiency(size, event_cl, event_gen, args, deltaR=0.1):
    print("-------------------")
    print("For triangle size", size)
    print("Number of clusters before matching", len(ak.flatten(event_cl.eta, axis=-1)))
    if args.gen_pt_cut != 0:
        mask_ev = ak.num(event_cl.eta) > 0
        filtered_events = event_cl[mask_ev]
        filtered_gen = event_gen[mask_ev]
        print("Number of events with clusters", len(filtered_gen))
        print("\n")
        print("Applying pt cut on gen particles of", args.gen_pt_cut, "GeV")
        mask_gen_pt = filtered_gen.pt > args.gen_pt_cut
        event_gen_pt_cut = filtered_gen[mask_gen_pt]
        mask_gen = ak.num(event_gen_pt_cut.pt, axis=-1) > 0
        event_gen = event_gen_pt_cut[mask_gen]
        event_cl = filtered_events[mask_gen]
        print(f"Number of gen particles after gen pt cut {args.gen_pt_cut} GeV", len(ak.flatten(event_gen.pt, axis=-1)))
        print(f"Number of events after gen pt cut {args.gen_pt_cut} GeV", len(event_gen))
        print(f"Number of clusters after gen pt cut {args.gen_pt_cut} GeV", len(ak.flatten(event_cl.eta , axis=-1)))
    pair_cluster_matched, pair_gen_masked, clusters_filtered, gen_filtered = ev.apply_matching(event_cl, event_gen, args, deltaR=deltaR)
    print("Number of events after matching",len(pair_cluster_matched.pt))
    print("Number of matched clusters:", len(ak.flatten(pair_cluster_matched.pt,axis=-1)))
    print("Number of particles after matching",len(ak.flatten(pair_gen_masked.pt,axis=-1)))
    if args.gen_pt_cut != 0 and args.pt_cut != 0:
        pass
        #print("Denominator is the number of particle after the gen cut only:", len(ak.flatten(event_gen.pt, axis=-1)))
    #print("Total efficiency at particle level:", len(ak.flatten(pair_gen_masked.pt,axis=-1)) / len(ak.flatten(event_gen.pt, axis=-1)) * 100)
    #print("Total efficiency at event level:", len(pair_gen_masked) / len(event_gen.pt) * 100)
    #print("\n")
    
    n_pass_part = len(ak.flatten(pair_gen_masked.pt, axis=-1))
    n_tot_part  = len(ak.flatten(event_gen.pt, axis=-1))
    n_pass_ev = len(pair_gen_masked)
    n_tot_ev  = len(event_gen)
    eff_part, err_part_low, err_part_up = get_efficiency_with_error(n_pass_part, n_tot_part)
    eff_ev, err_ev_low, err_ev_up = get_efficiency_with_error(n_pass_ev, n_tot_ev)
    print(f"Total efficiency at particle level: " f"{eff_part*100:.2f} -{err_part_low*100:.2f} +{err_part_up*100:.2f} %")
    print(f"Total efficiency at event level: " f"{eff_ev*100:.2f} -{err_ev_low*100:.2f} +{err_ev_up*100:.2f} %")
    return pair_cluster_matched, pair_gen_masked


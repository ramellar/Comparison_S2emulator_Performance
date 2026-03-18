import awkward as ak
from configs.config import EMU_CONFIG
import data_handle.plot_performances as plot
import data_handling.event_performances as ev
import numpy as np
import matplotlib.pyplot as plt
import mplhep
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
        print("Denominator is the number of particle after the gen cut only:", len(ak.flatten(event_gen.pt, axis=-1)))
    print("Total efficiency at particle level:", len(ak.flatten(pair_gen_masked.pt,axis=-1)) / len(ak.flatten(event_gen.pt, axis=-1)) * 100)
    print("Total efficiency at event level:", len(pair_gen_masked) / len(event_gen.pt) * 100)
    print("\n")
    return pair_cluster_matched, pair_gen_masked


def differential_efficiency(event_gen, pair_gen_matched, ax, args, label=[], var="pT", bin_n=10, range_=[0,100], color="blue"):
    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)
    # genparts = ak.zip({
    #     "eta": getattr(event_gen, 'genpart_exeta'),
    #     "phi": getattr(event_gen, 'genpart_exphi'),
    #     "pt": getattr(event_gen, 'genpart_pt'),
    #     "gen_flag": getattr(event_gen,'genpart_gen'),
    # })

    flat_total_gen_pt=ak.to_numpy(ak.flatten(event_gen.pt,axis=-1))
    flat_total_gen_eta=ak.to_numpy(ak.flatten(event_gen.eta,axis=-1))
    flat_total_gen_phi=ak.to_numpy(ak.flatten(event_gen.gen_flag,axis=-1))

    flat_matched_gen_pt=ak.to_numpy(ak.flatten(pair_gen_matched.pt,axis=-1))
    flat_matched_gen_eta=ak.to_numpy(ak.flatten(pair_gen_matched.eta,axis=-1))
    flat_matched_gen_phi=ak.to_numpy(ak.flatten(pair_gen_matched.phi,axis=-1))

    if var=="pT":
        indices = np.digitize(flat_total_gen_pt, bin_edges) - 1
        indices_matched = np.digitize(flat_matched_gen_pt, bin_edges) - 1
        
    if var== "phi":
        indices = np.digitize(flat_total_gen_phi, bin_edges) - 1
        indices_matched = np.digitize(flat_matched_gen_phi, bin_edges) - 1

    if var== "eta":
        indices = np.digitize(np.abs(flat_total_gen_eta), bin_edges) - 1
        indices_matched = np.digitize(np.abs(flat_matched_gen_eta), bin_edges) - 1

    # print("matched", len(flat_matched_gen_pt))
    # print("total", len(flat_total_gen_pt))

    efficiency, error= {}, {}
    for index in range(bin_n):
        bin_idx = np.where(indices == index)[0]
        bin_idx_matched = np.where(indices_matched == index)[0]
        eff=len(bin_idx_matched)/len(bin_idx)
        # print(len(bin_idx_matched))
        # print(len(bin_idx))
        # print(eff)
        efficiency[index] = eff
        error[index] = np.sqrt(eff * (1 - eff) / len(bin_idx))

    # print(efficiency)
    # print(error)

    ax.errorbar((bin_edges[1:] + bin_edges[:-1])/2, efficiency.values(), 
                    yerr=np.array(list(zip(error.values(), error.values()))).T,
                    xerr=(bin_edges[1] - bin_edges[0])/2, ls='None', lw=2, marker='s', label=label, color=color)
    plt.ylabel(r'$\epsilon$')
    plt.xlabel(r'$p_{T}^{gen}$ [GeV]' if var=='pT' else r'$|\eta^{gen}|$')
    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup+' '+args.particles)

    if args.pt_cut != 0.0 and not args.gen_pt_cut != 0.0:
        thr = args.pt_cut
        plt.legend(title=fr"$p_T^{{\mathrm{{cluster}}}} > {args.pt_cut}$ GeV", title_fontsize=15, fontsize=17, frameon=True, facecolor='white', edgecolor='black')
    if args.gen_pt_cut != 0.0 and not args.pt_cut != 0.0:
        thr = args.gen_pt_cut
        plt.legend(title=fr"$p_T^{{\mathrm{{gen}}}} > {args.gen_pt_cut}$ GeV", title_fontsize=15, fontsize=17, frameon=True, facecolor='white', edgecolor='black')
    if args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
        thr_gen = args.gen_pt_cut
        thr = args.pt_cut
        plt.legend(title=fr"$p_T^{{\mathrm{{gen}}}} > {thr_gen} GeV$" + "\n" + f"and $p_T^{{\mathrm{{cluster}}}} > {thr} GeV$", title_fontsize=15, fontsize=17, frameon=True, facecolor='white', edgecolor='black')
    else:
        plt.legend()
    
    plt.grid()
    plt.tight_layout()
    return

    
#Plotting the efficiency as a function pt_gen and eta_gen (still testing these functions)

def compute_efficiency(cluster, gen, bin_n, var, att_eta, dr_threshold=0.1):
    clusters = ak.zip({
        "eta": getattr(cluster, att_eta),
        "phi": getattr(cluster, att_eta.replace("eta", "phi")),
        "pt": getattr(cluster, att_eta.replace("eta", "pt")),
    })

    genparts = ak.zip({
        "eta": getattr(gen, 'genpart_exeta'),
        "phi": getattr(gen, 'genpart_exphi'),
        "pt": getattr(gen, 'genpart_pt'),
        "gen_flag": getattr(gen, 'genpart_gen'),
    })

    # Cartesian product of clusters and gen particles per event
    pairs = ak.cartesian([clusters, genparts], axis=1, nested=True)

    # Compute deltaR for each cluster-gen pair
    delta_eta = pairs['0'].eta - pairs['1'].eta
    delta_phi = np.abs(pairs['0'].phi - pairs['1'].phi)
    delta_phi = ak.where(delta_phi > np.pi, 2 * np.pi - delta_phi, delta_phi)
    delta_r = np.sqrt(delta_eta**2 + delta_phi**2)
    print("delta_R", delta_r)
    # Find matches: gen matched if ANY cluster in event is close enough
    matched = ak.any(delta_r < dr_threshold, axis=1)  # shape: (n_events, n_gen_per_event)

    print("Matched", matched)

    # Mask gen array to get matched status per gen particle
    gen_mask = delta_r < dr_threshold
    print("gen_mask", gen_mask)
    gen_matched = ak.any(gen_mask, axis=1)  # one per gen particle

    print("gen_matched", gen_matched)
    # Flatten to bin by pt
    gen_flat = ak.flatten(genparts)
    matched_flat = ak.flatten(gen_matched)

    print("matched_flat",matched_flat)

    # Bin edges (linear or log)
    bin_edges = np.linspace(0, 200, num=bin_n + 1) 
    bin_indices = np.digitize(gen_flat.pt, bin_edges) - 1  

    # Initialize efficiency arrays
    eff = np.zeros(bin_n)
    counts_total = np.zeros(bin_n)
    counts_matched = np.zeros(bin_n)

    for i in range(bin_n):
        in_bin = bin_indices == i
        counts_total[i] = np.sum(in_bin)
        counts_matched[i] = np.sum(matched_flat[in_bin])
        if counts_total[i] > 0:
            eff[i] = counts_matched[i] / counts_total[i]

    return bin_edges, eff, counts_total, counts_matched


def compute_efficiency_test(cluster, gen, bin_n ,var, att_eta):

    print(gen.genpart_pt)

    clusters = ak.zip({
        "eta": getattr(cluster, att_eta),
        "phi": getattr(cluster, att_eta.replace("eta", "phi")),
        "pt": getattr(cluster, att_eta.replace("eta", "pt")),
    })

    # print("clusters pt", clusters.pt[0])
    # print("clusters eta", clusters.eta[0])

    genparts = ak.zip({
        "eta": getattr(gen, 'genpart_exeta'),
        "phi": getattr(gen, 'genpart_exphi'),
        "pt": getattr(gen, 'genpart_pt'),
        "gen_flag": getattr(gen,'genpart_gen'),
    })
    
    genparts_flat= ak.flatten(genparts, axis=-1)
    clusters_flat=ak.flatten(clusters, axis=-1)

    # print(genparts.eta)
    # print(clusters.eta)
    
    if var=="pT":
        bin_edges = np.linspace(min(genparts_flat.pt), max(genparts_flat.pt),num=bin_n+1)
        indices = np.digitize(genparts_flat.pt, bin_edges) - 1
    
        for index in range(bin_n+1):
            gen_mask = indices == index
            # gen_pt_bin = [genparts_flat.pt[i] for i in bin_indices]
            # gen_eta_bin = [genparts_flat.eta[i] for i in bin_indices]
            # gen_phi_bin = [genparts_flat.phi[i] for i in bin_indices]
            # cluster_eta_bin = [clusters_flat.eta[i] for i in bin_indices]

            gen_bin = genparts_flat[gen_mask]

            # print("gen_bin",gen_bin)

            # #MATCHING
            # pairs = ak.cartesian([clusters, genparts], axis=1, nested=True)
            # # print("pairs[0]", pairs['0'].eta[0])
            # # print("pairs[1]", pairs['1'].eta[0])
            # delta_eta = np.abs(pairs['0'].eta - pairs['1'].eta)
            # # print("delta_eta", delta_eta[0])
            # delta_phi = np.abs(pairs['0'].phi - pairs['1'].phi)
            # delta_phi = ak.where(delta_phi > np.pi, 2 * np.pi - delta_phi, delta_phi)

            # delta_r = np.sqrt(delta_eta**2 + delta_phi**2)

            # # print("delta_r", delta_r[0])
            # mask = delta_r < deltaR


            # print(gen_pt_bin)

    if var=="eta":
        bin_edges = np.linspace(min(gen.genpart_eta), max(gen.genpart_eta),num=bin_n+1)
        indices = np.digitize(gen.genpart_eta, bin_edges) - 1


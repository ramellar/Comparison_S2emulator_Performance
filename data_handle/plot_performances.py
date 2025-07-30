#test
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
from scipy.optimize import curve_fit
import numpy as np
import yaml
import json
import mplhep
import os
import awkward as ak
import numpy as np
from matplotlib.patches import Rectangle
import data_handle.event_performances as ev


with open('config_performances.yaml', "r") as afile:
    cfg = yaml.safe_load(afile)["s2emu_config"]

#Plot distributions

def comparison_histo_performance(events, att_eta, args, var, bin_n, range_,label, color, legend_handles):
    event_info = ak.zip({
        "eta": getattr(events, att_eta),
        "phi": getattr(events, att_eta.replace("eta", "phi")),
        "pt": getattr(events, att_eta.replace("eta", "pt")),
    })

    event_flat = ak.flatten(event_info, axis=-1)
    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)
    # print(event_flat.pt)

    if var == "pT":
        plt.hist(event_flat.pt, bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(event_flat.pt, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
        plt.xlabel(r'$p_{T}^{cluster}$ [GeV]')
    elif var == "eta":
        plt.hist(np.abs(ak.to_numpy(event_flat.eta)), bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(np.abs(ak.to_numpy(event_flat.eta)), bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
        plt.xlabel(r'$|\eta^{cluster}|$')
    elif var == "phi":
        plt.hist(np.abs(ak.to_numpy(event_flat.phi)), bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(np.abs(ak.to_numpy(event_flat.phi)), bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
        plt.xlabel(r'$|\phi^{cluster}|$')

    plt.ylabel('Counts')
    legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
    plt.legend(handles=legend_handles)

    plt.grid(linestyle=":")
    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup + ' ' + args.particles)

    if args.pileup == 'PU200':
        plt.yscale('log')

#Plot response and resolution

def scale_distribution(events, gen, args, var, bin_n, range_,label, color, legend_handles):
    
    # pt_dist, eta_dist, phi_dist, pt_gen_dist, eta_gen_dist, phi_gen_dist= flatten(events,gen, att_eta)
    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)

    if var == "pT":
        flat_pt=ak.to_numpy(ak.flatten(ak.flatten(events.pt,axis=-1),axis=-1))
        flat_gen=ak.to_numpy(ak.flatten(ak.flatten(gen.pt,axis=-1),axis=-1))
        scale_simul = np.divide(flat_pt, flat_gen)
        plt.hist(scale_simul, bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(scale_simul, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
        plt.xlabel(r'$p_{T}^{cluster}/p_{T}^{gen}$')
    if var == "eta":
        flat_eta=ak.to_numpy(ak.flatten(ak.flatten(events.eta,axis=-1),axis=-1))
        flat_gen=ak.to_numpy(ak.flatten(ak.flatten(gen.eta,axis=-1),axis=-1))
        scale_simul_eta = np.subtract(flat_eta,flat_gen)
        plt.hist(scale_simul_eta, bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(scale_simul_eta, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
        plt.xlabel(r'$\eta^{cluster}-\eta^{gen}$')
    if var == "phi":
        flat_phi=ak.to_numpy(ak.flatten(ak.flatten(events.phi,axis=-1),axis=-1))
        flat_gen=ak.to_numpy(ak.flatten(ak.flatten(gen.phi,axis=-1),axis=-1))
        scale_simul_phi = np.subtract(flat_phi, flat_gen)
        plt.hist(scale_simul_phi, bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(scale_simul_phi, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
        plt.xlabel(r'$\phi^{cluster}-\phi^{gen}$')

    plt.ylabel('Counts')
    legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
    plt.legend(handles=legend_handles)

    plt.grid(linestyle=":")
    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup + ' ' + args.particles)

    if args.pileup == 'PU200':
        plt.yscale('log')
    

def effrms(resp_bin, c=0.68):
    """ Compute half-width of the shortest interval (min std)
    containing a fraction 'c' of items """
    resp_bin = np.sort(resp_bin, kind="mergesort")
    m = int(c * len(resp_bin)) + 1
    min_index = np.argmin(resp_bin[m:] - resp_bin[:-m])
    return resp_bin[min_index:min_index + m]

def flatten(events, gen, att_eta):
      event_info = ak.zip({
          "eta": getattr(events, att_eta),
          "phi": getattr(events, att_eta.replace("eta", "phi")),
          "pt": getattr(events, att_eta.replace("eta", "pt")),
      })
      event_info_flat=ak.flatten(event_info)
      pt_dist= ak.to_numpy(event_info_flat.pt)
      eta_dist= ak.to_numpy(event_info_flat.eta)
      phi_dist= ak.to_numpy(event_info_flat.phi)

      genparts = ak.zip({
      "eta": getattr(gen, 'genpart_exeta'),
      "phi": getattr(gen, 'genpart_exphi'),
      "pt": getattr(gen, 'genpart_pt'),
      "gen_flag": getattr(gen,'genpart_gen'),
      })

      genparts_flat=ak.flatten(genparts)
      pt_gen_dist= ak.to_numpy(genparts_flat.pt)
      eta_gen_dist= ak.to_numpy(genparts_flat.eta)
      phi_gen_dist= ak.to_numpy(genparts_flat.phi)

      return pt_dist, eta_dist, phi_dist, pt_gen_dist, eta_gen_dist, phi_gen_dist

def compute_responses_performance(simul, gen, args, var, bin_n=10, range_=[0,200]):

    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)

    flat_pt=ak.to_numpy(ak.flatten(ak.flatten(simul.pt,axis=-1),axis=-1))
    flat_pt_gen=ak.to_numpy(ak.flatten(ak.flatten(gen.pt,axis=-1),axis=-1))

    flat_eta=ak.to_numpy(ak.flatten(ak.flatten(simul.phi,axis=-1),axis=-1))
    flat_eta_gen=ak.to_numpy(ak.flatten(ak.flatten(gen.phi,axis=-1),axis=-1))

    flat_phi=ak.to_numpy(ak.flatten(ak.flatten(simul.phi,axis=-1),axis=-1))
    flat_phi_gen=ak.to_numpy(ak.flatten(ak.flatten(gen.phi,axis=-1),axis=-1))

    # print("cluster phi",flat_phi)
    # print("gen phi",flat_phi_gen)

    if var=="pT" or var=="n_cl_pt":
        indices = np.digitize(flat_pt_gen, bin_edges) - 1
    
    if var == 'pT_eta' or var=="n_cl_eta":
       indices = np.digitize(np.abs(flat_eta_gen), bin_edges) - 1
    
    if var== "phi":
        indices = np.digitize(flat_phi_gen, bin_edges) - 1

    if var== "eta":
        indices = np.digitize(np.abs(flat_eta_gen), bin_edges) - 1

    resp_simul, err_resp_simul, resol_simul, err_resol_simul = {}, {}, {}, {}
    for index in range(bin_n):
    
      bin_idx = np.where(indices == index)[0]
      resp_bin_simul =[flat_pt[i]/flat_pt_gen[i] for i in bin_idx] if var=='pT' else \
                      [flat_pt[i]/flat_pt_gen[i]  for i in bin_idx] if var=='pT_eta' else \
                      [len(flat_pt)[i] for i in bin_idx] if var=='n_cl_pt' or var=='n_cl_eta' else \
                      [flat_eta[i]-flat_eta_gen[i] for i in bin_idx] if var=="eta" else \
                      [flat_phi[i]-flat_phi_gen[i] for i in bin_idx] 
                      

      resp_simul[index]     = np.mean(resp_bin_simul) if len(resp_bin_simul)>0 else 0
      err_resp_simul[index] = np.std(resp_bin_simul)/np.sqrt(len(resp_bin_simul)) if len(resp_bin_simul)>0 else 0


      if var=="pT" or var=="pT_eta":
         resol_simul[index]     = np.std(resp_bin_simul)/np.abs(np.mean(resp_bin_simul)) if len(resp_bin_simul)>1 else 0
         err_resol_simul[index] = np.std(resp_bin_simul)/(np.sqrt(2*len(resp_bin_simul)-2)*np.mean(resp_bin_simul)) if len(resp_bin_simul)>1 else 0
    
      if var=="eta" or var=="phi" or var=="n_cl_pt" or var=="n_cl_eta":
         resol_simul[index]     = np.std(resp_bin_simul) if len(resp_bin_simul)>1 else 0
         err_resol_simul[index] = np.std(resp_bin_simul)/(np.sqrt(2*len(resp_bin_simul)-2)) if len(resp_bin_simul)>1 else 0
    
      if args.eff_rms and (var == 'pT' or var == 'pT_eta'):
        eff_rms_simul = effrms(resp_bin_simul) if len(resp_bin_simul)>1 else [0]
        resol_simul[index]     = np.std(eff_rms_simul)/np.mean(eff_rms_simul) if len(eff_rms_simul)>1 else 0
        err_resol_simul[index] = np.std(eff_rms_simul)/(np.sqrt(2*len(eff_rms_simul)-2)*np.mean(eff_rms_simul)) if len(eff_rms_simul)>1 else 0

    return resp_simul, err_resp_simul, resol_simul, err_resol_simul, bin_edges


def plot_responses(simul, gen, args, var, ax, label, color, bin_n=10, range_=[0,200]):
    resp_simul, err_resp_simul, resol_simul, err_resol_simul, bin_edges= compute_responses_performance(simul, gen, args, var , bin_n, range_)
    if args.response:
        plt.style.use(mplhep.style.CMS)
        ax.errorbar((bin_edges[1:] + bin_edges[:-1])/2, resp_simul.values(), 
                    yerr=np.array(list(zip(err_resp_simul.values(), err_resp_simul.values()))).T,
                    xerr=(bin_edges[1] - bin_edges[0])/2, ls='None', lw=2, marker='s', label=label, color=color)
        plt.ylabel(r'$\phi^{cluster}-\phi^{gen}$' if var=='phi' else r'$\eta^{cluster}-\eta^{gen}$' if var=='eta' else \
                r'$<cluster>$' if var=='n_cl_pt' or var=='n_cl_eta' else r'$p_{T}^{cluster}/p_{T}^{gen}$')
        plt.xlabel(r'$p_{T}^{gen}$ [GeV]' if var=='pT' or var=='n_cl_pt' else r'$\phi^{gen}$' if var=='phi' else r'$|\eta^{gen}|$')
        # legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
        # plt.legend(handles=legend_handles)
        mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup+' '+args.particles+' - '+str(cfg['thresholdMaximaParam_a'][0])+'GeV')
        plt.legend()
        plt.grid()
        plt.tight_layout()
    if args.resolution:
        plt.style.use(mplhep.style.CMS)
        ax.errorbar((bin_edges[1:] + bin_edges[:-1])/2, resol_simul.values(), 
                    yerr=np.array(list(zip(err_resol_simul.values(), err_resol_simul.values()))).T,
                    xerr=(bin_edges[1] - bin_edges[0])/2, ls='None', lw=2, marker='s', label=label, color=color)
        plt.ylabel(r'$\sigma^{cluster}$' if var=='phi' else r'$\sigma^{cluster}$' if var=='eta' else \
                r'$\sigma^{cluster}$' if var=='n_cl_pt' or var=='n_cl_eta' else r'$\sigma^{cluster}/\mu^{cluster}$')
        plt.xlabel(r'$p_{T}^{gen}$ [GeV]' if var=='pT' or var=='n_cl_pt' else r'$\phi^{gen}$' if var=='phi' else r'$|\eta^{gen}|$')
        # legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
        # plt.legend(handles=legend_handles)
        mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup+' '+args.particles+' - '+str(cfg['thresholdMaximaParam_a'][0])+'GeV')
        plt.legend()
        plt.grid()
        plt.tight_layout()


#Matching function (not used for the moment)

def calcDeltaR(eta_cl, phi_cl, gen):

    deta = np.abs(eta_cl - gen.eta)
    dphi = np.abs(phi_cl - gen.phi)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    # If condition is true choose 2 * np.pi - dphi, else choose dphi
    
    deltaR = np.sqrt(deta**2 + dphi**2)
    return deltaR


#Plotting total efficinecy

def compute_total_efficiency(size, event_cl, event_gen, args, att_eta, att_phi, deltaR=0.1):
    print("For triangle size", size)
    pair_cluster_matched, pair_gen_masked = ev.apply_matching(event_cl, att_eta, att_phi, event_gen, args, deltaR=deltaR)
    print("Number of events after matching",len(pair_cluster_matched.pt))
    print("Number of particles after matching",len(ak.flatten(pair_gen_masked.pt,axis=-1)))
    print("Total efficiency at particle level:", len(ak.flatten(pair_gen_masked.pt,axis=-1)) / len(ak.flatten(event_gen.genpart_pt, axis=-1)) * 100)
    print("Total efficiency at event level:", len(pair_gen_masked) / len(event_gen.genpart_pt) * 100)
    print("\n")
    return pair_cluster_matched, pair_gen_masked


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
        



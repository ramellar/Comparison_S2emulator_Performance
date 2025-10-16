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

#########################################################################################
#################################### DISTRIBUTIONS ######################################
#########################################################################################

def comparison_histo_performance(events, att_eta, args, var, bin_n, range_,label, color, legend_handles, matched=False):
    
    if matched==True:
        event_info = ak.zip({
            "eta": events.eta,
            "phi": events.phi,
            "pt": events.pt,
            "delta_r": events.delta_r,
        })
    else:
        event_info = ak.zip({
            "eta": getattr(events, att_eta),
            "phi": getattr(events, att_eta.replace("eta", "phi")),
            "pt": getattr(events, att_eta.replace("eta", "pt")),
        })

    event_flat = ak.flatten(event_info, axis=-1)
    if matched==True:
        # print("deltaR",event_info.delta_r)
        event_flat = ak.flatten(event_flat, axis=-1)
        # print("deltaR",event_flat.delta_r)
    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)
    # print("edges", bin_edges)
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
        plt.hist((ak.to_numpy(event_flat.phi)), bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist((ak.to_numpy(event_flat.phi)), bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
        plt.xlabel(r'$\phi^{cluster}$')
    elif var == "delta_r":
        # print("DeltaR min:", ak.to_numpy(event_flat.delta_r).min(), "max:", ak.to_numpy(event_flat.delta_r).max())
        # print("Delta R", event_flat.delta_r)
        plt.hist((ak.to_numpy(event_flat.delta_r)), bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist((ak.to_numpy(event_flat.delta_r)), bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
        plt.xlabel(r'$\Delta R (cluster, gen)$')


    plt.ylabel('Counts')
    if args.pt_cut != 0.0 and not args.gen_pt_cut != 0.0:
        thr = args.pt_cut
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
        plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{cluster}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black')
    if args.gen_pt_cut != 0.0 and not args.pt_cut != 0.0:
        thr = args.gen_pt_cut
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
        plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black')
    if args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
        thr = args.gen_pt_cut
        thr2 = args.pt_cut
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
        plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$" + "\n" + f"and $p_T^{{\mathrm{{cluster}}}} > {thr2} GeV$",frameon=True, facecolor='white', edgecolor='black', title_fontsize=14)
    else:
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
        plt.legend(handles=legend_handles)

    plt.grid(linestyle=":")
    # mask = (event_flat.pt >= 0) & (event_flat.pt < 5)
    # print("Events with pt in [0, 5):", ak.sum(mask))
    # mask = (event_flat.pt >= 20) & (event_flat.pt < 25)
    # print("Events with pt in [20, 25):", ak.sum(mask))
    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup + ' ' + args.particles)

    if args.pileup == 'PU200' or var=="delta_r":
        plt.yscale('log')
        # plt.ylim(1, None)

def plot_2D_histograms(events, att_eta, var1, var2, args, label, range_=[[-3.0, 3.0], [0, 100]], matched=False):
    plt.figure(figsize=(10, 10))

    if matched==True:
        event_info = ak.zip({
            "eta": events.eta,
            "phi": events.phi,
            "pt": events.pt,
            "delta_r": events.delta_r,
        })
    else:
        event_info = ak.zip({
        "eta": getattr(events, att_eta),
        "phi": getattr(events, att_eta.replace("eta", "phi")),
        "pt": getattr(events, att_eta.replace("eta", "pt")),
    })

    event_flat = ak.flatten(event_info, axis=-1)
    if matched==True:
        event_flat = ak.flatten(event_flat, axis=-1)
    # bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)
    # print("flatpt",event_flat.pt)
    # mask = (event_flat.pt >= 0) & (event_flat.pt < 5)
    # print("Events with pt in [0, 5):", ak.sum(mask))
    # mask = (event_flat.pt >= 20) & (event_flat.pt < 25)
    # print("Events with pt in [20, 25):", ak.sum(mask))
    if var1=="eta" and var2=="pt":
        plt.hist2d(ak.to_numpy(event_flat.eta), ak.to_numpy(event_flat.pt), bins=40, range=range_, cmap='magma_r')
        plt.xlabel(r'$\eta^{cluster}$')
        plt.ylabel(r'$p_T^{cluster}$')
        plt.colorbar(label="Counts")
    if var1=="eta" and var2=="delta_r":
        plt.hist2d(ak.to_numpy(event_flat.eta), ak.to_numpy(event_flat.delta_r), bins=40, range=range_, cmap='magma_r')
        plt.ylabel(r'$\Delta R (cluster, gen)$')
        plt.xlabel(r'$\eta^{cluster}$')
        plt.colorbar(label="Counts")
    if args.pt_cut !=0 :
        if label != "Ref":
            plt.title(f"Algorithm with triangle size: {label} and "+r'$pt^{cluster}>$'+f"{args.pt_cut} GeV",fontsize=18)
        else:
            plt.title(f"CMSSW simulation and "+r'$pt^{cluster}>$'+f"{args.pt_cut} GeV",fontsize=18)
    else:
        if label != "Ref":
            plt.title(f"Algorithm with triangle size: {label}",fontsize=20)
        else:
            plt.title(f"CMSSW simulation",fontsize=20)
     
    plt.tight_layout()
    


#Plot response and resolution

def scale_distribution(events, gen, args, var, bin_n, range_,label, color, legend_handles):
    
    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)
    # print("events",events)
    if var == "pT":
        # print(events.pt)
        # print("flattened pt:",ak.flatten(events.pt,axis=-1))
        flat_pt=ak.to_numpy(ak.flatten(events.pt,axis=-1))
        flat_gen=ak.to_numpy(ak.flatten(gen.pt,axis=-1))
        # print("flat pt:",flat_pt)
        scale_simul = np.divide(flat_pt, flat_gen)
        plt.hist(scale_simul, bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(scale_simul, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
        plt.xlabel(r'$p_{T}^{cluster}/p_{T}^{gen}$')
    if var == "eta":
        flat_eta=ak.to_numpy(ak.flatten(events.eta, axis=-1))
        flat_gen=ak.to_numpy(ak.flatten(gen.eta, axis=-1))
        scale_simul_eta = np.subtract(flat_eta,flat_gen)
        plt.hist(scale_simul_eta, bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(scale_simul_eta, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
        plt.xlabel(r'$\eta^{cluster}-\eta^{gen}$')
    if var == "phi":
        flat_phi=ak.to_numpy(ak.flatten(events.phi, axis=-1))
        flat_gen=ak.to_numpy(ak.flatten(gen.phi, axis=-1))
        scale_simul_phi = np.subtract(flat_phi, flat_gen)
        plt.hist(scale_simul_phi, bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(scale_simul_phi, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
        plt.xlabel(r'$\phi^{cluster}-\phi^{gen}$')
        # plt.yscale('log')

    plt.ylabel('Counts')
    legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))

    if args.pt_cut != 0.0 and not args.gen_pt_cut != 0.0:
        thr = args.pt_cut
        plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{cluster}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black')
    if args.gen_pt_cut != 0.0 and not args.pt_cut != 0.0:
        thr = args.gen_pt_cut
        plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black')
    if args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
        thr = args.gen_pt_cut
        thr2 = args.pt_cut
        plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$" + "\n" + f"and $p_T^{{\mathrm{{cluster}}}} > {thr2} GeV$",frameon=True, facecolor='white', edgecolor='black', title_fontsize=14)
    else:
        plt.legend(handles=legend_handles)

    # if args.pt_cut != 0:
    #     plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{cluster}}}} > {args.pt_cut} GeV$")
    # else:
    #     plt.legend(handles=legend_handles)

    plt.grid(linestyle=":")
    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup + ' ' + args.particles)

    if args.pileup == 'PU200':
        plt.yscale('log')
    

def effrms(resp_bin, c=0.68):
    """ Compute half-width of the shortest interval (min std)
    containing a fraction 'c' of items """
    resp_bin = np.sort(resp_bin, kind="mergesort")
    m = int(c * len(resp_bin)) + 1
    width = np.min(resp_bin[m:] - resp_bin[:-m])
    return width / 2.0

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

def number_of_clusters_per_event(event_info, genparts, att_eta, ax, args, gen_n=1, var="n_cl_pt", bin_n=10, range_=[0,100], label="0.0113", color='blue'):
    
    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)
    # event_info = ak.zip({
    #     "eta": getattr(events, att_eta),
    #     "phi": getattr(events, att_eta.replace("eta", "phi")),
    #     "pt": getattr(events, att_eta.replace("eta", "pt")),
    # })
    # genparts = ak.zip({
    #     "eta": getattr(gen, 'genpart_exeta'),
    #     "phi": getattr(gen, 'genpart_exphi'),
    #     "pt": getattr(gen, 'genpart_pt'),
    #     "gen_flag": getattr(gen,'genpart_gen'),
    # })
    n_clusters = ak.count(event_info.pt, axis=-1)
    number_gen = ak.count(genparts.pt, axis=-1)

    if gen_n ==1:
        #Number of clusters when there is only 1 gen particle per event
        mask_clusters_single_gen = number_gen == 1
        single_gen= genparts[mask_clusters_single_gen]
        flat_gen_pt = ak.flatten(single_gen.pt, axis=-1)
        flat_gen_eta = ak.flatten(single_gen.eta, axis=-1)
        cluster_for_single_gen= event_info[mask_clusters_single_gen]
        n_clusters= ak.count(cluster_for_single_gen.pt, axis=-1)

    if gen_n ==2:
        #Number of clusters when there are 2 gen particles per event
        #This can only be done as a function of pt because 
        # the eta and the phi of the two gen particles of the event 
        # are different, and we don't know how many clusters where 
        # reconstructed for each particle before we do the matching
        mask_clusters_double_gen = number_gen == 2
        double_gen= genparts[mask_clusters_double_gen]
        one_pt_saved= double_gen[:, 0]
        flat_gen_pt = ak.flatten(one_pt_saved.pt, axis=-1)
        cluster_for_double_gen= event_info[mask_clusters_double_gen]
        n_clusters= ak.count(cluster_for_double_gen.pt, axis=-1)

    if var == "n_cl_eta":
        indices = np.digitize(np.abs(flat_gen_eta), bin_edges) - 1

    if var == "n_cl_pt":
        indices = np.digitize(flat_gen_pt, bin_edges) - 1

    resp_simul, err_resp_simul, resol_simul, err_resol_simul = {}, {}, {}, {}
    for index in range(bin_n):
        bin_idx = np.where(indices == index)[0]
        resp_bin_simul =n_clusters[bin_idx]

        resp_simul[index]     = np.mean(resp_bin_simul) if len(resp_bin_simul)>0 else 0
        err_resp_simul[index] = np.std(resp_bin_simul)/np.sqrt(len(resp_bin_simul)) if len(resp_bin_simul)>0 else 0

        resol_simul[index]     = np.std(resp_bin_simul) if len(resp_bin_simul)>1 else 0
        err_resol_simul[index] = np.std(resp_bin_simul)/(np.sqrt(2*len(resp_bin_simul)-2)) if len(resp_bin_simul)>1 else 0

    plt.style.use(mplhep.style.CMS)
    if args.response:
        ax.errorbar((bin_edges[1:] + bin_edges[:-1])/2, resp_simul.values(), 
                    yerr=np.array(list(zip(err_resp_simul.values(), err_resp_simul.values()))).T,
                    xerr=(bin_edges[1] - bin_edges[0])/2, ls='None', lw=2, marker='s', label=label, color=color)
        plt.ylabel(r'$<cluster>$')
    if args.distribution:
        plt.hist(n_clusters, bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(n_clusters, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
        plt.xlabel(r'$N_{clusters}$')
    if args.resolution:
        ax.errorbar((bin_edges[1:] + bin_edges[:-1])/2, resol_simul.values(), 
                    yerr=np.array(list(zip(err_resol_simul.values(), err_resol_simul.values()))).T,
                    xerr=(bin_edges[1] - bin_edges[0])/2, ls='None', lw=2, marker='s', label=label, color=color)
        plt.ylabel(r'$\sigma^{cluster}$')
    plt.xlabel(r'$p_{T}^{gen}$ [GeV]' if var=='n_cl_pt' else r'$|\eta^{gen}|$')
    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup+' '+args.particles+' - '+ str(gen_n)+' gen particles')
    
    if args.pt_cut != 0.0 and not args.gen_pt_cut != 0.0:
        thr = args.pt_cut
        plt.legend(title=fr"$p_T^{{\mathrm{{cluster}}}} > {thr} GeV$", title_fontsize=15, fontsize=17)
    if args.gen_pt_cut != 0.0 and not args.pt_cut != 0.0:
        thr = args.gen_pt_cut
        plt.legend(title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$", title_fontsize=15, fontsize=17)
    if args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
        thr = args.gen_pt_cut
        thr2 = args.pt_cut
        plt.legend(title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$" + "\n" + f"and $p_T^{{\mathrm{{cluster}}}} > {thr2} GeV$", title_fontsize=15, fontsize=17)
    else:
        plt.legend(fontsize=18)
    
    # if args.pt_cut != 0:
    #     plt.legend(title=fr"$p_T^{{\mathrm{{cluster}}}} > {args.pt_cut}$ GeV", title_fontsize=15, fontsize=17)
    # else:
    #     plt.legend()
    plt.grid()
    plt.tight_layout()
    return 

def compute_number_of_clusters_per_event(event_info, genparts, att_eta, gen_n=1, bin_n=10, range_=[0,100], var=""):
    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)
    # event_info = ak.zip({
    #     "eta": getattr(events, att_eta),
    #     "phi": getattr(events, att_eta.replace("eta", "phi")),
    #     "pt": getattr(events, att_eta.replace("eta", "pt")),
    # })
    # genparts = ak.zip({
    #     "eta": getattr(gen, 'eta'),
    #     "phi": getattr(gen, 'phi'),
    #     "pt": getattr(gen, 'pt'),
    #     "gen_flag": getattr(gen,'gen_flag'),
    # })
    # print(ak.type(event_info))
    # print(ak.type(genparts))
    n_clusters = ak.count(event_info.pt, axis=-1)
    number_gen = ak.count(genparts.pt, axis=-1)

    if gen_n ==1:
        #Number of clusters when there is only 1 gen particle per event
        mask_clusters_single_gen = number_gen == 1
        single_gen= genparts[mask_clusters_single_gen]
        flat_gen_pt = ak.flatten(single_gen.pt, axis=-1)
        flat_gen_eta = ak.flatten(single_gen.eta, axis=-1)
        cluster_for_single_gen= event_info[mask_clusters_single_gen]
        n_clusters= ak.count(cluster_for_single_gen.pt, axis=-1)

    if gen_n ==2:
        #Number of clusters when there are 2 gen particles per event
        #This can only be done as a function of pt because 
        # the eta and the phi of the two gen particles of the event 
        # are different, and we don't know how many clusters where 
        # reconstructed for each particle before we do the matching
        mask_clusters_double_gen = number_gen == 2
        double_gen= genparts[mask_clusters_double_gen]
        one_pt_saved= double_gen[:, 0]
        flat_gen_pt = ak.flatten(one_pt_saved.pt, axis=-1)
        cluster_for_double_gen= event_info[mask_clusters_double_gen]
        n_clusters= ak.count(cluster_for_double_gen.pt, axis=-1)

    if var == "n_cl_eta":
        indices = np.digitize(np.abs(flat_gen_eta), bin_edges) - 1
    elif var == "n_cl_pt":
        indices = np.digitize(flat_gen_pt, bin_edges) - 1
    else:
        indices = None

    return n_clusters, indices, bin_edges



def distribution_of_clusters_per_event(events, gen, att_eta, args, legend_handles, gen_n=1, bin_n=10, range_=[0,100], label="0.0113", color='blue',var="", output_dir="", bin=40, range_n_cl_pt=[0,100]):
    n_clusters, indices, bin_edges_n = compute_number_of_clusters_per_event(events, gen, att_eta, gen_n=1)
    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)

    plt.style.use(mplhep.style.CMS)
    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup+' '+args.particles+' - '+ str(gen_n)+' gen particles')
    plt.hist(n_clusters, bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
    plt.hist(n_clusters, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
    plt.xlabel(r'$N_{clusters}$')
    plt.ylabel('Counts')
    plt.yscale('log')

    if args.pt_cut != 0.0 and not args.gen_pt_cut != 0.0:
        thr = args.pt_cut
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
        plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{cluster}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black')
    if args.gen_pt_cut != 0.0 and not args.pt_cut != 0.0:
        thr = args.gen_pt_cut
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
        plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black')
    if args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
        thr_gen = args.gen_pt_cut
        thr = args.pt_cut
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
        plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{gen}}}} > {thr_gen} GeV$" + "\n" + f"and $p_T^{{\mathrm{{cluster}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black', title_fontsize=15, fontsize=17)
    else:
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
        plt.legend(handles=legend_handles)

    return 




def plot_clusters_per_bin(datasets, bin_n, range_, bin_nb, range_nb, var, args, output_dir, gen_n=1):
    results = []
    for ev, events_gen, label, color, att in datasets:
        n_clusters, indices, bin_edges = compute_number_of_clusters_per_event(ev, events_gen, att, gen_n, bin_n, range_, var)
        results.append((n_clusters, indices, label, color))

    for i in range(bin_n):
        plt.figure(figsize=(10, 10))
        plt.style.use(mplhep.style.CMS)

    
        if var == "n_cl_pt":
            bin_label = fr"${bin_edges[i]:.1f} < p_T^{{\mathrm{{gen}}}} < {bin_edges[i+1]:.1f}\,\mathrm{{GeV}}$"
        elif var == "n_cl_eta":
            bin_label = fr"${bin_edges[i]:.2f} < |\eta|^{{\mathrm{{gen}}}} < {bin_edges[i+1]:.2f}$"
        else:
            bin_label = f"Bin {i}"

        legend_handles = []
        for n_clusters, indices, label, color in results:
            bin_idx = np.where(indices == i)[0]
            if len(bin_idx) == 0:
                continue
            plt.hist(n_clusters[bin_idx], bins=bin_nb, range= range_nb,
                    alpha=0.2, color=color, histtype='stepfilled')
            plt.hist(n_clusters[bin_idx], bins=bin_nb, range= range_nb,
                    histtype='step', linewidth=2.5, color=color, label=label)

            legend_handles.append(
                Rectangle((0, 0), 1, 1,
                        facecolor=color, edgecolor=color,
                        linewidth=4, alpha=0.2, label=label)
            )

        plt.xlabel(r"$N_{clusters}$")
        plt.ylabel("Counts")
        plt.yscale("log")
        mplhep.cms.label("Preliminary", data=True,
                        rlabel=args.pileup + " " + args.particles + f" - {gen_n} gen particle")
        
        if args.pt_cut != 0.0 and not args.gen_pt_cut != 0.0:
            thr = args.pt_cut
            plt.legend(handles=legend_handles,
                title=bin_label + "\n" + fr"$p_T^{{\mathrm{{cluster}}}} > {thr} GeV$",
                frameon=True, facecolor="white", edgecolor="black", title_fontsize=14)
        if args.gen_pt_cut != 0.0 and not args.pt_cut != 0.0:
            thr = args.gen_pt_cut
            plt.legend(handles=legend_handles,
                title=bin_label + "\n" + fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$",
                frameon=True, facecolor="white", edgecolor="black", title_fontsize=14)
        if args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
            thr = args.gen_pt_cut
            thr2 = args.pt_cut
            plt.legend(handles=legend_handles,
                title=bin_label + "\n" + fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$" + "\n" + f"and $p_T^{{\mathrm{{cluster}}}} > {thr2} GeV$",
                frameon=True, facecolor="white", edgecolor="black", title_fontsize=14)
        else:
            plt.legend(handles=legend_handles,
                title=bin_label,
                frameon=True, facecolor="white", edgecolor="black")

        plt.grid()
        plt.tight_layout()

        plt.savefig(f"{output_dir}/distributions/Distribution_Nclusters_bin{i}_{var}.png", dpi=300)
        plt.savefig(f"{output_dir}/distributions/Distribution_Nclusters_bin{i}_{var}.pdf")
        print(f"Saved figure: {output_dir}/distributions/Distribution_Nclusters_bin{i}_{var}.png")
        plt.close()
        legend_handles = []

    return

def plot_responses_per_bin(datasets, bin_n, range_, bin_nb, range_nb, var, args, output_dir):
    results = []
    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)

    for ev, matched_ev, matched_gen, label, color, att in datasets:
        resp_simul, err_resp_simul, resol_simul, err_resol_simul, _, indices, resp_bin_simul = compute_responses_performance(
            matched_ev, matched_gen, args, var, ev, att, bin_n, range_
        )

        flat_pt = ak.to_numpy(ak.flatten(matched_ev.pt, axis=-1))
        flat_pt_gen = ak.to_numpy(ak.flatten(matched_gen.pt, axis=-1))
        flat_eta = ak.to_numpy(ak.flatten(matched_ev.eta, axis=-1))
        flat_eta_gen = ak.to_numpy(ak.flatten(matched_gen.eta, axis=-1))
        flat_phi = ak.to_numpy(ak.flatten(matched_ev.phi, axis=-1))
        flat_phi_gen = ak.to_numpy(ak.flatten(matched_gen.phi, axis=-1))

        results.append((resp_bin_simul, indices, label, color,
                        flat_pt, flat_pt_gen, flat_eta, flat_eta_gen, flat_phi, flat_phi_gen))

  
    for i in range(bin_n):
        plt.figure(figsize=(10, 10))
        plt.style.use(mplhep.style.CMS)

        # Bin label
        if var == "pT":
            bin_label = fr"${bin_edges[i]:.1f} < p_T^{{\mathrm{{gen}}}} < {bin_edges[i+1]:.1f}\,\mathrm{{GeV}}$"
        elif var in ["eta", "pT_eta"]:
            bin_label = fr"${bin_edges[i]:.2f} < |\eta|^{{\mathrm{{gen}}}} < {bin_edges[i+1]:.2f}$"
        elif var in ["phi", "pT_phi"]:
            bin_label = fr"${bin_edges[i]:.2f} < \phi^{{\mathrm{{gen}}}} < {bin_edges[i+1]:.2f}$"
        else:
            bin_label = f"Bin {i}"

        legend_handles = []

        # Plot for each dataset
        for resp_bin_simul, indices, label, color, flat_pt, flat_pt_gen, flat_eta, flat_eta_gen, flat_phi, flat_phi_gen in results:
            bin_idx = np.where(indices == i)[0]
            if len(bin_idx) == 0:
                continue

            if var in ["pT", "pT_eta", "pT_phi"]:
                ratio = flat_pt[bin_idx] / flat_pt_gen[bin_idx]
            elif var == "eta":
                ratio = flat_eta[bin_idx] - flat_eta_gen[bin_idx]
            elif var == "phi":
                ratio = flat_phi[bin_idx] - flat_phi_gen[bin_idx]
            else:
                continue
            # if var =="phi":
            #     print("Mean phi response bin", i, ":", np.mean(ratio))
            #     print("RMS phi response bin", i, ":", np.std(ratio))
            #     ratio = (ratio + np.pi) % (2*np.pi) - np.pi
            #     print("Mean phi response bin (after adjustment):", np.mean(ratio))
            #     print("RMS phi response bin (after adjustment):", np.std(ratio))

            plt.hist(ratio, bins=bin_nb, range=range_nb,
                     alpha=0.2, color=color, histtype='stepfilled')
            plt.hist(ratio, bins=bin_nb, range=range_nb,
                     histtype='step', linewidth=2.5, color=color, label=label)

            legend_handles.append(
                Rectangle((0, 0), 1, 1,
                          facecolor=color, edgecolor=color,
                          linewidth=4, alpha=0.2, label=label)
            )

        plt.xlabel(r'$<\phi^{cluster}-\phi^{gen}>$' if var=='phi' else r'$<\eta^{cluster}-\eta^{gen}>$' if var=='eta' else r'$<p_{T}^{cluster}/p_{T}^{gen}>$')
        plt.ylabel("Counts")
        if args.pileup == 'PU200':
            plt.yscale('log')  
        # plt.yscale("log")
        mplhep.cms.label("Preliminary", data=True,
                         rlabel=args.pileup + " " + args.particles)
        
        if args.pt_cut != 0.0 and not args.gen_pt_cut != 0.0:
            thr = args.pt_cut
            plt.legend(handles=legend_handles,
                title=bin_label + "\n" + fr"$p_T^{{\mathrm{{cluster}}}} > {thr} GeV$",
                frameon=True, facecolor="white", edgecolor="black", title_fontsize=14)
        if args.gen_pt_cut != 0.0 and not args.pt_cut != 0.0:
            thr = args.gen_pt_cut
            plt.legend(handles=legend_handles,
                title=bin_label + "\n" + fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$",
                frameon=True, facecolor="white", edgecolor="black", title_fontsize=14)
        if args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
            thr = args.gen_pt_cut
            thr2 = args.pt_cut
            plt.legend(handles=legend_handles,
                title=bin_label + "\n" + fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$" + "\n" + f"and $p_T^{{\mathrm{{cluster}}}} > {thr2} GeV$",
                frameon=True, facecolor="white", edgecolor="black", title_fontsize=14)
        else:
            plt.legend(handles=legend_handles,
                title=bin_label,
                frameon=True, facecolor="white", edgecolor="black")

     
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/responses/Response_bin{i}_{var}.png", dpi=300)
        plt.savefig(f"{output_dir}/responses/Response_bin{i}_{var}.pdf")
        print(f"Saved figure: {output_dir}/responses/Response_bin{i}_{var}.png")
        plt.close()
        legend_handles = []


# def plot_responses_per_bin(datasets, events_gen, bin_n, range_, bin_nb, range_nb, var, args, output_dir, gen_n=1):
#     results = []
#     for ev, matched_ev, matched_gen, label, color, att in datasets:
#         resp_simul, err_resp_simul, resol_simul, err_resol_simul, bin_edges, indices, resp_bin_simul= compute_responses_performance(matched_ev, matched_gen, args, var , ev, att, bin_n, range_)
#         print(resp_bin_simul)
#         results.append((resp_bin_simul, indices, label, color))

#         bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)

#         flat_pt=ak.to_numpy(ak.flatten(matched_ev.pt,axis=-1))
#         flat_pt_gen=ak.to_numpy(ak.flatten(matched_gen.pt,axis=-1))

#         flat_eta=ak.to_numpy(ak.flatten(matched_ev.eta,axis=-1))
#         flat_eta_gen=ak.to_numpy(ak.flatten(matched_gen.eta,axis=-1))

#         flat_phi=ak.to_numpy(ak.flatten(matched_ev.phi,axis=-1))
#         flat_phi_gen=ak.to_numpy(ak.flatten(matched_gen.phi,axis=-1))

#     if var=="pT":
#         indices = np.digitize(flat_pt_gen, bin_edges) - 1
    
#     if var == 'pT_eta':
#        indices = np.digitize(np.abs(flat_eta_gen), bin_edges) - 1
    
#     if var== "pT_phi":
#         indices = np.digitize(flat_phi_gen, bin_edges) - 1

#     if var== "phi":
#         indices = np.digitize(flat_phi_gen, bin_edges) - 1

#     if var== "eta":
#         indices = np.digitize(np.abs(flat_eta_gen), bin_edges) - 1

#     for i in range(bin_n):
#         plt.figure(figsize=(10, 10))
#         plt.style.use(mplhep.style.CMS)

#         if var == "pt":
#             bin_label = fr"${bin_edges[i]:.1f} < p_T < {bin_edges[i+1]:.1f}\,\mathrm{{GeV}}$"
#         elif var == "eta":
#             bin_label = fr"${bin_edges[i]:.2f} < |\eta| < {bin_edges[i+1]:.2f}$"
#         elif var == "phi":
#             bin_label = fr"${bin_edges[i]:.2f} < |\phi| < {bin_edges[i+1]:.2f}$"
#         elif var == "pT_eta":
#             bin_label = fr"${bin_edges[i]:.2f} < |\eta| < {bin_edges[i+1]:.2f}$"
#         elif var == "pT_phi":
#             bin_label = fr"${bin_edges[i]:.2f} < |\phi| < {bin_edges[i+1]:.2f}$"
#         else:
#             bin_label = f"Bin {i}"

#         legend_handles = []
#         for indices, label, color in results:
#             bin_idx = np.where(indices == i)[0]
#             if len(bin_idx) == 0:
#                 continue
#             plt.hist(flat_pt[bin_idx]/flat_pt_gen[bin_idx], bins=bin_nb, range= range_nb, alpha=0.2, color=color, histtype='stepfilled')
#             plt.hist(flat_pt[bin_idx]/flat_pt_gen[bin_idx], bins=bin_nb, range= range_nb, histtype='step', linewidth=2.5, color=color, label=label)

#             legend_handles.append(
#                 Rectangle((0, 0), 1, 1,
#                         facecolor=color, edgecolor=color,
#                         linewidth=4, alpha=0.2, label=label)
#             )

#         plt.xlabel(r"$N_{clusters}$")
#         plt.ylabel("Counts")
#         plt.yscale("log")
#         mplhep.cms.label("Preliminary", data=True,
#                         rlabel=args.pileup + " " + args.particles + f" - {gen_n} gen particle")

#         plt.legend(handles=legend_handles,
#                 title=bin_label,
#                 frameon=True, facecolor="white", edgecolor="black")
#         plt.grid()
#         plt.tight_layout()

#         plt.savefig(f"{output_dir}/distributions/Distribution_Nclusters_bin{i}_{var}.png", dpi=300)
#         plt.savefig(f"{output_dir}/distributions/Distribution_Nclusters_bin{i}_{var}.pdf")
#         print(f"Saved figure: {output_dir}/distributions/Distribution_Nclusters_bin{i}_{var}.png")
#         plt.close()
#         legend_handles = []

#     return


def compute_responses_performance(matched, matched_gen, args, var, events, att_eta, bin_n=10, range_=[0,200]):

    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)

    flat_pt=ak.to_numpy(ak.flatten(matched.pt,axis=-1))
    flat_pt_gen=ak.to_numpy(ak.flatten(matched_gen.pt,axis=-1))

    flat_eta=ak.to_numpy(ak.flatten(matched.eta,axis=-1))
    flat_eta_gen=ak.to_numpy(ak.flatten(matched_gen.eta,axis=-1))

    flat_phi=ak.to_numpy(ak.flatten(matched.phi,axis=-1))
    flat_phi_gen=ak.to_numpy(ak.flatten(matched_gen.phi,axis=-1))

    if var=="pT":
        indices = np.digitize(flat_pt_gen, bin_edges) - 1
    
    if var == 'pT_eta':
       indices = np.digitize(np.abs(flat_eta_gen), bin_edges) - 1
    
    if var== "pT_phi":
        indices = np.digitize(flat_phi_gen, bin_edges) - 1

    if var== "phi":
        indices = np.digitize(flat_phi_gen, bin_edges) - 1

    if var== "eta":
        indices = np.digitize(np.abs(flat_eta_gen), bin_edges) - 1

    resp_simul, err_resp_simul, resol_simul, err_resol_simul = {}, {}, {}, {}
    for index in range(bin_n):
      bin_idx = np.where(indices == index)[0]
      resp_bin_simul =flat_pt[bin_idx]/flat_pt_gen[bin_idx] if var=='pT' else \
                      flat_pt[bin_idx]/flat_pt_gen[bin_idx] if var=='pT_eta' else \
                      flat_pt[bin_idx]/flat_pt_gen[bin_idx] if var=='pT_phi' else \
                      flat_eta[bin_idx]-flat_eta_gen[bin_idx] if var=="eta" else \
                      flat_phi[bin_idx]-flat_phi_gen[bin_idx] 
                      

      resp_simul[index]     = np.mean(resp_bin_simul) if len(resp_bin_simul)>0 else 0
      err_resp_simul[index] = np.std(resp_bin_simul)/np.sqrt(len(resp_bin_simul)) if len(resp_bin_simul)>0 else 0
      if var=="phi":
          resp_bin_simul = (resp_bin_simul + np.pi) % (2*np.pi) - np.pi
          resp_simul[index]     = np.mean(resp_bin_simul) if len(resp_bin_simul)>0 else 0
          err_resp_simul[index] = np.std(resp_bin_simul)/np.sqrt(len(resp_bin_simul)) if len(resp_bin_simul)>0 else 0
          resol_simul[index]     = np.std(resp_bin_simul) if len(resp_bin_simul)>1 else 0
          err_resol_simul[index] = np.std(resp_bin_simul)/(np.sqrt(2*len(resp_bin_simul)-2)) if len(resp_bin_simul)>1 else 0

      if var=="pT" or var=="pT_eta" or var=="pT_phi":
         resol_simul[index]     = np.std(resp_bin_simul)/np.abs(np.mean(resp_bin_simul)) if len(resp_bin_simul)>1 else 0
         err_resol_simul[index] = np.std(resp_bin_simul)/(np.sqrt(2*len(resp_bin_simul)-2)*np.mean(resp_bin_simul)) if len(resp_bin_simul)>1 else 0
    
      if var=="eta":
         resol_simul[index]     = np.std(resp_bin_simul) if len(resp_bin_simul)>1 else 0
         err_resol_simul[index] = np.std(resp_bin_simul)/(np.sqrt(2*len(resp_bin_simul)-2)) if len(resp_bin_simul)>1 else 0
    

      if args.eff_rms and (var == 'pT' or var == 'pT_eta' or var == 'pT_phi'):
        eff_rms = effrms(resp_bin_simul) if len(resp_bin_simul)>1 else [0]
        resol_simul[index]     = eff_rms/np.mean(resp_bin_simul) if len(resp_bin_simul)>1 else 0
        err_resol_simul[index] = eff_rms/(np.sqrt(2*len(resp_bin_simul)-2)) if len(resp_bin_simul)>1 else 0

    return resp_simul, err_resp_simul, resol_simul, err_resol_simul, bin_edges, indices, resp_bin_simul


def model(x, a, c):
    return a + c / x

def plot_responses(simul, gen, args, var, ax, label, event, att_eta, color, bin_n=10, range_=[0,200]):
    resp_simul, err_resp_simul, resol_simul, err_resol_simul, bin_edges, indices, resp_bin_simul= compute_responses_performance(simul, gen, args, var , event, att_eta, bin_n, range_)
    if args.response:
        plt.style.use(mplhep.style.CMS)
        ax.errorbar((bin_edges[1:] + bin_edges[:-1])/2, resp_simul.values(), 
                    yerr=np.array(list(zip(err_resp_simul.values(), err_resp_simul.values()))).T,
                    xerr=(bin_edges[1] - bin_edges[0])/2, ls='None', lw=2, marker='s', label=label, color=color)
        if args.fit:
            if (label == "0p03" or label == "0p045" or label == "Ref") and var=="pT":
                x_data = (bin_edges[1:] + bin_edges[:-1]) / 2
                print(x_data)
                y_data = np.array(list(resp_simul.values()))
                y_err  = np.array(list(err_resp_simul.values()))
                popt, pcov = curve_fit(model, x_data, y_data, sigma=y_err, absolute_sigma=True)
                a_fit, c_fit = popt
                err_a, err_c= np.sqrt(np.diag(pcov))
                print(f"Fitted parameters for {label}: a = {a_fit:.3f} + {err_a:.3f} , c = {c_fit:.3f} + {err_c:.3f} ")
                x_fit = np.linspace(min(x_data), max(x_data), 1000)
                y_fit = model(x_fit, a_fit, c_fit)
                ax.plot(x_fit, y_fit, linestyle='--', color=color)
                # plt.ylim(0.8,1)
            if label == "0p03" and var=="pT":
                x_curve = np.linspace(range_[0] + 1e-3, range_[1], 1000)
                y_curve = 0.95 + 0.5 / x_curve
                ax.plot(x_curve, y_curve, linestyle='--', color='grey', label=f'Fit: 0.95 + 0.5/Pt_gen')
        plt.ylabel(r'$<\phi^{cluster}-\phi^{gen}>$' if var=='phi' else r'$<\eta^{cluster}-\eta^{gen}>$' if var=='eta' else \
                r'$<cluster>$' if var=='n_cl_pt' or var=='n_cl_eta' else r'$<p_{T}^{cluster}/p_{T}^{gen}>$')
        plt.xlabel(r'$p_{T}^{gen}$ [GeV]' if var=='pT' or var=='n_cl_pt' else r'$\phi^{gen}$' if var=='phi' or var=='pT_phi' else r'$|\eta^{gen}|$')
        # if var == "pT":
            # plt.ylim(0.8,1.0)
        # if var == "pT_eta":
        #     plt.ylim(0.8,1.0)
        # if var=="eta":
        #     plt.ylim(-0.0005,0.0005)
        # if var=="phi":
        #     plt.ylim(-0.0010,0.0020)
    if args.resolution or args.eff_rms:
        plt.style.use(mplhep.style.CMS)
        ax.errorbar((bin_edges[1:] + bin_edges[:-1])/2, resol_simul.values(), 
                    yerr=np.array(list(zip(err_resol_simul.values(), err_resol_simul.values()))).T,
                    xerr=(bin_edges[1] - bin_edges[0])/2, ls='None', lw=2, marker='s', label=label, color=color)
        if args.eff_rms:
            plt.ylabel(r'$\sigma^{cluster}$' if var=='phi' else r'$\sigma^{cluster}$' if var=='eta' else \
                    r'$(\sigma^{cluster}/\mu^{cluster})_{eff-RMS}$')
        else:
            plt.ylabel(r'$\sigma^{cluster}$' if var=='phi' else r'$\sigma^{cluster}$' if var=='eta' else \
                    r'$\sigma^{cluster}/\mu^{cluster}$')
        plt.xlabel(r'$p_{T}^{gen}$ [GeV]' if var=='pT' or var=='n_cl_pt' else r'$\phi^{gen}$' if var=='phi' or var=='pT_phi' else r'$|\eta^{gen}|$')
        # if var == "pT":
        #     plt.ylim(0.02,0.14)
        # if var == "pT" and args.eff_rms:
        #     plt.ylim(0.005,0.060)
        # if var == "pT_eta":
        #     plt.ylim(0.02,0.11)
        # if var == "pT_eta" and args.eff_rms:
        #     plt.ylim(0.010,0.028)
        # if var=="eta":
        #     plt.ylim(0.0,0.006)
        # if var=="phi":
        #     plt.ylim(0.001,0.010)
    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup+' '+args.particles)
        
    if args.pt_cut != 0.0 and not args.gen_pt_cut != 0.0:
        thr = args.pt_cut
        plt.legend(title=fr"$p_T^{{\mathrm{{cluster}}}} > {thr} GeV$", title_fontsize=15, fontsize=17)
    if args.gen_pt_cut != 0.0 and not args.pt_cut != 0.0:
        thr = args.gen_pt_cut
        plt.legend(title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$", title_fontsize=15, fontsize=17)
    if args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
        thr = args.gen_pt_cut
        thr2 = args.pt_cut
        plt.legend(title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$" + "\n" + f"and $p_T^{{\mathrm{{cluster}}}} > {thr2} GeV$", title_fontsize=15, fontsize=17)
    else:
        plt.legend(fontsize=18)
    # if args.pt_cut != 0:
    #     plt.legend(title=fr"$p_T^{{\mathrm{{cluster}}}} > {args.pt_cut}$ GeV", title_fontsize=15, fontsize=17)
    # else:
    #     plt.legend(fontsize=18)
    
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
    print("-------------------")
    print("For triangle size", size)
    print("Number of clusters before matching", len(ak.flatten(getattr(event_cl,att_eta), axis=-1)))
    if args.gen_pt_cut != 0:
        mask_ev = ak.num(getattr(event_cl, att_eta)) > 0
        filtered_events = event_cl[mask_ev]
        filtered_gen = event_gen[mask_ev]
        print("Number of events with clusters", len(filtered_gen))
        print("\n")
        print("Applying pt cut on gen particles of", args.gen_pt_cut, "GeV")
        mask_gen_pt = filtered_gen.genpart_pt > args.gen_pt_cut
        event_gen_pt_cut = filtered_gen[mask_gen_pt]
        mask_gen = ak.num(event_gen_pt_cut.genpart_pt, axis=-1) > 0
        event_gen = event_gen_pt_cut[mask_gen]
        event_cl = filtered_events[mask_gen]
        print(f"Number of gen particles after gen pt cut {args.gen_pt_cut} GeV", len(ak.flatten(getattr(event_gen,'genpart_pt'), axis=-1)))
        print(f"Number of events after gen pt cut {args.gen_pt_cut} GeV", len(event_gen))
        print(f"Number of clusters after gen pt cut {args.gen_pt_cut} GeV", len(ak.flatten(getattr(event_cl,att_eta), axis=-1)))
    pair_cluster_matched, pair_gen_masked, clusters_filtered, gen_filtered = ev.apply_matching(event_cl, att_eta, att_phi, event_gen, args, deltaR=deltaR)
    print("Number of events after matching",len(pair_cluster_matched.pt))
    print("Number of matched clusters:", len(ak.flatten(pair_cluster_matched.pt,axis=-1)))
    print("Number of particles after matching",len(ak.flatten(pair_gen_masked.pt,axis=-1)))
    if args.gen_pt_cut != 0 and args.pt_cut != 0:
        print("Denominator is the number of particle after the gen cut only:", len(ak.flatten(event_gen.genpart_pt, axis=-1)))
    print("Total efficiency at particle level:", len(ak.flatten(pair_gen_masked.pt,axis=-1)) / len(ak.flatten(event_gen.genpart_pt, axis=-1)) * 100)
    print("Total efficiency at event level:", len(pair_gen_masked) / len(event_gen.genpart_pt) * 100)
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
        



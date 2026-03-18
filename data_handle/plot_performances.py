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
import data_handling.event_performances as ev
from scipy.optimize import lsq_linear

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
            "Ecalib": events.Ecalib,
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
    elif var == "Ecalib":
        plt.hist(np.abs(ak.to_numpy(event_flat.Ecalib)), bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(np.abs(ak.to_numpy(event_flat.Ecalib)), bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
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
    if args.pt_cut != 0.0 and  args.gen_pt_cut == 0.0:
        thr = args.pt_cut
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
        plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{cluster}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black')
    elif args.gen_pt_cut != 0.0 and args.pt_cut == 0.0:
        thr = args.gen_pt_cut
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
        plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black',title_fontsize=14, fontsize=15)
    elif args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
        thr = args.gen_pt_cut
        thr2 = args.pt_cut
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))
        plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$" + "\n" + f"and $p_T^{{\mathrm{{cluster}}}} > {thr2} GeV$",frameon=True, facecolor='white', edgecolor='black', title_fontsize=10)
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
    if var1=="phi" and var2=="pt":
        plt.hist2d(ak.to_numpy(event_flat.phi), ak.to_numpy(event_flat.pt), bins=40, range=range_, cmap='magma_r')
        plt.xlabel(r'$\phi^{cluster}$')
        plt.ylabel(r'$p_T^{cluster}$')
        plt.colorbar(label="Counts")
    if var1=="eta" and var2=="delta_r":
        plt.hist2d(ak.to_numpy(event_flat.eta), ak.to_numpy(event_flat.delta_r), bins=40, range=range_, cmap='magma_r')
        plt.ylabel(r'$\Delta R (cluster, gen)$')
        plt.xlabel(r'$\eta^{cluster}$')
        plt.colorbar(label="Counts")
    if var1=="eta" and var2=="phi":
        plt.hist2d(ak.to_numpy(event_flat.phi), ak.to_numpy(event_flat.eta), bins=40, range=range_, cmap='magma_r')
        plt.xlabel(r'$\phi^{cluster}$')
        plt.ylabel(r'$\eta^{cluster}$')
        plt.colorbar(label="Counts")

    if args.pt_cut != 0.0 and not args.gen_pt_cut != 0.0:
        if label != "Ref":
            plt.title(f"Algorithm with triangle size: {label} and "+r'$p_t^{cluster}>$'+f"{args.pt_cut} GeV",fontsize=18)
        else:
            plt.title(f"CMSSW simulation and "+r'$p_t^{cluster}>$'+f"{args.pt_cut} GeV",fontsize=18)
    if args.gen_pt_cut != 0.0 and not args.pt_cut != 0.0:
        if label != "Ref":
            plt.title(f"Algorithm with triangle size: {label} and "+r'$p_t^{gen}>$'+f"{args.gen_pt_cut} GeV",fontsize=18)
        else:
            plt.title(f"CMSSW simulation and "+r'$p_t^{gen}>$'+f"{args.gen_pt_cut} GeV",fontsize=18)
    if args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
        if label != "Ref":
            plt.title(f"Algorithm with triangle size: {label} and "+r'$p_t^{gen}>$'+f"{args.gen_pt_cut} GeV and "+r'$p_t^{cluster}>$'+f"{args.pt_cut} GeV",fontsize=14)
        else:
            plt.title(f"CMSSW simulation and "+r'$p_t^{gen}>$'+f"{args.gen_pt_cut} GeV and "+r'$p_t^{cluster}>$'+f"{args.pt_cut} GeV",fontsize=14)
    else:
        if label != "Ref":
            plt.title(f"Algorithm with triangle size: {label}",fontsize=20)
        else:
            plt.title(f"CMSSW simulation",fontsize=20)
    # if args.pt_cut !=0 :
    #     if label != "Ref":
    #         plt.title(f"Algorithm with triangle size: {label} and "+r'$pt^{cluster}>$'+f"{args.pt_cut} GeV",fontsize=18)
    #     else:
    #         plt.title(f"CMSSW simulation and "+r'$pt^{cluster}>$'+f"{args.pt_cut} GeV",fontsize=18)
    # else:
    #     if label != "Ref":
    #         plt.title(f"Algorithm with triangle size: {label}",fontsize=20)
    #     else:
    #         plt.title(f"CMSSW simulation",fontsize=20)
     
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
    if var == "Ecalib":
        # print(events.pt)
        # print("flattened pt:",ak.flatten(events.pt,axis=-1))
        flat_pt=ak.to_numpy(ak.flatten(events.Ecalib,axis=-1))
        flat_gen=ak.to_numpy(ak.flatten(gen.pt,axis=-1))
        # print("flat pt:",flat_pt)
        scale_simul = np.divide(flat_pt, flat_gen)
        plt.hist(scale_simul, bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(scale_simul, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
    if var == "Ecalib_cal_eta":
        # print(events.pt)
        # print("flattened pt:",ak.flatten(events.pt,axis=-1))
        flat_pt=ak.to_numpy(ak.flatten(events.Ecalib_PU_term,axis=-1))
        flat_gen=ak.to_numpy(ak.flatten(gen.pt,axis=-1))
        # print("flat pt:",flat_pt)
        scale_simul = np.divide(flat_pt, flat_gen)
        plt.hist(scale_simul, bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(scale_simul, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
    if var == "Ecalib_all_":
        # print(events.pt)
        # print("flattened pt:",ak.flatten(events.pt,axis=-1))
        flat_pt=ak.to_numpy(ak.flatten(events.Ecalib_all_,axis=-1))
        flat_gen=ak.to_numpy(ak.flatten(gen.pt,axis=-1))
        # print("flat pt:",flat_pt)
        scale_simul = np.divide(flat_pt, flat_gen)
        plt.hist(scale_simul, bins=bin_edges, alpha=0.2, color=color, histtype='stepfilled')
        plt.hist(scale_simul, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
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
    elif args.gen_pt_cut != 0.0 and not args.pt_cut != 0.0:
        thr = args.gen_pt_cut
        plt.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black', loc="best", title_fontsize=15, fontsize=17)
    elif args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
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
        flat_Ecalib = ak.to_numpy(ak.flatten(matched_ev.Ecalib, axis=-1))
        flat_pt_gen = ak.to_numpy(ak.flatten(matched_gen.pt, axis=-1))
        flat_eta = ak.to_numpy(ak.flatten(matched_ev.eta, axis=-1))
        flat_eta_gen = ak.to_numpy(ak.flatten(matched_gen.eta, axis=-1))
        flat_phi = ak.to_numpy(ak.flatten(matched_ev.phi, axis=-1))
        flat_phi_gen = ak.to_numpy(ak.flatten(matched_gen.phi, axis=-1))

        results.append((resp_bin_simul, indices, label, color,
                        flat_pt, flat_Ecalib, flat_pt_gen, flat_eta, flat_eta_gen, flat_phi, flat_phi_gen))

  
    for i in range(bin_n):
        plt.figure(figsize=(10, 10))
        plt.style.use(mplhep.style.CMS)

        # Bin label
        if var in ["pT","Ecalib"]:
            bin_label = fr"${bin_edges[i]:.1f} < p_T^{{\mathrm{{gen}}}} < {bin_edges[i+1]:.1f}\,\mathrm{{GeV}}$"
            # bin_nb=[np.min(flat_pt),np.max(flat_pt)]
        elif var in ["eta", "pT_eta","Ecalib_PU_term"]:
            bin_label = fr"${bin_edges[i]:.2f} < |\eta|^{{\mathrm{{gen}}}} < {bin_edges[i+1]:.2f}$"
        elif var in ["phi", "pT_phi","Ecalib_phi"]:
            bin_label = fr"${bin_edges[i]:.2f} < \phi^{{\mathrm{{gen}}}} < {bin_edges[i+1]:.2f}$"
        else:
            bin_label = f"Bin {i}"

        legend_handles = []

        # Plot for each dataset
        for resp_bin_simul, indices, label, color, flat_Ecalib, flat_pt, flat_pt_gen, flat_eta, flat_eta_gen, flat_phi, flat_phi_gen in results:
            bin_idx = np.where(indices == i)[0]
            if len(bin_idx) == 0:
                continue

            if var in ["Ecalib", "Ecalib_PU_term", "Ecalib_phi"]:
                ratio = flat_Ecalib[bin_idx] / flat_pt_gen[bin_idx]
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

            plt.hist(ratio, bins=bin_nb,
                     alpha=0.2, color=color, histtype='stepfilled')
            plt.hist(ratio, bins=bin_nb, 
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



def compute_responses_performance(matched, matched_gen, args, var, bin_n=10, range_=[0,200],name=""):
    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)

    flat_pt=ak.to_numpy(ak.flatten(matched.pt,axis=-1))
    if  "Ecalib" in var:
        flat_Ecalib=ak.to_numpy(ak.flatten(getattr(matched, f'Ecalib_{name}'),axis=-1))
    else:
        flat_Ecalib = None
    # flat_Ecalib_PU_term=ak.to_numpy(ak.flatten(matched.Ecalib_PU_term,axis=-1))
    # flat_Ecalib_all=ak.to_numpy(ak.flatten(getattr(matched, f'Ecalib_all_{name}'),axis=-1))
    flat_pt_gen=ak.to_numpy(ak.flatten(matched_gen.pt,axis=-1))

    flat_eta=ak.to_numpy(ak.flatten(matched.eta,axis=-1))
    flat_eta_gen=ak.to_numpy(ak.flatten(matched_gen.eta,axis=-1))

    flat_phi=ak.to_numpy(ak.flatten(matched.phi,axis=-1))
    flat_phi_gen=ak.to_numpy(ak.flatten(matched_gen.phi,axis=-1))

    # if var in ["pT", "Ecalib", "Ecalib_cal_eta", f'Ecalib_all_{name}']:
    #     print("got in")
    #     indices = np.digitize(flat_pt_gen, bin_edges) - 1

    # if var in ['pT_eta', 'Ecalib_PU_term', 'Ecalib_PU_term_eta',f'Ecalib_all_eta_{name}']:
    #    indices = np.digitize(np.abs(flat_eta_gen), bin_edges) - 1
    
    # if var in ['pT_phi', 'Ecalib_phi', 'Ecalib_PU_term_phi',f'Ecalib_all_phi_{name}']:
    #     indices = np.digitize(flat_phi_gen, bin_edges) - 1

    # if var== "phi":
    #     indices = np.digitize(flat_phi_gen, bin_edges) - 1

    # if var== "eta":
    #     indices = np.digitize(np.abs(flat_eta_gen), bin_edges) - 1

    # Generic, robust way
    print("var", var)
    if "eta" in var:
        # print("Hi")
        indices = np.digitize(np.abs(flat_eta_gen), bin_edges) - 1
    elif "phi" in var:
        # print("Hi2")
        indices = np.digitize(flat_phi_gen, bin_edges) - 1
    else:
        indices = np.digitize(flat_pt_gen, bin_edges) - 1

    print("range_:", range_)
    print("bin_edges:", bin_edges)
    print("flat_pt_gen min/max:", flat_pt_gen.min(), flat_pt_gen.max())
    # print("flat_Ecalib_all length:", len(flat_Ecalib_all))
    print("indices unique:", indices)

    resp_simul, err_resp_simul, resol_simul, err_resol_simul = {}, {}, {}, {}
    for index in range(bin_n):
      bin_idx = np.where(indices == index)[0]
      print(f"index {index}: bin_idx length = {len(bin_idx)}")
    #   resp_bin_simul =flat_pt[bin_idx]/flat_pt_gen[bin_idx] if var=='pT' else \
    #                   flat_pt[bin_idx]/flat_pt_gen[bin_idx] if var=='pT_eta' else \
    #                   flat_pt[bin_idx]/flat_pt_gen[bin_idx] if var=='pT_phi' else \
    #                   flat_Ecalib[bin_idx]/flat_pt_gen[bin_idx] if var=='Ecalib' else \
    #                   flat_Ecalib[bin_idx]/flat_pt_gen[bin_idx] if var=='Ecalib_eta' else \
    #                   flat_Ecalib[bin_idx]/flat_pt_gen[bin_idx] if var=='Ecalib_phi' else \
    #                   flat_Ecalib_PU_term[bin_idx]/flat_pt_gen[bin_idx] if var=='Ecalib_PU_term_' else \
    #                   flat_Ecalib_PU_term[bin_idx]/flat_pt_gen[bin_idx] if var=='Ecalib_PU_term_eta' else \
    #                   flat_Ecalib_PU_term[bin_idx]/flat_pt_gen[bin_idx] if var=='Ecalib_PU_term_phi' else \
    #                   flat_Ecalib_all[bin_idx]/flat_pt_gen[bin_idx] if 'Ecalib_all_' in var  else \
    #                   flat_Ecalib_all[bin_idx]/flat_pt_gen[bin_idx] if 'Ecalib_all_eta_' in var else \
    #                   flat_Ecalib_all[bin_idx]/flat_pt_gen[bin_idx] if 'Ecalib_all_phi_' in var else \
    #                   flat_eta[bin_idx]-flat_eta_gen[bin_idx] if var=="eta" else \
    #                   flat_phi[bin_idx]-flat_phi_gen[bin_idx] 
    #   resp_bin_simul =flat_pt[bin_idx]/flat_pt_gen[bin_idx] if var=='pT' else \
    #                   flat_Ecalib[bin_idx]/flat_pt_gen[bin_idx] if var=='Ecalib' else \
    #                   flat_Ecalib_PU_term[bin_idx]/flat_pt_gen[bin_idx] if var=='Ecalib_PU_term_' else \
    #                   flat_Ecalib_all[bin_idx]/flat_pt_gen[bin_idx] if 'Ecalib_all_' in var  else \
    #                   flat_eta[bin_idx]-flat_eta_gen[bin_idx] if "eta" in var else \
    #                   flat_phi[bin_idx]-flat_phi_gen[bin_idx] 
      resp_bin_simul =flat_pt[bin_idx]/flat_pt_gen[bin_idx] if var=='pT' else \
                      flat_Ecalib[bin_idx]/flat_pt_gen[bin_idx] if var==f'Ecalib_{name}' else \
                      flat_eta[bin_idx]-flat_eta_gen[bin_idx] if "eta" in var else \
                      flat_phi[bin_idx]-flat_phi_gen[bin_idx] 
                      
    #   print(resp_bin_simul)
      resp_simul[index]     = np.mean(resp_bin_simul) if len(resp_bin_simul)>0 else 0
      err_resp_simul[index] = np.std(resp_bin_simul)/np.sqrt(len(resp_bin_simul)) if len(resp_bin_simul)>0 else 0
      if var=="phi":
          resp_bin_simul = (resp_bin_simul + np.pi) % (2*np.pi) - np.pi
          resp_simul[index]     = np.mean(resp_bin_simul) if len(resp_bin_simul)>0 else 0
          err_resp_simul[index] = np.std(resp_bin_simul)/np.sqrt(len(resp_bin_simul)) if len(resp_bin_simul)>0 else 0
          resol_simul[index]     = np.std(resp_bin_simul) if len(resp_bin_simul)>1 else 0
          err_resol_simul[index] = np.std(resp_bin_simul)/(np.sqrt(2*len(resp_bin_simul)-2)) if len(resp_bin_simul)>1 else 0

      if var in ["pT", "Ecalib", "Ecalib_cal_eta", f"Ecalib_all_{name}", 'pT_eta', 'Ecalib_PU_term', 'Ecalib_PU_term_eta', f'Ecalib_all_eta_{name}','pT_phi', 'Ecalib_phi', 'Ecalib_PU_term_phi', f'Ecalib_all_phi_{name}']:
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

def plot_responses(simul, gen, args, var, ax, label, color, bin_n=10, range_=[0,200],name=""):
    resp_simul, err_resp_simul, resol_simul, err_resol_simul, bin_edges, indices, resp_bin_simul= compute_responses_performance(simul, gen, args, var , bin_n, range_, name)
    # print(resp_simul)
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
        plt.xlabel(r'$p_{T}^{gen}$ [GeV]' if var=='pT' or var=='n_cl_pt' or var==f'Ecalib_{name}' or var==f'Ecalib_PU_term{name}' or var==f'Ecalib_all_{name}'  else r'$\phi^{gen}$' if var=='phi' or var=='pT_phi' or var=='Ecalib_phi' or var=='Ecalib_PU_term_phi' or var==f'Ecalib_all_phi_{name}'  else r'$|\eta^{gen}|$')
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
        plt.xlabel(r'$p_{T}^{gen}$ [GeV]' if var=='pT' or var=='n_cl_pt' or var=='Ecalib' or var=='Ecalib_PU_term' or var==f'Ecalib_all_{name}' else r'$\phi^{gen}$' if var=='phi' or var=='pT_phi' or var=='Ecalib_phi' or var=='Ecalib_PU_term_phi' or var==f'Ecalib_all_phi_{name}' else r'$|\eta^{gen}|$')
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
    elif args.gen_pt_cut != 0.0 and args.pt_cut == 0.0:
        thr = args.gen_pt_cut
        plt.legend(title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$", title_fontsize=15, fontsize=17)
    elif args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
        thr = args.gen_pt_cut
        thr2 = args.pt_cut
        plt.legend(title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$" + "\n" + f"and $p_T^{{\mathrm{{cluster}}}} > {thr2} GeV$", title_fontsize=15, fontsize=17)
    else:
        plt.legend(fontsize=18)
    # if args.pt_cut != 0:
    #     plt.legend(title=fr"$p_T^{{\mathrm{{cluster}}}} > {args.pt_cut}$ GeV", title_fontsize=15, fontsize=17)
    # else:
    #     plt.legend(fontsize=18)
    
    plt.grid(linestyle=":")
    plt.tight_layout()


#Matching function (not used for the moment)

def calcDeltaR(eta_cl, phi_cl, gen):

    deta = np.abs(eta_cl - gen.eta)
    dphi = np.abs(phi_cl - gen.phi)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    # If condition is true choose 2 * np.pi - dphi, else choose dphi
    
    deltaR = np.sqrt(deta**2 + dphi**2)
    return deltaR




def derive_calibration (cluster, gen, args, lower_bound=-np.inf, upper_bound=np.inf, name=""):
    # The information of the pT per layer is given in 34 trigger layer
    # We are only interested in the CE-E layers because we are calibrating EM objects, which in trigger layers translates as taking 1 every 2 detector layers
    # There are 26 electromagnetic detector layers, corresponding to 26/2=13 first trigger layers
    if args.PU0calibration:
        cluster_flat= ak.flatten(cluster.layer_pt, axis=1)[:,1:13]
        gen_flat= ak.flatten(gen.pt,axis=-1)
        cluster_np = np.asarray(cluster_flat)
        gen_np = np.asarray(gen_flat)
        regression = lsq_linear(cluster_np, gen_np, bounds=(lower_bound, upper_bound), lsmr_tol='auto', method='bvls')
        weight = regression.x
    if args.PU200calibration:
        cluster_flat= ak.flatten(cluster.layer_pt, axis=1)[:,:13]
        cluster_eta= ak.flatten(cluster.eta, axis=1)
        gen_flat= ak.flatten(gen.pt,axis=-1)
        cluster_np = np.asarray(cluster_flat)
        cluster_eta_np = -np.abs(np.asarray(cluster_eta))
        N = cluster_np.shape[0]
        A = np.hstack([
            cluster_np,
            cluster_eta_np[:, None],     
            -1*np.ones((N, 1))            
        ])
        gen_np = np.asarray(gen_flat)
        regression = lsq_linear(A, gen_np, bounds=(lower_bound, upper_bound), lsmr_tol='auto', method='bvls')
        weight = regression.x
    if args.PileUpcalibration:
        # Defining Target vector
        cluster_flat_Ecorr= ak.flatten(getattr(cluster, f"Ecalib_{name}"),axis=-1)
        gen_flat_E= ak.flatten(gen.pt,axis=-1)
        cluster_np = np.asarray(cluster_flat_Ecorr)
        gen_np = np.asarray(gen_flat_E)
        residual= gen_np - cluster_np

        #defining matrix
        cluster_eta= ak.flatten(cluster.eta, axis=1)
        cluster_eta_np = -np.abs(np.asarray(cluster_eta))
        N = cluster_eta_np.shape[0]
        A = np.hstack([
            cluster_eta_np[:, None],     
            -1*np.ones((N, 1))            
        ])
        regression = lsq_linear(A, residual, bounds=(lower_bound, upper_bound), lsmr_tol='auto', method='bvls')
        weight = regression.x

    if args.PileUpcalibrationFinal:
        # Defining Target vector
        cluster_flat_Ecorr= ak.flatten(getattr(cluster, f"Ecalib_{name}"),axis=-1)
        gen_flat_E= ak.flatten(gen.pt,axis=-1)
        cluster_np = np.asarray(cluster_flat_Ecorr)
        gen_np = np.asarray(gen_flat_E)
        residual= gen_np - cluster_np

        #defining matrix
        cluster_eta= ak.flatten(cluster.eta, axis=1)
        cluster_eta_np = -np.abs(np.asarray(cluster_eta))
        N = cluster_eta_np.shape[0]
        A = np.hstack([
            cluster_eta_np[:, None],     
            -1*np.ones((N, 1))            
        ])
        regression = lsq_linear(A, residual, bounds=(lower_bound, upper_bound), lsmr_tol='auto', method='bvls')
        weight = regression.x
    return weight

def apply_calibration(cluster, weights,name=""):
    num_cluster = ak.num(cluster.layer_pt,axis=1)
    flat = ak.flatten(cluster.layer_pt,axis=1)
    flat_numpy=ak.to_numpy(flat)[:,:13]
    # Weight array per layer (length 34 for example)
    weights = np.array(weights)  
    if len(weights) == 12:
        flat_numpy=ak.to_numpy(flat)[:,1:13]
        Ecalib=ak.unflatten(ak.sum(flat_numpy*weights, axis=1) ,num_cluster)
    else:
        Ecalib=ak.unflatten(ak.sum(flat_numpy*weights, axis=1),num_cluster)
    # Ecalib=ak.unflatten(ak.sum(flat_numpy*weights[:12], axis=1) - np.abs(flat_eta_numpy)*weights[12:13] + weights[13] ,num_cluster)
    # print(getattr(cluster, f"Ecalib_all_{name}"))

    cluster= ak.with_field(cluster, Ecalib, f"Ecalib_{name}")
    # print(f"Ecalib_{name}",weights)
    return cluster

def apply_calibration_eta(cluster, weights, name="", name1=""):
    num_cluster = ak.num(getattr(cluster, f"Ecalib_{name}"),axis=-1)
    # print("num_cluster", num_cluster)
    flat_Ecalib = ak.flatten(getattr(cluster, f"Ecalib_{name}"),axis=1)
    # print("flat_Ecalib", flat_Ecalib)
    flat_eta = ak.flatten(cluster.eta,axis=1)
    # print("flat_eta", flat_eta)
    flat_Ecalib_numpy=ak.to_numpy(flat_Ecalib)
    flat_eta_numpy=ak.to_numpy(flat_eta)
    flat_weights_numpy=ak.to_numpy(ak.flatten(weights,axis=-1))
    print(flat_weights_numpy)
    # Weight array per layer (length 34 for example)
    weights = np.array(weights)  
    Ecalib_PU_term=ak.unflatten(flat_Ecalib_numpy - np.abs(flat_eta_numpy)*flat_weights_numpy[0] - flat_weights_numpy[1],num_cluster)
    cluster= ak.with_field(cluster, Ecalib_PU_term, f"Ecalib_PU_term_{name1}")
    return cluster
        
def apply_calibration_all_weights(cluster, weights, name=""):
    num_cluster = ak.num(cluster.layer_pt,axis=1)
    flat = ak.flatten(cluster.layer_pt,axis=1)
    flat_eta = ak.flatten(cluster.eta,axis=1)
    flat_numpy=ak.to_numpy(flat)[:,:13]
    flat_eta_numpy=ak.to_numpy(flat_eta)
    # Weight array per layer (length 34 for example)
    # print(len(weights))
    flat_eta_numpy=ak.to_numpy(flat_eta)
    if len(weights) == 14:
        flat_numpy=ak.to_numpy(flat)[:,1:13]
        Ecalib=ak.unflatten(ak.sum(flat_numpy*weights[:12], axis=1) - np.abs(flat_eta_numpy)*weights[12] - weights[13] ,num_cluster)
    else:
        Ecalib=ak.unflatten(ak.sum(flat_numpy*weights[:13], axis=1) - np.abs(flat_eta_numpy)*weights[13] - weights[14] ,num_cluster)
    # Ecalib=ak.unflatten(ak.sum(flat_numpy*weights[:12], axis=1) - np.abs(flat_eta_numpy)*weights[12:13] + weights[13] ,num_cluster)
    cluster= ak.with_field(cluster, Ecalib, f"Ecalib_all_{name}")
    # print(getattr(cluster, f"Ecalib_all_{name}"))
    
    return cluster

def compute_response_calib(
    matched,
    matched_gen,
    args,
    response_variable="pt",   # pt, eta, phi
    bin_variable="pt",        # pt, eta, phi
    quantity="raw",
    calib_name=None,
    bin_n=10,
    range_=(0, 200),
):

    bin_edges = np.linspace(range_[0], range_[1], bin_n + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # -------------------------
    # Flatten
    # -------------------------
    pt      = ak.to_numpy(ak.flatten(matched.pt, axis=-1))
    pt_gen  = ak.to_numpy(ak.flatten(matched_gen.pt, axis=-1))

    eta     = ak.to_numpy(ak.flatten(matched.eta, axis=-1))
    eta_gen = ak.to_numpy(ak.flatten(matched_gen.eta, axis=-1))

    phi     = ak.to_numpy(ak.flatten(matched.phi, axis=-1))
    phi_gen = ak.to_numpy(ak.flatten(matched_gen.phi, axis=-1))

    if quantity == "calibPU0":
        energy = ak.to_numpy(
            ak.flatten(getattr(matched, f"Ecalib_{calib_name}"))
        )
    elif quantity == "calibPU200":
        energy = ak.to_numpy(
            ak.flatten(getattr(matched, f"Ecalib_all_{calib_name}"))
        )
    elif quantity == "calibPUterm":
        energy = ak.to_numpy(
            ak.flatten(getattr(matched, f"Ecalib_PU_term_{calib_name}"))
        )
    else:
        energy = pt

    # -------------------------
    # Choose binning variable
    # -------------------------
    if bin_variable == "pt":
        x_for_binning = pt_gen
    elif bin_variable == "eta":
        x_for_binning = np.abs(eta_gen)
    elif bin_variable == "phi":
        x_for_binning = phi_gen
    else:
        raise ValueError("Invalid bin_variable")

    indices = np.digitize(x_for_binning, bin_edges) - 1

    # -------------------------
    # Prepare outputs
    # -------------------------
    response     = np.zeros(bin_n)
    response_err = np.zeros(bin_n)
    resolution   = np.zeros(bin_n)
    resolution_err = np.zeros(bin_n)

    # -------------------------
    # Loop over bins
    # -------------------------
    for i in range(bin_n):

        mask = indices == i
        if not np.any(mask):
            continue

        # ===== RESPONSE VARIABLE =====
        # print(response_variable)
        if response_variable == "pt":
            resp = energy[mask] / pt_gen[mask]

        elif response_variable == "eta":
            resp = eta[mask] - eta_gen[mask]

        elif response_variable == "phi":
            resp = phi[mask] - phi_gen[mask]
            resp = (resp + np.pi) % (2*np.pi) - np.pi

        else:
            raise ValueError("Invalid response_variable")

        # ---- Mean
        response[i] = np.mean(resp)
        response_err[i] = np.std(resp) / np.sqrt(len(resp))

        # ---- Resolution
        if response_variable == "pt":
            # sigma = effrms(resp) if args.eff_rms else np.std(resp)
            sigma= np.std(resp)
            resolution[i] = sigma / np.abs(np.mean(resp))
            resolution_err[i] = sigma / np.sqrt(2*len(resp)-2)
        else:
            sigma = np.std(resp)
            resolution[i] = sigma
            resolution_err[i] = sigma / np.sqrt(2*len(resp)-2)

    return bin_centers, response, response_err, resolution, resolution_err

    
    
def plot_response_calib(
    matched,
    matched_gen,
    args,
    ax,
    label,
    color,
    response_variable="pt",
    bin_variable="pt", 
    quantity="raw",
    calib_name=None,
    bin_n=10,
    range_=(0, 200),
):

    (x,response,response_err,resolution,resolution_err,) = compute_response_calib( matched, matched_gen,
                                                                                    args,response_variable,bin_variable,
                                                                                    quantity, calib_name, bin_n, range_,
                                                                                    )

    plt.style.use(mplhep.style.CMS)

    # -------------------------------------------------
    # Response plot
    # -------------------------------------------------
    if args.response:
        ax.errorbar(
            x,
            response,
            yerr=response_err,
            xerr=(x[1] - x[0]) / 2,
            marker="s",
            ls="None",
            label=label,
            color=color,
        )

        ax.set_ylabel(
            r"$\langle p_T^{cluster}/p_T^{gen} \rangle$"
            if response_variable == "pt"
            else r"$\langle \eta^{cluster} - \eta^{gen} \rangle$"
            if response_variable == "eta"
            else r"$\langle \phi^{cluster} - \phi^{gen} \rangle$"
        )

    # -------------------------------------------------
    # Resolution plot
    # -------------------------------------------------
    if args.resolution:
        ax.errorbar(
            x,
            resolution,
            yerr=resolution_err,
            xerr=(x[1] - x[0]) / 2,
            marker="s",
            ls="None",
            label=label,
            color=color,
        )

        if response_variable == "pt":
            ax.set_ylabel(
                # r"$(\sigma/\mu)_{eff}$"
                # if args.eff_rms
                # else r"$\sigma/\mu$"
                r'$\sigma^{cluster}/\mu^{cluster}$'
            )
        else:
            ax.set_ylabel(r"$\sigma^{cluster}$")

    # -------------------------------------------------
    # X label
    # -------------------------------------------------
    ax.set_xlabel(
        r"$p_T^{gen}$ [GeV]"
        if bin_variable == "pt"
        else r"$|\eta^{gen}|$"
        if bin_variable == "eta"
        else r"$\phi^{gen}$"
    )

    mplhep.cms.label("Preliminary", data=True,
                     rlabel=args.pileup + " " + args.particles)

    ax.legend(fontsize=15)
    ax.grid(linestyle=":")

def scale_distribution_calib(events, gen, args, field, bin_n, range_, label, color, legend_handles, ax):

    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)
    plt.style.use(mplhep.style.CMS)

    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup + ' ' + args.particles)


    if field == "pT":
        numerator = events.pt
        denominator = gen.pt
        xlabel = r'$p_T^{cluster}/p_T^{gen}$'

    elif field == "eta":
        numerator = events.eta
        denominator = gen.eta
        xlabel = r'$\eta^{cluster}-\eta^{gen}$'

    else:
        numerator = ak.flatten(getattr(events, field),axis=-1)
        denominator = ak.flatten(gen.pt,axis=-1)
        xlabel = r'$E^{calib}/p_T^{gen}$'

    flat_num = ak.to_numpy(ak.flatten(numerator, axis=-1))
    flat_den = ak.to_numpy(ak.flatten(denominator, axis=-1))


    mask = flat_den != 0
    flat_num = flat_num[mask]
    flat_den = flat_den[mask]

    if field == "eta" or field == "phi":
        values = flat_num - flat_den
    else:
        values = np.divide(flat_num, flat_den)

    ax.hist(
        values, 
        bins=bin_edges, 
        alpha=0.2, 
        color=color, 
        histtype='stepfilled')
    ax.hist(
        values,
        bins=bin_edges,
        histtype='step',
        linewidth=2.5,
        color=color,
        label=label
    )

    legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, linewidth=4, alpha=0.2, label=label))

    if args.pt_cut != 0.0 and not args.gen_pt_cut != 0.0:
        thr = args.pt_cut
        ax.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{cluster}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black')
    elif args.gen_pt_cut != 0.0 and not args.pt_cut != 0.0:
        thr = args.gen_pt_cut
        ax.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black', loc="best", title_fontsize=15, fontsize=17)
    elif args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
        thr = args.gen_pt_cut
        thr2 = args.pt_cut
        ax.legend(handles=legend_handles, title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$" + "\n" + f"and $p_T^{{\mathrm{{cluster}}}} > {thr2} GeV$",frameon=True, facecolor='white', edgecolor='black', title_fontsize=14)
    else:
        ax.legend(handles=legend_handles)
    
    # plt.hist(values, bins=bin_edges, histtype='step', linewidth=2.5, color=color, label=label)
    ax.set_xlabel(xlabel)
    ax.set_yscale('log')
    ax.grid(linestyle=":")

def plot_weight_val(weights, tri, color, calib_name, calib_idx,args):
    markers = ['o','s','^','D','x','*','v']
    marker = markers[calib_idx]
    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup + ' ' + args.particles)

    w = ak.to_numpy(weights[calib_name][tri])
    n = len(w)

    if n == 15:
        x = np.arange(1, 16)
    elif n == 14:
        x = np.arange(1, n+1) + 1   
    elif n == 13:
        x = np.arange(1, 14)
    elif n ==12:
        x = np.arange(1, n+1) + 1   
    else:
        x = np.arange(1, 3) 


    plt.scatter(
        x,
        w,
        marker=marker,
        color=color,
        label=f"{tri}-{calib_name}"
    )


    plt.xlabel("Weight index")
    plt.ylabel("Weight value")

    if args.pt_cut != 0.0 and not args.gen_pt_cut != 0.0:
        thr = args.pt_cut
        plt.legend(title=fr"$p_T^{{\mathrm{{cluster}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black')
    elif args.gen_pt_cut != 0.0 and not args.pt_cut != 0.0:
        thr = args.gen_pt_cut
        plt.legend( title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$",frameon=True, facecolor='white', edgecolor='black', loc="best", title_fontsize=15, fontsize=17)
    elif args.pt_cut != 0.0 and args.gen_pt_cut != 0.0:
        thr = args.gen_pt_cut
        thr2 = args.pt_cut
        plt.legend( title=fr"$p_T^{{\mathrm{{gen}}}} > {thr} GeV$" + "\n" + f"and $p_T^{{\mathrm{{cluster}}}} > {thr2} GeV$",frameon=True, facecolor='white', edgecolor='black', title_fontsize=14)
    else:
       plt.legend()

    plt.grid(":")

def plot_eta(cluster, weights, color, tri="", name="", idx_0=0, idx_1=1):

    print("weights")
    print(weights)
    weights= ak.flatten(weights, axis =-1)
    print(weights[idx_0])
    print(len(weights))
  
    cluster_eta= ak.flatten(cluster.eta, axis=1)
    cluster_eta_np = np.abs(np.asarray(cluster_eta))
    plt.plot(np.abs(cluster_eta_np), np.abs(cluster_eta_np) * weights[idx_0] + weights[idx_1], color=color, label=f"For {tri} with {name}")
    plt.legend()
    plt.grid(":")
    plt.xlabel(r"$|\eta$|")
    plt.ylabel(r"$\alpha |\eta| + \beta$")
    plt.title("Linear function of eta")
    


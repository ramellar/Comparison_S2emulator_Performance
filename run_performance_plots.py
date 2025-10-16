#Main script to run
import os
import sys
import argparse
import cppyy
import data_handle.plot_performances as plot
from   data_handle.event_performances import provide_events_performaces, apply_matching
import matplotlib
import matplotlib.pyplot as plt
import mplhep
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import json
plt.style.use(mplhep.style.CMS)
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import awkward as ak

colors = ["tab:olive", "tab:cyan", "darkorchid" , "darkorange", "deeppink"]
my_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", colors)
n_colors = len(colors)


if __name__ == '__main__':
  ''' python run_performance_plots.py -n 2 --pileup PU0 --particles photons '''

  parser = argparse.ArgumentParser(description='Stage-2 Emulator Parameters')
  parser.add_argument('-n',          type=int, default=1,         help='Provide the number of events')
  parser.add_argument('--particles', type=str, default='photons', help='Choose the particle sample')
  parser.add_argument('--pileup',    type=str, default='PU0',     help='Choose the pileup - PU0 or PU200')
  parser.add_argument('--tag',       type=str, default='',        help='Name to make unique json files')
  parser.add_argument('--var',       type=str, default='pT',        help='Name of the variable to be plotted')
  parser.add_argument('--parquet',    action='store_true', help='Produce parquet file to plot')
  parser.add_argument('--pt_cut',    type=float, default=0,         help='Provide the cut for the pt')
  parser.add_argument('--gen_pt_cut',    type=float, default=0,         help='Provide the cut for the gen pt')
  #plotting arguments
  parser.add_argument('--eff_rms',     action='store_true', help='Extract mean and std from selecting the shorted interval containing 68& events')
  parser.add_argument('--total_efficiency',     action='store_true', help='Compute the total efficiency for each emulation test')
  parser.add_argument('--resp',     action='store_true', help='Plot the response and resolution of the emulator')
  parser.add_argument('--resolution',     action='store_true', help='Extract mean and std from selecting the shorted interval containing 68& events')
  parser.add_argument('--response',     action='store_true', help='Extract mean and std from selecting the shorted interval containing 68& events')
  parser.add_argument('--distribution',     action='store_true', help='Plot the distributions')
  parser.add_argument('--scale',     action='store_true', help='Plot the response distribution of the emulator')
  parser.add_argument('--efficiency',     action='store_true', help='Plot the response distribution of the emulator')
  parser.add_argument('--fit',     action='store_true', help='Fit Pt response')
  parser.add_argument('--bins',     action='store_true', help='Fit Pt response')
  args = parser.parse_args()

  output_dir = f"results_performance_plots_{args.particles}_{args.pileup}"
  os.makedirs(output_dir, exist_ok=True)

  #if we want to plot different cuts in the same plot it is probably easier to save the arrays in a parquet file and plot from there
  if args.pt_cut:
    output_dir = f"results_performance_plots_{args.particles}_{args.pileup}_cut_{args.pt_cut}_GeV"
    os.makedirs(output_dir, exist_ok=True)

  if args.gen_pt_cut:
    output_dir = f"results_performance_plots_{args.particles}_{args.pileup}_gen_cut_{args.gen_pt_cut}_GeV"
    os.makedirs(output_dir, exist_ok=True)

  if args.gen_pt_cut and args.pt_cut:
    output_dir = f"results_performance_plots_{args.particles}_{args.pileup}_gen_cut_{args.gen_pt_cut}_GeV_cut_{args.pt_cut}_GeV"
    os.makedirs(output_dir, exist_ok=True)

  #awkward arrays containing the cluster and gen information
  # Since the dataset is too heavy to load the events every time, we just need to load them once and save them as parquet files
  if args.parquet:
    events_gen, events_0p0113, events_0p016, events_0p03 , events_0p045, events_Ref =provide_events_performaces(args.n, args.particles, args.pileup, args.pt_cut)

    output_dir = "/eos/home-r/ramellar/parquet_files/" + args.particles + "_" + args.pileup
    os.makedirs(output_dir, exist_ok=True)

    ak.to_parquet(events_gen, os.path.join(output_dir, "events_gen.parquet"))
    ak.to_parquet(events_0p0113, os.path.join(output_dir, "events_0p0113.parquet"))
    ak.to_parquet(events_0p016, os.path.join(output_dir, "events_0p016.parquet"))
    ak.to_parquet(events_0p03, os.path.join(output_dir, "events_0p03.parquet"))
    ak.to_parquet(events_0p045, os.path.join(output_dir, "events_0p045.parquet"))
    ak.to_parquet(events_Ref, os.path.join(output_dir, "events_Ref.parquet"))

  # Once the events have been skimmed and saves in the parquet files we can read them from there
  parquet_dir='/data_CMS/cms/amella/HGCAL_samples/parquet_files/'+args.particles+'_'+args.pileup+'/'
  if os.path.exists(parquet_dir):
    print("Opening", parquet_dir)
    events_0p0113 = ak.from_parquet(parquet_dir+"events_0p0113.parquet")
    events_0p016 = ak.from_parquet(parquet_dir+"events_0p016.parquet")
    events_0p03 = ak.from_parquet(parquet_dir+"events_0p03.parquet")
    events_0p045 = ak.from_parquet(parquet_dir+"events_0p045.parquet")
    events_Ref = ak.from_parquet(parquet_dir+"events_Ref.parquet")
    events_gen = ak.from_parquet(parquet_dir+"events_gen.parquet")
  else:
    print("Create the parquet files!")

  
  #########################################################################################
  ##################################### EFFICIENCY ########################################
  #########################################################################################

  if args.total_efficiency:
    print("------------------------------------------")
    print("Total efficiency for each emulation test")
    print("------------------------------------------")
    print("\n")
    print("After the selections on the gen particles we are left with", len(events_gen), "events and", len(ak.flatten(events_gen.genpart_pt,axis=-1)), "particles")
    print("\n")
    pair_cluster_0p0113_matched, pair_gen_masked_0p0113 = plot.compute_total_efficiency("0.0113", events_0p0113, events_gen, args, 'cl3d_p0113Tri_eta', 'cl3d_p0113Tri_phi')
    pair_cluster_0p016_matched, pair_gen_masked_0p016 = plot.compute_total_efficiency("0.016", events_0p016, events_gen, args, 'cl3d_p016Tri_eta', 'cl3d_p016Tri_phi')
    pair_cluster_0p03_matched, pair_gen_masked_0p03 = plot.compute_total_efficiency("0.03", events_0p03, events_gen, args, 'cl3d_p03Tri_eta', 'cl3d_p03Tri_phi')
    pair_cluster_0p045_matched, pair_gen_masked_0p045 = plot.compute_total_efficiency("0.045", events_0p045, events_gen, args, 'cl3d_p045Tri_eta', 'cl3d_p045Tri_phi')
    pair_cluster_Ref_matched, pair_gen_masked_Ref = plot.compute_total_efficiency("Ref", events_Ref, events_gen, args, 'cl3d_Ref_eta', 'cl3d_Ref_phi')

  else:
    #Apply matching of clusters and gen partciles requiring DeltaR <0.1 and taking the highest pt cluster
    pair_cluster_0p0113_matched, pair_gen_masked_0p0113, events_0p0113, events_gen_filtered_0p0113 = apply_matching(events_0p0113, 'cl3d_p0113Tri_eta','cl3d_p0113Tri_phi', events_gen, args, deltaR=0.1)
    pair_cluster_0p016_matched, pair_gen_masked_0p016, events_0p016, events_gen_filtered_0p016 = apply_matching(events_0p016, 'cl3d_p016Tri_eta','cl3d_p016Tri_phi', events_gen, args, deltaR=0.1)
    pair_cluster_0p03_matched, pair_gen_masked_0p03, events_0p03, events_gen_filtered_0p03 = apply_matching(events_0p03, 'cl3d_p03Tri_eta','cl3d_p03Tri_phi', events_gen, args, deltaR=0.1)
    pair_cluster_0p045_matched, pair_gen_masked_0p045, events_0p045, events_gen_filtered_0p045 = apply_matching(events_0p045, 'cl3d_p045Tri_eta','cl3d_p045Tri_phi', events_gen, args, deltaR=0.1)
    pair_cluster_Ref_matched, pair_gen_masked_Ref, events_Ref, events_gen_filtered_Ref = apply_matching(events_Ref, 'cl3d_Ref_eta','cl3d_Ref_phi', events_gen, args, deltaR=0.1)


  if args.efficiency:
    bin_eta=10
    range_eta = [1.6, 2.9]
    bin_pt=10
    if args.gen_pt_cut!=0:
      range_pt=[args.gen_pt_cut,100]
    else:
      range_pt=[0,100]
    efficiency = f"{output_dir}/efficiency"
    os.makedirs(efficiency, exist_ok=True)

    for var, bin_size, range_var in zip(['pT', 'eta'], [bin_pt, bin_eta], [range_pt, range_eta]):
      figure,ax=plt.subplots(figsize=(10, 10))
      plot.differential_efficiency(events_gen_filtered_0p0113, pair_gen_masked_0p0113, ax, args, "0p0113", var, bin_size, range_var, my_cmap(0/(n_colors-1)))
      plot.differential_efficiency(events_gen_filtered_0p016, pair_gen_masked_0p016, ax, args, "0p016", var, bin_size, range_var, my_cmap(1/(n_colors-1)))
      plot.differential_efficiency(events_gen_filtered_0p03, pair_gen_masked_0p03, ax, args, "0p03", var, bin_size, range_var, my_cmap(2/(n_colors-1)))
      plot.differential_efficiency(events_gen_filtered_0p045, pair_gen_masked_0p045, ax, args, "0p045", var, bin_size, range_var, my_cmap(3/(n_colors-1)))
      plot.differential_efficiency(events_gen_filtered_Ref, pair_gen_masked_Ref, ax, args, "Ref",var,  bin_size, range_var, my_cmap(4/(n_colors-1)))
      plt.savefig(f"{output_dir}/efficiency/Eficiency_{var}_differential.png",dpi=300)
      plt.savefig(f"{output_dir}/efficiency/Eficiency_{var}_differential.pdf")
      print(f"{output_dir}/efficiency/Eficiency_{var}_differential.png")

  #########################################################################################
  #################################### DISTRIBUTIONS ######################################
  #########################################################################################
  if args.distribution:

    distributions = f"{output_dir}/distributions"
    os.makedirs(distributions, exist_ok=True)
    # print("events", events_0p0113)
    bin_pt = 40
    if args.pileup=="PU0":
      range_pt = [0, 100]
      range_eta = [1.6, 2.9]
      range_phi = [-3.14, 3.14]
      range_deltaR=[0,0.1]
    else:
      range_pt = [0, 200]
      range_eta = [1.0, 3.5]
      range_phi = [-3.14, 3.14]
      range_deltaR=[0,0.1]
    bin_eta = 40
    bin_phi = 40
    bin_deltaR = 40
    legend_handles = []

    '''Plot 2D distributions in eta-pt and eta-deltaR'''

    for events, att_eta, triangle in zip([events_0p0113,events_0p016,events_0p03,events_0p045,events_Ref],['eta','eta','eta','eta','eta'],["0.0113","0.016","0.03","0.045","Ref"]):
      for range, label in zip([[[-2.9, -1.6], [0, 100]],[[1.6, 2.9], [0, 100]]], ["negative_eta","positive_eta"]):
        plot.plot_2D_histograms(events, att_eta, "eta", "pt", args, triangle, range)
        plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}_pt_eta.png", dpi=300)
        plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}_pt_eta.pdf")
        print(f"Saved figure: {output_dir}/distributions/2D_distributions_{triangle}_{label}_pt_eta.png")

    for events, att_eta, triangle in zip([pair_cluster_0p0113_matched,pair_cluster_0p016_matched,pair_cluster_0p03_matched,pair_cluster_0p045_matched,pair_cluster_Ref_matched],['eta','eta','eta','eta','eta'],["0.0113","0.016","0.03","0.045","Ref"]):
      for range, label in zip([[[-2.9, -1.6], [0, 0.1]],[[1.6, 2.9], [0, 0.1]]], ["negative_eta","positive_eta"]):
        plot.plot_2D_histograms(events, att_eta, "eta","delta_r", args, triangle, range, True)
        plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}_delta_r_eta.png", dpi=300)
        plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}_delta_r_eta.pdf")
        print(f"Saved figure: {output_dir}/distributions/2D_distributions_{triangle}_{label}_delta_r_eta.png")

    '''Plot distributions of pt, eta and phi of the cluster'''

    for var, bin_size, range_var in zip(['pT', 'eta', 'phi'], [bin_pt, bin_eta, bin_phi], [range_pt, range_eta, range_phi]):
      plt.figure(figsize=(10, 10))
      plot.comparison_histo_performance(events_0p0113, 'eta', args, var, bin_size, range_var, "0.0113", my_cmap(0 / (n_colors - 1)), legend_handles)
      plot.comparison_histo_performance(events_0p016, 'eta', args, var, bin_size, range_var, "0.016", my_cmap(1 / (n_colors - 1)), legend_handles)
      plot.comparison_histo_performance(events_0p03, 'eta', args, var, bin_size, range_var, "0.03", my_cmap(2 / (n_colors - 1)), legend_handles)
      plot.comparison_histo_performance(events_0p045, 'eta', args, var, bin_size, range_var, "0.045", my_cmap(3 / (n_colors - 1)), legend_handles)
      plot.comparison_histo_performance(events_Ref, 'eta', args, var, bin_size, range_var, "Ref", my_cmap(4 / (n_colors - 1)), legend_handles)
      plt.tight_layout()
      plt.savefig(f"{output_dir}/distributions/Non_matched_{var}_distributions.png", dpi=300)
      plt.savefig(f"{output_dir}/distributions/Non_matched_{var}_distributions.pdf")
      print(f"Saved figure: {output_dir}/distributions/Non_matched_{var}_distributions.png")
      legend_handles = []

    '''Plot distributions of pt, eta, phi and delta R of matched clusters'''

    for var, bin_size, range_var in zip(['pT', 'eta', 'phi', 'delta_r'], [bin_pt, bin_eta, bin_phi, bin_deltaR], [range_pt, range_eta, range_phi, range_deltaR]):
      plt.figure(figsize=(10, 10))
      plot.comparison_histo_performance(pair_cluster_0p0113_matched, 'eta', args, var, bin_size, range_var, "0.0113", my_cmap(0 / (n_colors - 1)), legend_handles, True)
      plot.comparison_histo_performance(pair_cluster_0p016_matched, 'eta', args, var, bin_size, range_var, "0.016", my_cmap(1 / (n_colors - 1)), legend_handles, True)
      plot.comparison_histo_performance(pair_cluster_0p03_matched, 'eta', args, var, bin_size, range_var, "0.03", my_cmap(2 / (n_colors - 1)), legend_handles, True)
      plot.comparison_histo_performance(pair_cluster_0p045_matched, 'eta', args, var, bin_size, range_var, "0.045", my_cmap(3 / (n_colors - 1)), legend_handles, True)
      plot.comparison_histo_performance(pair_cluster_Ref_matched, 'eta', args, var, bin_size, range_var, "Ref", my_cmap(4 / (n_colors - 1)), legend_handles, True)
      plt.tight_layout()
      plt.savefig(f"{output_dir}/distributions/Matched_{var}_distributions.png", dpi=300)
      plt.savefig(f"{output_dir}/distributions/Matched_{var}_distributions.pdf")
      print(f"Saved figure: {output_dir}/distributions/Matched_{var}_distributions.png")
      legend_handles = []

    if args.pileup=="PU0":
      range_n=[0,20]
      bin_n=20
    else:
      range_n=[0,50]
      bin_n=40

    '''Plot distributions of the number of reconstructed clusters'''

    plot.distribution_of_clusters_per_event(events_0p0113, events_gen_filtered_0p0113, 'eta', args, legend_handles, 1 , bin_n, range_n, "0p0113", my_cmap(0 / (n_colors - 1)))
    plot.distribution_of_clusters_per_event(events_0p016, events_gen_filtered_0p016, 'eta', args, legend_handles, 1 , bin_n, range_n, "0p016", my_cmap(1 / (n_colors - 1)))
    plot.distribution_of_clusters_per_event(events_0p03, events_gen_filtered_0p03, 'eta', args, legend_handles, 1 , bin_n, range_n, "0p03", my_cmap(2 / (n_colors - 1)))
    plot.distribution_of_clusters_per_event(events_0p045, events_gen_filtered_0p045, 'eta', args, legend_handles, 1 , bin_n, range_n, "0p045", my_cmap(3 / (n_colors - 1)))
    plot.distribution_of_clusters_per_event(events_Ref, events_gen_filtered_Ref, 'eta', args, legend_handles, 1 , bin_n, range_n, "Ref", my_cmap(4 / (n_colors - 1)))
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distributions/Distribution_Nclusters_1.png",dpi=300)
    plt.savefig(f"{output_dir}/distributions/Distribution_Nclusters_1.pdf")
    print(f"Saved figure: {output_dir}/distributions/Distribution_Nclusters_1.png")
    legend_handles = []
  
  #########################################################################################
  ############################### PER BIN DISTRIBUTIONS ###################################
  #########################################################################################

  if args.bins:

    if args.pileup=="PU0":
      range_n=[0,20]
      bin_n=20
    elif args.pileup =="PU200" and args.gen_pt_cut !=0:
      range_n=[0,50]
      bin_n=40
    else:
      range_n=[0,1400]
      bin_n=40
      
    bin_n_cl_pt=10
    bin_n_cl_eta=10
    range_n_cl_pt=[0,100]
    range_n_cl_eta=[1.6, 2.8]
    legend_handles = []

    datasets = [
    (events_0p0113, events_gen_filtered_0p0113,  "0p0113", my_cmap(0 / (n_colors - 1)),'eta'),
    (events_0p016, events_gen_filtered_0p016, "0p016",  my_cmap(1 / (n_colors - 1)),'eta'),
    (events_0p03, events_gen_filtered_0p03,  "0p03",   my_cmap(2 / (n_colors - 1)),'eta'),
    (events_0p045, events_gen_filtered_0p045, "0p045",  my_cmap(3 / (n_colors - 1)),'eta'),
    (events_Ref,  events_gen_filtered_Ref,  "Ref",    my_cmap(4 / (n_colors - 1)),'eta'),
    ]

    for var, bin_var, range_var in zip(['n_cl_pt','n_cl_eta'],[bin_n_cl_pt, bin_n_cl_eta],[range_n_cl_pt, range_n_cl_eta]):
      plot.plot_clusters_per_bin(datasets=datasets,bin_n=bin_var, range_=range_var,bin_nb=bin_n, range_nb=range_n,var=var, args=args,output_dir=output_dir,gen_n=1)

    if args.pileup=="PU0":
      range_pt = [0, 100]
      range_eta = [1.6, 2.9]
      range_phi = [-3.14, 3.14]
      range_pt_resp= [0.4, 1.4]
      range_eta_resp= [-0.05, 0.05]
      range_phi_resp= [-0.05, 0.05]
      bin_n=20
    else:
      #range of the bins
      range_pt = [0, 100]
      range_eta = [1.6, 2.9]
      range_phi = [-3.14, 3.14]
      #range of the distribution
      range_pt_resp= [0,5]
      range_eta_resp= [-0.1, 0.1]
      range_phi_resp= [-0.1, 0.1]
    bin_pt = 10
    bin_eta = 10
    bin_phi = 10
    legend_handles = []

    responses = f"{output_dir}/responses"
    os.makedirs(responses, exist_ok=True)
    
    datasets = [
    (events_0p0113, pair_cluster_0p0113_matched, pair_gen_masked_0p0113,  "0p0113", my_cmap(0 / (n_colors - 1)),'eta'),
    (events_0p016,  pair_cluster_0p016_matched, pair_gen_masked_0p016,  "0p016",  my_cmap(1 / (n_colors - 1)),'eta'),
    (events_0p03,   pair_cluster_0p03_matched, pair_gen_masked_0p03,   "0p03",   my_cmap(2 / (n_colors - 1)),'eta'),
    (events_0p045,  pair_cluster_0p045_matched, pair_gen_masked_0p045,  "0p045",  my_cmap(3 / (n_colors - 1)),'eta'),
    (events_Ref,    pair_cluster_Ref_matched, pair_gen_masked_Ref,    "Ref",    my_cmap(4 / (n_colors - 1)),'eta'),
    ]

    for var, bin_var, range_var, range_resp in zip(['pT','pT_eta','pT_phi', 'phi','eta'],[bin_pt,bin_eta,bin_phi,bin_phi,bin_eta],[range_pt, range_eta, range_phi, range_phi, range_eta],[range_pt_resp, range_pt_resp, range_pt_resp, range_phi_resp, range_eta_resp]):
      plot.plot_responses_per_bin(datasets=datasets, bin_n=bin_var, range_=range_var,bin_nb=bin_n, range_nb=range_resp, var=var, args=args,output_dir=output_dir)

    # plot.plot_responses_per_bin(datasets=datasets,events_gen=events_gen,bin_n=bin_phi, range_=range_phi,bin_nb=bin_n, range_nb=range_phi_resp, var='phi', args=args,output_dir=output_dir)

  #########################################################################################
  ############################### RESPONSE DISTRIBUTION ###################################
  #########################################################################################

  if args.scale:
    responses = f"{output_dir}/responses"
    os.makedirs(responses, exist_ok=True)
    legend_handles = []
    #Plotting for pt
    if args.pileup=="PU0":
      range_pt= [0.25, 1.25]
      range_eta= [-0.05, 0.05]
      range_phi= [-0.05, 0.05]
    else:
      range_pt= [0,5]
      range_eta= [-0.1, 0.1]
      range_phi= [-0.1, 0.1]
    bin_pt=30
   
    bin_eta=30
    
    bin_phi=30

    for var, bin_size, range_var in zip(['pT', 'eta', 'phi'], [bin_pt, bin_eta, bin_phi], [range_pt, range_eta, range_phi]):
      plt.figure(figsize=(10, 10))
      plot.scale_distribution(pair_cluster_0p0113_matched, pair_gen_masked_0p0113, args, var, bin_size, range_var, "0.0113", my_cmap(0 / (n_colors - 1)), legend_handles)
      plot.scale_distribution(pair_cluster_0p016_matched, pair_gen_masked_0p016, args, var, bin_size, range_var, "0.016", my_cmap(1 / (n_colors - 1)), legend_handles)
      plot.scale_distribution(pair_cluster_0p03_matched, pair_gen_masked_0p03, args, var, bin_size, range_var, "0.03", my_cmap(2 / (n_colors - 1)), legend_handles)
      plot.scale_distribution(pair_cluster_0p045_matched, pair_gen_masked_0p045, args, var, bin_size, range_var, "0.045", my_cmap(3 / (n_colors - 1)), legend_handles)
      plot.scale_distribution(pair_cluster_Ref_matched, pair_gen_masked_Ref, args, var, bin_size, range_var, "Ref", my_cmap(4 / (n_colors - 1)), legend_handles)
      plt.tight_layout()
      plt.savefig(f"{output_dir}/responses/Response_{var}_distributions_linear.png", dpi=300)
      plt.savefig(f"{output_dir}/responses/Response_matched_{var}_distributions_linear.pdf")
      print(f"Saved figure: {output_dir}/responses/Response_{var}_distributions_linear.png")
      legend_handles = []

  #########################################################################################
  ############################ RESPONSE AND RESOLUTION DIFFRENTIAL ########################
  #########################################################################################

  if args.resp:
    responses = f"{output_dir}/responses"
    os.makedirs(responses, exist_ok=True)

    if args.pileup=="PU0":
      range_pt = [0, 100]
      range_eta=[1.6, 2.9]
      range_phi = [-3.14, 3.14]
    else:
      range_pt = [0, 100]
      range_eta = [1.6, 2.9]
      range_phi = [-3.14, 3.14]

    legend_handles = []
    bin_phi=10
    bin_eta=10 
    bin_pt=10
    bin_n_cl_pt=10
    bin_n_cl_eta=10
    range_n_cl_pt=[0,100]
    range_n_cl_eta=[1.6, 2.8]

    #Response and resolution plots for pt, eta and phi

    for var, bin_size, range_var in zip(['pT', 'eta', 'phi', 'pT_eta','pT_phi'], [bin_pt, bin_eta, bin_phi, bin_eta, bin_phi], [range_pt, range_eta, range_phi, range_eta, range_phi]):
      fig, ax = plt.subplots(figsize=(10,10))
      plot.plot_responses(pair_cluster_0p0113_matched, pair_gen_masked_0p0113, args, var, ax, "0p0113", events_0p0113, "eta", my_cmap(0/ (n_colors - 1)), bin_size,range_var)
      plot.plot_responses(pair_cluster_0p016_matched, pair_gen_masked_0p016, args, var, ax, "0p016", events_0p016, "eta", my_cmap(1/ (n_colors - 1)), bin_size,range_var)
      plot.plot_responses(pair_cluster_0p03_matched, pair_gen_masked_0p03, args, var, ax, "0p03", events_0p03, "eta", my_cmap(2/ (n_colors - 1)), bin_size,range_var)
      plot.plot_responses(pair_cluster_0p045_matched, pair_gen_masked_0p045, args, var, ax, "0p045", events_0p045, "eta", my_cmap(3/ (n_colors - 1)), bin_size,range_var)
      plot.plot_responses(pair_cluster_Ref_matched, pair_gen_masked_Ref, args, var, ax, "Ref", events_Ref, "eta", my_cmap(4/ (n_colors - 1)), bin_size,range_var)
      if args.response:
        if args.fit and var=="pT":
          plt.savefig(f"{output_dir}/responses/Response_{var}_differential_fit.png",dpi=300)
          plt.savefig(f"{output_dir}/responses/Response_matched_{var}_differential_fit.pdf")
          print(f"{output_dir}/responses/Response_{var}_differential_fit.png")
          plt.close(fig)
        else:
          plt.savefig(f"{output_dir}/responses/Response_{var}_differential.png",dpi=300)
          plt.savefig(f"{output_dir}/responses/Response_matched_{var}_differential.pdf")
          print(f"{output_dir}/responses/Response_{var}_differential.png")
          plt.close(fig)
      if (var=="pT" or var=="pT_eta" or var=="pT_phi") and args.eff_rms:
        plt.savefig(f"{output_dir}/responses/Resolution_{var}_differential_effrms.png",dpi=300)
        plt.savefig(f"{output_dir}/responses/Resolution_matched_{var}_differential_effrms.pdf")
        print(f"{output_dir}/responses/Resolution_{var}_differential_effrms.png")
        plt.close(fig)
      if args.resolution:
        plt.savefig(f"{output_dir}/responses/Resolution_{var}_differential.png",dpi=300)
        plt.savefig(f"{output_dir}/responses/Resolution_matched_{var}_differential.pdf")
        print(f"{output_dir}/responses/Resolution_{var}_differential.png")
        plt.close(fig)
        
    if not args.eff_rms:
      #Number of clusters per event plots for pt and eta
      for var, bin, range in zip(['n_cl_pt', 'n_cl_eta'], [bin_n_cl_pt, bin_n_cl_eta], [range_n_cl_pt, range_n_cl_eta]):
        figure, ax = plt.subplots(figsize=(10, 10))
        plot.number_of_clusters_per_event(events_0p0113, events_gen_filtered_0p0113, 'eta', ax, args, 1,var,bin, range, "0p0113", my_cmap(0 / (n_colors - 1)))
        plot.number_of_clusters_per_event(events_0p016, events_gen_filtered_0p016, 'eta', ax, args, 1, var,bin, range, "0p016", my_cmap(1 / (n_colors - 1)))
        plot.number_of_clusters_per_event(events_0p03, events_gen_filtered_0p03, 'eta', ax, args, 1, var, bin, range, "0p03", my_cmap(2 / (n_colors - 1)))
        plot.number_of_clusters_per_event(events_0p045, events_gen_filtered_0p045, 'eta', ax, args, 1, var,bin, range, "0p045", my_cmap(3 / (n_colors - 1)))
        plot.number_of_clusters_per_event(events_Ref, events_gen_filtered_Ref, 'eta', ax, args, 1, var, bin, range, "Ref", my_cmap(4 / (n_colors - 1)))
        if args.response:
          plt.savefig(f"{output_dir}/responses/Response_{var}_differential_{1}.png",dpi=300)
          plt.savefig(f"{output_dir}/responses/Response_{var}_differential_{1}.pdf")
          print(f"Saved figure: {output_dir}/responses/Response_{var}_differential_{1}.png")
          plt.close(fig)
        if args.resolution:
          plt.savefig(f"{output_dir}/responses/Resolution_{var}_differential_{1}.png",dpi=300)
          plt.savefig(f"{output_dir}/responses/Resolution_{var}_differential_{1}.pdf")
          print(f"Saved figure: {output_dir}/responses/Resolution_{var}_differential_{1}.png")
          plt.close(fig)
        if args.distribution:
          plt.savefig(f"{output_dir}/distributions/Distribution_ncl_{var}_{1}.png",dpi=300)
          plt.savefig(f"{output_dir}/distributions/Distribution_{var}_{1}.pdf")
          print(f"Saved figure: {output_dir}/distributions/Distribution_{var}_{1}.png")
          plt.close(fig)

      figure, ax = plt.subplots(figsize=(10, 10))
      plot.number_of_clusters_per_event(events_0p0113, events_gen_filtered_0p0113, 'eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "0p0113", my_cmap(0 / (n_colors - 1)))
      plot.number_of_clusters_per_event(events_0p016, events_gen_filtered_0p016, 'eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "0p016", my_cmap(1 / (n_colors - 1)))
      plot.number_of_clusters_per_event(events_0p03, events_gen_filtered_0p03, 'eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "0p03", my_cmap(2 / (n_colors - 1)))
      plot.number_of_clusters_per_event(events_0p045, events_gen_filtered_0p045, 'eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "0p045", my_cmap(3 / (n_colors - 1)))
      plot.number_of_clusters_per_event(events_Ref, events_gen_filtered_Ref, 'eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "Ref", my_cmap(4 / (n_colors - 1)))
      if args.response:
        plt.savefig(f"{output_dir}/responses/Response_n_cl_pt_differential_2.png",dpi=300)
        plt.savefig(f"{output_dir}/responses/Response_matched_n_cl_pt_differential_2.pdf")
        print(f"Saved figure: {output_dir}/responses/Response_n_cl_pt_differential_2.png")
        plt.close(fig)
      if args.resolution:
        plt.savefig(f"{output_dir}/responses/Resolution_n_cl_pt_differential_2.png",dpi=300)
        plt.savefig(f"{output_dir}/responses/Resolution_matched_n_cl_pt_differential_2.pdf")
        print(f"Saved figure: {output_dir}/responses/Resolution_n_cl_pt_differential_2.png")
        plt.close(fig)
      if args.distribution:
          plt.savefig(f"{output_dir}/distributions/Distribution_ncl_{var}_2.png",dpi=300)
          plt.savefig(f"{output_dir}/distributions/Distribution_{var}_2.pdf")
          print(f"Saved figure: {output_dir}/distributions/Distribution_{var}_2.png")
          plt.close(fig)

##############################################################################
#######If needed to save the awk arrays for plotting into parquet files#######
##############################################################################

# if args.parquet:
#       #awkward arrays containing the cluster and gen information
#       events_gen, events_0p0113, events_0p016, events_0p03 , events_0p045, events_Ref =provide_events_performaces(args.n, args.particles, args.pileup)

#       # plot.compute_efficiency_test(events_0p0113, events_gen, 5, "pT", 'eta')
      
#       pair_cluster_0p0113_matched, pair_gen_masked_0p0113 = plot.apply_matching(events_0p0113, 'eta','cl3d_p0113Tri_phi', events_gen, deltaR=0.1)
#       pair_cluster_0p016_matched, pair_gen_masked_0p016 = plot.apply_matching(events_0p016, 'eta','cl3d_p016Tri_phi', events_gen, deltaR=0.1)
#       pair_cluster_0p03_matched, pair_gen_masked_0p03 = plot.apply_matching(events_0p03, 'eta','cl3d_p03Tri_phi', events_gen, deltaR=0.1)
#       pair_cluster_0p045_matched, pair_gen_masked_0p045 = plot.apply_matching(events_0p045, 'eta','cl3d_p045Tri_phi', events_gen, deltaR=0.1)
#       pair_cluster_Ref_matched, pair_gen_masked_Ref = plot.apply_matching(events_Ref, 'eta','cl3d_Ref_phi', events_gen, deltaR=0.1)

#       # Define your directory path
#       output_dir = "parquet_files/" + args.particles + "_" + args.pileup

#       # Create the directory if it doesn't exist
#       os.makedirs(output_dir, exist_ok=True)

#       #Save gen info
#       ak.to_parquet(events_gen, os.path.join(output_dir, "events_gen.parquet"))
#       ak.to_parquet(pair_gen_masked_0p0113,os.path.join(output_dir, "pair_gen_masked_0p0113.parquet"))
#       ak.to_parquet(pair_gen_masked_0p016,os.path.join(output_dir, "pair_gen_masked_0p016.parquet"))
#       ak.to_parquet(pair_gen_masked_0p03,os.path.join(output_dir, "pair_gen_masked_0p03.parquet"))
#       ak.to_parquet(pair_gen_masked_0p045,os.path.join(output_dir, "pair_gen_masked_0p045.parquet"))

#       #Save cluster reco
#       ak.to_parquet(events_0p0113, os.path.join(output_dir, "events_0p0113.parquet"))
#       ak.to_parquet(pair_cluster_0p0113_matched,os.path.join(output_dir, "pair_cluster_0p0113_matched.parquet"))
#       ak.to_parquet(events_0p016, os.path.join(output_dir, "events_0p016.parquet"))
#       ak.to_parquet(pair_cluster_0p016_matched,os.path.join(output_dir, "pair_cluster_0p016_matched.parquet"))
#       ak.to_parquet(events_0p03, os.path.join(output_dir, "events_0p03.parquet"))
#       ak.to_parquet(pair_cluster_0p03_matched,os.path.join(output_dir, "pair_cluster_0p03_matched.parquet"))
#       ak.to_parquet(events_0p045, os.path.join(output_dir, "events_0p045.parquet"))
#       ak.to_parquet(pair_cluster_0p045_matched,os.path.join(output_dir, "pair_cluster_0p045_matched.parquet"))
#       ak.to_parquet(events_Ref, os.path.join(output_dir, "events_Ref.parquet"))
#       ak.to_parquet(pair_cluster_Ref_matched,os.path.join(output_dir, "pair_cluster_Ref_matched.parquet"))

    
#       #To open these arrays:

#       events_0p0113 = ak.from_parquet("parquet_files/event_data_0p0113.parquet")
#       pair_cluster_0p0113_matched = ak.from_parquet("parquet_files/pair_cluster_0p0113_matched.parquet")

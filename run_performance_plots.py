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
  ''' python run_{output_dir}.py -n 2 --pileup PU0 --particles photons '''

  parser = argparse.ArgumentParser(description='Stage-2 Emulator Parameters')
  parser.add_argument('-n',          type=int, default=1,         help='Provide the number of events')
  parser.add_argument('--particles', type=str, default='photons', help='Choose the particle sample')
  parser.add_argument('--pileup',    type=str, default='PU0',     help='Choose the pileup - PU0 or PU200')
  parser.add_argument('--tag',       type=str, default='',        help='Name to make unique json files')
  parser.add_argument('--var',       type=str, default='pT',        help='Name of the variable to be plotted')
  parser.add_argument('--parquet',    action='store_true', help='Produce parquet file to plot')
  parser.add_argument('--pt_cut',    type=float, default=0,         help='Provide the cut for the pt')
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
  args = parser.parse_args()

  output_dir = f"results_performance_plots_{args.particles}_{args.pileup}"
  os.makedirs(output_dir, exist_ok=True)

  #if we want to plot different cuts in the same plot it is probably easier to save the arrays in a parquet file and plot from there
  if args.pt_cut:
    output_dir = f"results_performance_plots_{args.particles}_{args.pileup}_cut_{args.pt_cut}_GeV"
    os.makedirs(output_dir, exist_ok=True)


  #awkward arrays containing the cluster and gen information
  events_gen, events_0p0113, events_0p016, events_0p03 , events_0p045, events_Ref =provide_events_performaces(args.n, args.particles, args.pileup, args.pt_cut)
  # pint(ak.min(ak.flatten(events_0p045.cl3d_p045Tri_pt,axis=-1)))

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
    pair_cluster_0p0113_matched, pair_gen_masked_0p0113 = apply_matching(events_0p0113, 'cl3d_p0113Tri_eta','cl3d_p0113Tri_phi', events_gen, args, deltaR=0.1)
    pair_cluster_0p016_matched, pair_gen_masked_0p016 = apply_matching(events_0p016, 'cl3d_p016Tri_eta','cl3d_p016Tri_phi', events_gen, args, deltaR=0.1)
    pair_cluster_0p03_matched, pair_gen_masked_0p03 = apply_matching(events_0p03, 'cl3d_p03Tri_eta','cl3d_p03Tri_phi', events_gen, args, deltaR=0.1)
    pair_cluster_0p045_matched, pair_gen_masked_0p045 = apply_matching(events_0p045, 'cl3d_p045Tri_eta','cl3d_p045Tri_phi', events_gen, args, deltaR=0.1)
    pair_cluster_Ref_matched, pair_gen_masked_Ref = apply_matching(events_Ref, 'cl3d_Ref_eta','cl3d_Ref_phi', events_gen, args, deltaR=0.1)

  if args.efficiency:
    bin_eta=10
    range_eta=[1.6, 2.8]
    bin_pt=10
    range_pt=[0,100]
    #Response and resolution plots for pt, eta and phi

    efficiency = f"{output_dir}/efficiency"
    os.makedirs(efficiency, exist_ok=True)

    for var, bin_size, range_var in zip(['pT', 'eta'], [bin_pt, bin_eta], [range_pt, range_eta]):
      figure,ax=plt.subplots(figsize=(10, 10))
      plot.differential_efficiency(events_gen, pair_gen_masked_0p0113, ax, args, "0p0113", var, bin_size, range_var, my_cmap(0/(n_colors-1)))
      plot.differential_efficiency(events_gen, pair_gen_masked_0p016, ax, args, "0p016", var, bin_size, range_var, my_cmap(1/(n_colors-1)))
      plot.differential_efficiency(events_gen, pair_gen_masked_0p03, ax, args, "0p03", var, bin_size, range_var, my_cmap(2/(n_colors-1)))
      plot.differential_efficiency(events_gen, pair_gen_masked_0p045, ax, args, "0p045", var, bin_size, range_var, my_cmap(3/(n_colors-1)))
      plot.differential_efficiency(events_gen, pair_gen_masked_Ref, ax, args, "Ref",var,  bin_size, range_var, my_cmap(4/(n_colors-1)))
      plt.savefig(f"{output_dir}/efficiency/Eficiency_{var}_differential.png",dpi=300)
      plt.savefig(f"{output_dir}/efficiency/Eficiency_{var}_differential.pdf")
      print(f"{output_dir}/efficiency/Eficiency_{var}_differential.png")


  if args.distribution:
    distributions = f"{output_dir}/distributions"
    os.makedirs(distributions, exist_ok=True)
    # print("events", events_0p0113)
    bin_pt = 40
    range_pt = [0, 100]
    bin_eta = 40
    range_eta = [1.6, 2.8]
    bin_phi = 40
    range_phi = [0, 2.2]
    legend_handles = []

    for events, att_eta, triangle in zip([events_0p0113,events_0p016,events_0p03,events_0p045,events_Ref],['cl3d_p0113Tri_eta','cl3d_p016Tri_eta','cl3d_p03Tri_eta','cl3d_p045Tri_eta','cl3d_Ref_eta'],["0.0113","0.016","0.03","0.045","Ref"]):
      for range, label in zip([[[-2.9, -1.6], [0, 100]],[[1.6, 2.9], [0, 100]]], ["negative_eta","positive_eta"]):
        plot.plot_2D_histograms(events, att_eta, args, triangle, range)
        plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}.png", dpi=300)
        plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}.pdf")
        print(f"Saved figure: {output_dir}/distributions/2D_distributions_{triangle}_{label}.png")

    for var, bin_size, range_var in zip(['pT', 'eta', 'phi'], [bin_pt, bin_eta, bin_phi], [range_pt, range_eta, range_phi]):
      plt.figure(figsize=(10, 10))
      plot.comparison_histo_performance(events_0p0113, 'cl3d_p0113Tri_eta', args, var, bin_size, range_var, "0.0113", my_cmap(0 / (n_colors - 1)), legend_handles)
      plot.comparison_histo_performance(events_0p016, 'cl3d_p016Tri_eta', args, var, bin_size, range_var, "0.016", my_cmap(1 / (n_colors - 1)), legend_handles)
      plot.comparison_histo_performance(events_0p03, 'cl3d_p03Tri_eta', args, var, bin_size, range_var, "0.03", my_cmap(2 / (n_colors - 1)), legend_handles)
      plot.comparison_histo_performance(events_0p045, 'cl3d_p045Tri_eta', args, var, bin_size, range_var, "0.045", my_cmap(3 / (n_colors - 1)), legend_handles)
      plot.comparison_histo_performance(events_Ref, 'cl3d_Ref_eta', args, var, bin_size, range_var, "Ref", my_cmap(4 / (n_colors - 1)), legend_handles)
      plt.tight_layout()
      plt.savefig(f"{output_dir}/distributions/Non_matched_{var}_distributions.png", dpi=300)
      plt.savefig(f"{output_dir}/distributions/Non_matched_{var}_distributions.pdf")
      print(f"Saved figure: {output_dir}/distributions/Non_matched_{var}_distributions.png")
      legend_handles = []


  if args.scale:
    responses = f"{output_dir}/responses"
    os.makedirs(responses, exist_ok=True)
    legend_handles = []
    #Plotting for pt
    range_pt= [0.25, 1.25]
    bin_pt=30
    range_eta= [-0.05, 0.05]
    bin_eta=30
    range_phi= [-0.05, 0.05]
    bin_phi=30

    for var, bin_size, range_var in zip(['pT', 'eta', 'phi'], [bin_pt, bin_eta, bin_phi], [range_pt, range_eta, range_phi]):
      plt.figure(figsize=(10, 10))
      plot.scale_distribution(pair_cluster_0p0113_matched, pair_gen_masked_0p0113, args, var, bin_size, range_var, "0.0113", my_cmap(0 / (n_colors - 1)), legend_handles)
      plot.scale_distribution(pair_cluster_0p016_matched, pair_gen_masked_0p016, args, var, bin_size, range_var, "0.016", my_cmap(1 / (n_colors - 1)), legend_handles)
      plot.scale_distribution(pair_cluster_0p03_matched, pair_gen_masked_0p03, args, var, bin_size, range_var, "0.03", my_cmap(2 / (n_colors - 1)), legend_handles)
      plot.scale_distribution(pair_cluster_0p045_matched, pair_gen_masked_0p045, args, var, bin_size, range_var, "0.045", my_cmap(3 / (n_colors - 1)), legend_handles)
      plot.scale_distribution(pair_cluster_Ref_matched, pair_gen_masked_Ref, args, var, bin_size, range_var, "Ref", my_cmap(4 / (n_colors - 1)), legend_handles)
      plt.tight_layout()
      plt.savefig(f"{output_dir}/responses/Response_{var}_distributions.png", dpi=300)
      plt.savefig(f"{output_dir}/responses/Response_matched_{var}_distributions.pdf")
      print(f"Saved figure: {output_dir}/responses/Response_{var}_distributions.png")
      legend_handles = []


  if args.resp:
    responses = f"{output_dir}/responses"
    os.makedirs(responses, exist_ok=True)
    legend_handles = []
    range_phi= [-2.2, 2.2]
    bin_phi=10
    bin_eta=10
    range_eta=[1.6, 2.8]
    bin_pt=10
    range_pt=[0,100]
    bin_n_cl_pt=10
    bin_n_cl_eta=10
    range_n_cl_pt=[0,100]
    range_n_cl_eta=[1.6, 2.8]

    #Response and resolution plots for pt, eta and phi

    for var, bin_size, range_var in zip(['pT', 'eta', 'phi', 'pT_eta'], [bin_pt, bin_eta, bin_phi, bin_eta], [range_pt, range_eta, range_phi, range_eta]):
      fig, ax = plt.subplots(figsize=(10,10))
      plot.plot_responses(pair_cluster_0p0113_matched, pair_gen_masked_0p0113, args, var, ax, "0p0113", events_0p0113, "cl3d_p0113Tri_eta", my_cmap(0/ (n_colors - 1)), bin_size,range_var)
      plot.plot_responses(pair_cluster_0p016_matched, pair_gen_masked_0p016, args, var, ax, "0p016", events_0p016, "cl3d_p016Tri_eta", my_cmap(1/ (n_colors - 1)), bin_size,range_var)
      plot.plot_responses(pair_cluster_0p03_matched, pair_gen_masked_0p03, args, var, ax, "0p03", events_0p03, "cl3d_p03Tri_eta", my_cmap(2/ (n_colors - 1)), bin_size,range_var)
      plot.plot_responses(pair_cluster_0p045_matched, pair_gen_masked_0p045, args, var, ax, "0p045", events_0p045, "cl3d_p045Tri_eta", my_cmap(3/ (n_colors - 1)), bin_size,range_var)
      plot.plot_responses(pair_cluster_Ref_matched, pair_gen_masked_Ref, args, var, ax, "Ref", events_Ref, "cl3d_Ref_eta", my_cmap(4/ (n_colors - 1)), bin_size,range_var)
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
      if (var=="pT" or var=="pT_eta") and args.eff_rms:
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
        plot.number_of_clusters_per_event(events_0p0113, events_gen, 'cl3d_p0113Tri_eta', ax, args, 1,var,bin, range, "0p0113", my_cmap(0 / (n_colors - 1)))
        plot.number_of_clusters_per_event(events_0p016, events_gen, 'cl3d_p016Tri_eta', ax, args, 1, var,bin, range, "0p016", my_cmap(1 / (n_colors - 1)))
        plot.number_of_clusters_per_event(events_0p03, events_gen, 'cl3d_p03Tri_eta', ax, args, 1, var, bin, range, "0p03", my_cmap(2 / (n_colors - 1)))
        plot.number_of_clusters_per_event(events_0p045, events_gen, 'cl3d_p045Tri_eta', ax, args, 1, var,bin, range, "0p045", my_cmap(3 / (n_colors - 1)))
        plot.number_of_clusters_per_event(events_Ref, events_gen, 'cl3d_Ref_eta', ax, args, 1, var, bin, range, "Ref", my_cmap(4 / (n_colors - 1)))
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

      figure, ax = plt.subplots(figsize=(10, 10))
      plot.number_of_clusters_per_event(events_0p0113, events_gen, 'cl3d_p0113Tri_eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "0p0113", my_cmap(0 / (n_colors - 1)))
      plot.number_of_clusters_per_event(events_0p016, events_gen, 'cl3d_p016Tri_eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "0p016", my_cmap(1 / (n_colors - 1)))
      plot.number_of_clusters_per_event(events_0p03, events_gen, 'cl3d_p03Tri_eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "0p03", my_cmap(2 / (n_colors - 1)))
      plot.number_of_clusters_per_event(events_0p045, events_gen, 'cl3d_p045Tri_eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "0p045", my_cmap(3 / (n_colors - 1)))
      plot.number_of_clusters_per_event(events_Ref, events_gen, 'cl3d_Ref_eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "Ref", my_cmap(4 / (n_colors - 1)))
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

    

##############################################################################
#######If needed to save the awk arrays for plotting into parquet files#######
##############################################################################

if args.parquet:
      #awkward arrays containing the cluster and gen information
      events_gen, events_0p0113, events_0p016, events_0p03 , events_0p045, events_Ref =provide_events_performaces(args.n, args.particles, args.pileup)

      # plot.compute_efficiency_test(events_0p0113, events_gen, 5, "pT", 'cl3d_p0113Tri_eta')
      
      pair_cluster_0p0113_matched, pair_gen_masked_0p0113 = plot.apply_matching(events_0p0113, 'cl3d_p0113Tri_eta','cl3d_p0113Tri_phi', events_gen, deltaR=0.1)
      pair_cluster_0p016_matched, pair_gen_masked_0p016 = plot.apply_matching(events_0p016, 'cl3d_p016Tri_eta','cl3d_p016Tri_phi', events_gen, deltaR=0.1)
      pair_cluster_0p03_matched, pair_gen_masked_0p03 = plot.apply_matching(events_0p03, 'cl3d_p03Tri_eta','cl3d_p03Tri_phi', events_gen, deltaR=0.1)
      pair_cluster_0p045_matched, pair_gen_masked_0p045 = plot.apply_matching(events_0p045, 'cl3d_p045Tri_eta','cl3d_p045Tri_phi', events_gen, deltaR=0.1)
      pair_cluster_Ref_matched, pair_gen_masked_Ref = plot.apply_matching(events_Ref, 'cl3d_Ref_eta','cl3d_Ref_phi', events_gen, deltaR=0.1)

      # Define your directory path
      output_dir = "parquet_files/" + args.particles + "_" + args.pileup

      # Create the directory if it doesn't exist
      os.makedirs(output_dir, exist_ok=True)

      #Save gen info
      ak.to_parquet(events_gen, os.path.join(output_dir, "events_gen.parquet"))
      ak.to_parquet(pair_gen_masked_0p0113,os.path.join(output_dir, "pair_gen_masked_0p0113.parquet"))
      ak.to_parquet(pair_gen_masked_0p016,os.path.join(output_dir, "pair_gen_masked_0p016.parquet"))
      ak.to_parquet(pair_gen_masked_0p03,os.path.join(output_dir, "pair_gen_masked_0p03.parquet"))
      ak.to_parquet(pair_gen_masked_0p045,os.path.join(output_dir, "pair_gen_masked_0p045.parquet"))

      #Save cluster reco
      ak.to_parquet(events_0p0113, os.path.join(output_dir, "events_0p0113.parquet"))
      ak.to_parquet(pair_cluster_0p0113_matched,os.path.join(output_dir, "pair_cluster_0p0113_matched.parquet"))
      ak.to_parquet(events_0p016, os.path.join(output_dir, "events_0p016.parquet"))
      ak.to_parquet(pair_cluster_0p016_matched,os.path.join(output_dir, "pair_cluster_0p016_matched.parquet"))
      ak.to_parquet(events_0p03, os.path.join(output_dir, "events_0p03.parquet"))
      ak.to_parquet(pair_cluster_0p03_matched,os.path.join(output_dir, "pair_cluster_0p03_matched.parquet"))
      ak.to_parquet(events_0p045, os.path.join(output_dir, "events_0p045.parquet"))
      ak.to_parquet(pair_cluster_0p045_matched,os.path.join(output_dir, "pair_cluster_0p045_matched.parquet"))
      ak.to_parquet(events_Ref, os.path.join(output_dir, "events_Ref.parquet"))
      ak.to_parquet(pair_cluster_Ref_matched,os.path.join(output_dir, "pair_cluster_Ref_matched.parquet"))

    
      #To open these arrays:

      events_0p0113 = ak.from_parquet("parquet_files/event_data_0p0113.parquet")
      pair_cluster_0p0113_matched = ak.from_parquet("parquet_files/pair_cluster_0p0113_matched.parquet")

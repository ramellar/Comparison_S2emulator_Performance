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
  #plotting arguments
  parser.add_argument('--eff_rms',     action='store_true', help='Extract mean and std from selecting the shorted interval containing 68& events')
  parser.add_argument('--total_efficiency',     action='store_true', help='Compute the total efficiency for each emulation test')
  parser.add_argument('--resolution',     action='store_true', help='Extract mean and std from selecting the shorted interval containing 68& events')
  parser.add_argument('--response',     action='store_true', help='Extract mean and std from selecting the shorted interval containing 68& events')
  parser.add_argument('--distribution',     action='store_true', help='Plot the distributions')
  parser.add_argument('--resp',     action='store_true', help='Plot the response and resolution of the emulator')
  parser.add_argument('--scale',     action='store_true', help='Plot the response distribution of the emulator')
  args = parser.parse_args()


  #awkward arrays containing the cluster and gen information
  events_gen, events_0p0113, events_0p016, events_0p03 , events_0p045, events_Ref =provide_events_performaces(args.n, args.particles, args.pileup)

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


  if args.distribution:
    # print("events", events_0p0113)

    legend_handles = []

    #Plotting for pT
    plt.figure(figsize=(10,10))
    plot.comparison_histo_performance(events_0p0113, 'cl3d_p0113Tri_eta', args, "pT", 40, [0,100], "0.0113",my_cmap(0/ (n_colors - 1)), legend_handles)
    plot.comparison_histo_performance(events_0p016, 'cl3d_p016Tri_eta', args, "pT", 40, [0,100], "0.016",my_cmap(1/ (n_colors - 1)), legend_handles)
    plot.comparison_histo_performance(events_0p03, 'cl3d_p03Tri_eta', args, "pT", 40, [0,100], "0.03",my_cmap(2/ (n_colors - 1)), legend_handles)
    plot.comparison_histo_performance(events_0p045, 'cl3d_p045Tri_eta', args, "pT", 40, [0,100], "0.045",my_cmap(3/ (n_colors - 1)), legend_handles)
    plot.comparison_histo_performance(events_Ref, 'cl3d_Ref_eta', args, "pT", 40, [0,100], "Ref",my_cmap(4/ (n_colors - 1)), legend_handles)
    plt.tight_layout
    plt.savefig("performance_plots/Non_matched_pt_distributions.png",dpi=300)
    plt.savefig("performance_plots/Non_matched_pt_distributions.pdf")

    print("Saved figure: performance_plots/Non_matched_pt_distributions.png")

    legend_handles = []
    #Plotting for eta

    range_eta=[1.6, 2.8]
    plt.figure(figsize=(10,10))
    plot.comparison_histo_performance(events_0p0113, 'cl3d_p0113Tri_eta', args, "eta", 40, range_eta, "0.0113",my_cmap(0/ (n_colors - 1)), legend_handles)
    plot.comparison_histo_performance(events_0p016, 'cl3d_p016Tri_eta', args, "eta", 40, range_eta, "0.016",my_cmap(1/ (n_colors - 1)), legend_handles)
    plot.comparison_histo_performance(events_0p03, 'cl3d_p03Tri_eta', args, "eta", 40, range_eta, "0.03",my_cmap(2/ (n_colors - 1)), legend_handles)
    plot.comparison_histo_performance(events_0p045, 'cl3d_p045Tri_eta', args, "eta", 40, range_eta, "0.045",my_cmap(3/ (n_colors - 1)), legend_handles)
    plot.comparison_histo_performance(events_Ref, 'cl3d_Ref_eta', args, "eta", 40, range_eta, "Ref",my_cmap(4/ (n_colors - 1)), legend_handles)
    plt.tight_layout
    plt.savefig("performance_plots/Non_matched_eta_distributions.png",dpi=300)
    plt.savefig("performance_plots/Non_matched_eta_distributions.pdf")

    print("performance_plots/Non_matched_eta_distributions.png")

    legend_handles = []
    #Plotting for phi
    range_phi= [0,2.2]
    plt.figure(figsize=(10,10))
    plot.comparison_histo_performance(events_0p0113, 'cl3d_p0113Tri_eta', args, "phi", 40, range_phi, "0.0113", my_cmap(0/ (n_colors - 1)), legend_handles)
    plot.comparison_histo_performance(events_0p016, 'cl3d_p016Tri_eta', args, "phi", 40,range_phi , "0.016", my_cmap(1/ (n_colors - 1)), legend_handles)
    plot.comparison_histo_performance(events_0p03, 'cl3d_p03Tri_eta', args, "phi", 40, range_phi, "0.03", my_cmap(2/ (n_colors - 1)), legend_handles)
    plot.comparison_histo_performance(events_0p045, 'cl3d_p045Tri_eta', args, "phi", 40, range_phi, "0.045", my_cmap(3/ (n_colors - 1)), legend_handles)
    plot.comparison_histo_performance(events_Ref, 'cl3d_Ref_eta', args, "phi", 40,range_phi, "Ref", my_cmap(4/ (n_colors - 1)), legend_handles)
    plt.tight_layout
    plt.savefig("performance_plots/Non_matched_phi_distributions.png",dpi=300)
    plt.savefig("performance_plots/Non_matched_phi_distributions.pdf")

    print("performance_plots/Non_matched_phi_distributions.png")

  if args.scale:
    legend_handles = []
    #Plotting for pt
    range_pt= [0.25, 1.25]
    bin_pt=30
    plt.figure(figsize=(10,10))
    plot.scale_distribution(pair_cluster_0p0113_matched, pair_gen_masked_0p0113, args, "pT", bin_pt, range_pt, "0.0113", my_cmap(0/ (n_colors - 1)), legend_handles)
    plot.scale_distribution(pair_cluster_0p016_matched, pair_gen_masked_0p016, args, "pT", bin_pt, range_pt, "0.016", my_cmap(1/ (n_colors - 1)), legend_handles)
    plot.scale_distribution(pair_cluster_0p03_matched, pair_gen_masked_0p03, args, "pT", bin_pt, range_pt, "0.03", my_cmap(2/ (n_colors - 1)), legend_handles)
    plot.scale_distribution(pair_cluster_0p045_matched, pair_gen_masked_0p045, args, "pT", bin_pt, range_pt, "0.045", my_cmap(3/ (n_colors - 1)), legend_handles)
    plot.scale_distribution(pair_cluster_Ref_matched, pair_gen_masked_Ref, args, "pT", bin_pt, range_pt, "Ref", my_cmap(4/ (n_colors - 1)), legend_handles)
    plt.tight_layout
    plt.savefig("performance_plots/Response_pt_distributions.png",dpi=300)
    plt.savefig("performance_plots/Response_matched_pt_distributions.pdf")
    print("performance_plots/Response_pt_distributions.png")

    #Plotting for phi
    legend_handles = []
    range_eta= [-0.03, 0.03]
    bin_eta=20
    plt.figure(figsize=(10,10))
    plot.scale_distribution(pair_cluster_0p0113_matched, pair_gen_masked_0p0113, args, "eta", bin_eta, range_eta, "0.0113", my_cmap(0/ (n_colors - 1)), legend_handles)
    plot.scale_distribution(pair_cluster_0p016_matched, pair_gen_masked_0p016, args, "eta", bin_eta, range_eta, "0.016", my_cmap(1/ (n_colors - 1)), legend_handles)
    plot.scale_distribution(pair_cluster_0p03_matched, pair_gen_masked_0p03, args, "eta", bin_eta, range_eta, "0.03", my_cmap(2/ (n_colors - 1)), legend_handles)
    plot.scale_distribution(pair_cluster_0p045_matched, pair_gen_masked_0p045, args, "eta", bin_eta, range_eta, "0.045", my_cmap(3/ (n_colors - 1)), legend_handles)
    plot.scale_distribution(pair_cluster_Ref_matched, pair_gen_masked_Ref, args, "eta", bin_eta, range_eta, "Ref", my_cmap(4/ (n_colors - 1)), legend_handles)
    plt.tight_layout
    plt.savefig("performance_plots/Response_eta_distributions.png",dpi=300)
    plt.savefig("performance_plots/Response_matched_eta_distributions.pdf")
    print("performance_plots/Response_eta_distributions.png")

    #Plotting for phi
    legend_handles = []
    range_phi= [-0.03, 0.03]
    bin_phi=20
    plt.figure(figsize=(10,10))
    plot.scale_distribution(pair_cluster_0p0113_matched, pair_gen_masked_0p0113, args, "phi", bin_phi, range_phi, "0.0113", my_cmap(0/ (n_colors - 1)), legend_handles)
    plot.scale_distribution(pair_cluster_0p016_matched, pair_gen_masked_0p016, args, "phi", bin_phi, range_phi, "0.016", my_cmap(1/ (n_colors - 1)), legend_handles)
    plot.scale_distribution(pair_cluster_0p03_matched, pair_gen_masked_0p03, args, "phi", bin_phi, range_phi, "0.03", my_cmap(2/ (n_colors - 1)), legend_handles)
    plot.scale_distribution(pair_cluster_0p045_matched, pair_gen_masked_0p045, args, "phi", bin_phi, range_phi, "0.045", my_cmap(3/ (n_colors - 1)), legend_handles)
    plot.scale_distribution(pair_cluster_Ref_matched, pair_gen_masked_Ref, args, "phi", bin_phi, range_phi, "Ref", my_cmap(4/ (n_colors - 1)), legend_handles)
    plt.tight_layout
    plt.savefig("performance_plots/Response_phi_distributions.png",dpi=300)
    plt.savefig("performance_plots/Response_matched_phi_distributions.pdf")
    print("performance_plots/Response_phi_distributions.png")

  if args.resp:
    legend_handles = []
    range_phi= [-2.2, 2.2]
    bin_phi=10
    fig, ax = plt.subplots(figsize=(10,10))
    plot.plot_responses(pair_cluster_0p0113_matched, pair_gen_masked_0p0113, args, "phi", ax, "0p0113", my_cmap(0/ (n_colors - 1)), bin_phi,range_phi)
    plot.plot_responses(pair_cluster_0p016_matched, pair_gen_masked_0p016, args, "phi", ax, "0p016", my_cmap(1/ (n_colors - 1)), bin_phi,range_phi)
    plot.plot_responses(pair_cluster_0p03_matched, pair_gen_masked_0p03, args, "phi", ax, "0p03", my_cmap(2/ (n_colors - 1)), bin_phi,range_phi)
    plot.plot_responses(pair_cluster_0p045_matched, pair_gen_masked_0p045, args, "phi", ax, "0p045", my_cmap(3/ (n_colors - 1)), bin_phi,range_phi)
    plot.plot_responses(pair_cluster_Ref_matched, pair_gen_masked_Ref, args, "phi", ax, "Ref", my_cmap(4/ (n_colors - 1)), bin_phi,range_phi)
    if args.response:
      plt.savefig(f"performance_plots/Response_phi_differential_{range_phi}.png",dpi=300)
      plt.savefig(f"performance_plots/Response_matched_phi_differential_{range_phi}.pdf")
      print(f"performance_plots/Response_phi_differential_{range_phi}.png")
    if args.resolution:
       plt.savefig(f"performance_plots/Resolution_phi_differential_{range_phi}.png",dpi=300)
       plt.savefig(f"performance_plots/Resolution_matched_phi_differential_{range_phi}.pdf")
       print(f"performance_plots/Resolution_phi_differential_{range_phi}.png")
    
    range_eta=[1.6, 2.8]
    bin_eta=10
    fig, ax = plt.subplots(figsize=(10,10))
    plot.plot_responses(pair_cluster_0p0113_matched, pair_gen_masked_0p0113, args, "eta", ax, "0p0113", my_cmap(0/ (n_colors - 1)), bin_eta,range_eta)
    plot.plot_responses(pair_cluster_0p016_matched, pair_gen_masked_0p016, args, "eta", ax, "0p016", my_cmap(1/ (n_colors - 1)), bin_eta,range_eta)
    plot.plot_responses(pair_cluster_0p03_matched, pair_gen_masked_0p03, args, "eta", ax, "0p03", my_cmap(2/ (n_colors - 1)), bin_eta,range_eta)
    plot.plot_responses(pair_cluster_0p045_matched, pair_gen_masked_0p045, args, "eta", ax, "0p045", my_cmap(3/ (n_colors - 1)), bin_eta,range_eta)
    plot.plot_responses(pair_cluster_Ref_matched, pair_gen_masked_Ref, args, "eta", ax, "Ref", my_cmap(4/ (n_colors - 1)), bin_eta,range_eta)
    if args.response:
      plt.savefig(f"performance_plots/Response_eta_differential.png",dpi=300)
      plt.savefig(f"performance_plots/Response_matched_eta_differential.pdf")
      print(f"performance_plots/Response_eta_differential.png")
    if args.resolution:
      plt.savefig(f"performance_plots/Resolution_eta_differential.png",dpi=300)
      plt.savefig(f"performance_plots/Resolution_matched_eta_differential.pdf")
      print(f"performance_plots/Resolution_eta_differential.png")

    range_pt=[0,100]
    bin_pt=10
    fig, ax = plt.subplots(figsize=(10,10))
    plot.plot_responses(pair_cluster_0p0113_matched, pair_gen_masked_0p0113, args, "pT", ax, "0p0113", my_cmap(0/ (n_colors - 1)), bin_pt,range_pt)
    plot.plot_responses(pair_cluster_0p016_matched, pair_gen_masked_0p016, args, "pT", ax, "0p016", my_cmap(1/ (n_colors - 1)), bin_pt,range_pt)
    plot.plot_responses(pair_cluster_0p03_matched, pair_gen_masked_0p03, args, "pT", ax, "0p03", my_cmap(2/ (n_colors - 1)), bin_pt,range_pt)
    plot.plot_responses(pair_cluster_0p045_matched, pair_gen_masked_0p045, args, "pT", ax, "0p045", my_cmap(3/ (n_colors - 1)), bin_pt,range_pt)
    plot.plot_responses(pair_cluster_Ref_matched, pair_gen_masked_Ref, args, "pT", ax, "Ref", my_cmap(4/ (n_colors - 1)), bin_pt,range_pt)
    if args.response:
      plt.savefig(f"performance_plots/Response_pt_differential.png",dpi=300)
      plt.savefig(f"performance_plots/Response_matched_pt_differential.pdf")
      print(f"performance_plots/Response_pt_differential.png")
    if args.resolution:
      plt.savefig(f"performance_plots/Resolution_pt_differential.png",dpi=300)
      plt.savefig(f"performance_plots/Resolution_matched_pt_differential.pdf")
      print(f"performance_plots/Resolution_pt_differential.png")




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

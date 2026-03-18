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
from scipy.optimize import lsq_linear
from pathlib import Path

colors = ["tab:olive", "tab:cyan", "darkorchid" , "darkorange", "deeppink", "lightseagreen", "steelblue", "gold", "mediumslateblue", "coral"]
my_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", colors)
n_colors = len(colors)


if __name__ == '__main__':
  ''' python run_performance_plots.py -n 2 --pileup PU0 --particles photons '''

  parser = argparse.ArgumentParser(description='Stage-2 Emulator Parameters')
  parser.add_argument('--particles', type=str, default='photons', help='Choose the particle sample')
  parser.add_argument('--pileup',    type=str, default='PU0',     help='Choose the pileup - PU0 or PU200')
  parser.add_argument('--tag',       type=str, default='',        help='Name to make unique json files')
  parser.add_argument('--pt_cut',    type=float, default=0,         help='Provide the cut for the cluster pt')
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
  parser.add_argument('--calib_test',     action='store_true', help='Fit Pt response')
  parser.add_argument('--calib_test_PU0',     action='store_true', help='Fit Pt response')
  args = parser.parse_args()

  output_dir="plots/"

  if args.pt_cut:
    output_dir += f"results_performance_plots_{args.particles}_{args.pileup}_cut_{args.pt_cut}_GeV"
    parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/cluster_pt_cut{args.pt_cut}_GeV/'
    os.makedirs(output_dir, exist_ok=True)
  elif args.gen_pt_cut:
    output_dir += f"results_performance_plots_{args.particles}_{args.pileup}_gen_cut_{args.gen_pt_cut}_GeV"
    os.makedirs(output_dir, exist_ok=True)
    parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/'
  elif args.gen_pt_cut and args.pt_cut:
    output_dir += f"results_performance_plots_{args.particles}_{args.pileup}_gen_cut_{args.gen_pt_cut}_GeV_cut_{args.pt_cut}_GeV"
    os.makedirs(output_dir, exist_ok=True)
    parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/cluster_pt_cut{args.pt_cut}_GeV_gen_pt_cut{args.gen_pt_cut}_GeV/'
  else:
    parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/'
    output_dir += f"results_performance_plots_{args.particles}_{args.pileup}"
    os.makedirs(output_dir, exist_ok=True)

  if args.tag:
    output_dir += f"_{args.tag}"
    os.makedirs(output_dir, exist_ok=True)
    # parquet_dir += f"{args.tag}/"

  print(output_dir)
  

  #########################################################################################
  ##################################### LOAD DATA #########################################
  #########################################################################################

  # The data is loaded from parquet files saved in the parquet_dir
  # These parquet files are created with the matching.py script

  print("Loading cluster info")

  print("opening", parquet_dir)

  pair_cluster_0p0113_matched = ak.from_parquet(parquet_dir+"pair_cluster_0p0113_matched.parquet")
  pair_cluster_0p016_matched = ak.from_parquet(parquet_dir+"pair_cluster_0p016_matched.parquet")
  pair_cluster_0p03_matched = ak.from_parquet(parquet_dir+"pair_cluster_0p03_matched.parquet")
  pair_cluster_0p045_matched = ak.from_parquet(parquet_dir+"pair_cluster_0p045_matched.parquet")
  pair_cluster_Ref_matched = ak.from_parquet(parquet_dir+"pair_cluster_Ref_matched.parquet")

  print("Loading gen info")
  pair_gen_masked_0p0113 = ak.from_parquet(parquet_dir+"pair_gen_masked_0p0113.parquet")
  pair_gen_masked_0p016 = ak.from_parquet(parquet_dir+"pair_gen_masked_0p016.parquet")
  pair_gen_masked_0p03 = ak.from_parquet(parquet_dir+"pair_gen_masked_0p03.parquet")
  pair_gen_masked_0p045 = ak.from_parquet(parquet_dir+"pair_gen_masked_0p045.parquet")
  pair_gen_masked_Ref = ak.from_parquet(parquet_dir+"pair_gen_masked_Ref.parquet")

  # events_0p0113 = ak.from_parquet(parquet_dir+"events_0p0113_filtered.parquet")
  # events_0p016 = ak.from_parquet(parquet_dir+"events_0p016_filtered.parquet")
  # events_0p03 = ak.from_parquet(parquet_dir+"events_0p03_filtered.parquet")
  # events_0p045 = ak.from_parquet(parquet_dir+"events_0p045_filtered.parquet")
  # events_Ref = ak.from_parquet(parquet_dir+"events_Ref_filtered.parquet")

  print("Loading gen non matched info")

  # events_gen_filtered_0p0113 = ak.from_parquet(parquet_dir+"events_gen_filtered_0p0113.parquet")
  # events_gen_filtered_0p016 = ak.from_parquet(parquet_dir+"events_gen_filtered_0p016.parquet")
  # events_gen_filtered_0p03 = ak.from_parquet(parquet_dir+"events_gen_filtered_0p03.parquet")
  # events_gen_filtered_0p045 = ak.from_parquet(parquet_dir+"events_gen_filtered_0p045.parquet")
  # events_gen_filtered_Ref = ak.from_parquet(parquet_dir+"events_gen_filtered_Ref.parquet")

  if args.gen_pt_cut!=0:
    parquet_dir_PU0= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU0_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/'
  if os.path.exists(parquet_dir_PU0+"weight_wl_0p0113.parquet"):
    print("Opening calibration weights from", parquet_dir_PU0)
    # Weights per layer derived with pile up 0
    weight_wl_0p0113 = ak.from_parquet(parquet_dir_PU0+"weight_wl_0p0113.parquet")
    weight_wl_0p016 = ak.from_parquet(parquet_dir_PU0+"weight_wl_0p016.parquet")
    weight_wl_0p03 = ak.from_parquet(parquet_dir_PU0+"weight_wl_0p03.parquet")
    weight_wl_0p045 = ak.from_parquet(parquet_dir_PU0+"weight_wl_0p045.parquet")
    weight_wl_Ref = ak.from_parquet(parquet_dir_PU0+"weight_wl_Ref.parquet")

    # Apply calibration
    # pair_cluster_0p0113_matched= plot.apply_calibration(pair_cluster_0p0113_matched, weight_wl_0p0113)
    # pair_cluster_0p016_matched= plot.apply_calibration(pair_cluster_0p016_matched, weight_wl_0p016)
    # pair_cluster_0p03_matched= plot.apply_calibration(pair_cluster_0p03_matched, weight_wl_0p03)
    # pair_cluster_0p045_matched= plot.apply_calibration(pair_cluster_0p045_matched, weight_wl_0p045)
    # pair_cluster_Ref_matched= plot.apply_calibration(pair_cluster_Ref_matched, weight_wl_Ref)

  # if args.tag and args.gen_pt_cut!=0:
  #   parquet_dir_a= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/{args.tag}/'
  # if not args.tag and args.gen_pt_cut!=0:
  #   parquet_dir_a= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/'
  # if os.path.exists(parquet_dir_a+"weight_a_0p0113.parquet"):
  #   # Weights alpha and beta derived from PU200
  #   print("Opening calibration weights from", parquet_dir_a)
  #   weight_a_0p0113 = ak.from_parquet(parquet_dir_a+"weight_a_0p0113.parquet")
  #   weight_a_0p016 = ak.from_parquet(parquet_dir_a+"weight_a_0p016.parquet")
  #   weight_a_0p03 = ak.from_parquet(parquet_dir_a+"weight_a_0p03.parquet")
  #   weight_a_0p045 = ak.from_parquet(parquet_dir_a+"weight_a_0p045.parquet")
  #   weight_a_Ref = ak.from_parquet(parquet_dir_a+"weight_a_Ref.parquet")

  #   # Apply calibration
  #   pair_cluster_0p0113_matched= plot.apply_calibration_eta(pair_cluster_0p0113_matched, weight_a_0p0113)
  #   pair_cluster_0p016_matched= plot.apply_calibration_eta(pair_cluster_0p016_matched, weight_a_0p016)
  #   pair_cluster_0p03_matched= plot.apply_calibration_eta(pair_cluster_0p03_matched, weight_a_0p03)
  #   pair_cluster_0p045_matched= plot.apply_calibration_eta(pair_cluster_0p045_matched, weight_a_0p045)
  #   pair_cluster_Ref_matched= plot.apply_calibration_eta(pair_cluster_Ref_matched, weight_a_Ref)

  # # if args.tag and args.gen_pt_cut!=0:
  # #   parquet_dir_PU0= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU0_new_branch/{args.tag}/'
  # if args.tag and args.gen_pt_cut!=0:
  #   parquet_dir_wl= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/{args.tag}/'
  # if not args.tag and args.gen_pt_cut!=0:
  #   parquet_dir_wl= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/'
  # if os.path.exists(parquet_dir_wl+"weight_w_0p0113_PU200.parquet"):
  #   print("Opening calibration weights from", parquet_dir_PU0)
  #   # All weights, wl, a and b derived from PU200
  #   weight_0p0113 = ak.from_parquet(parquet_dir_wl+"weight_w_0p0113_PU200.parquet")
  #   weight_0p016 = ak.from_parquet(parquet_dir_wl+"weight_w_0p016_PU200.parquet")
  #   weight_0p03 = ak.from_parquet(parquet_dir_wl+"weight_w_0p03_PU200.parquet")
  #   weight_0p045 = ak.from_parquet(parquet_dir_wl+"weight_w_0p045_PU200.parquet")
  #   weight_Ref = ak.from_parquet(parquet_dir_wl+"weight_w_Ref_PU200.parquet")

  #   # Apply calibration
  #   pair_cluster_0p0113_matched= plot.apply_calibration_all_weights(pair_cluster_0p0113_matched, weight_0p0113)
  #   pair_cluster_0p016_matched= plot.apply_calibration_all_weights(pair_cluster_0p016_matched, weight_0p016)
  #   pair_cluster_0p03_matched= plot.apply_calibration_all_weights(pair_cluster_0p03_matched, weight_0p03)
  #   pair_cluster_0p045_matched= plot.apply_calibration_all_weights(pair_cluster_0p045_matched, weight_0p045)
  #   pair_cluster_Ref_matched= plot.apply_calibration_all_weights(pair_cluster_Ref_matched, weight_Ref)


  

  
  #########################################################################################
  ##################################### EFFICIENCY ########################################
  #########################################################################################
  

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
    bin_phi = 20
    bin_deltaR = 40
    legend_handles = []

    '''Plot 2D distributions in eta-pt and eta-deltaR'''

    # for events, att_eta, triangle in zip([events_0p0113,events_0p016,events_0p03,events_0p045,events_Ref],['eta','eta','eta','eta','eta'],["0.0113","0.016","0.03","0.045","Ref"]):
    #   for range, label in zip([[[-2.9, -1.6], [0, 100]],[[1.6, 2.9], [0, 100]]], ["negative_eta","positive_eta"]):
    #     plot.plot_2D_histograms(events, att_eta, "eta", "pt", args, triangle, range)
    #     plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}_pt_eta.png", dpi=300)
    #     plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}_pt_eta.pdf")
    #     print(f"Saved figure: {output_dir}/distributions/2D_distributions_{triangle}_{label}_pt_eta.png")

    # for events, att_eta, triangle in zip([pair_cluster_0p0113_matched,pair_cluster_0p016_matched,pair_cluster_0p03_matched,pair_cluster_0p045_matched,pair_cluster_Ref_matched],['eta','eta','eta','eta','eta'],["0.0113","0.016","0.03","0.045","Ref"]):
    #   for range, label in zip([[[-2.9, -1.6], [0, 0.1]],[[1.6, 2.9], [0, 0.1]]], ["negative_eta","positive_eta"]):
    #     plot.plot_2D_histograms(events, att_eta, "eta","delta_r", args, triangle, range, True)
    #     plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}_delta_r_eta.png", dpi=300)
    #     plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}_delta_r_eta.pdf")
    #     print(f"Saved figure: {output_dir}/distributions/2D_distributions_{triangle}_{label}_delta_r_eta.png")

    # for events, att_eta, triangle in zip([pair_cluster_0p0113_matched,pair_cluster_0p016_matched,pair_cluster_0p03_matched,pair_cluster_0p045_matched,pair_cluster_Ref_matched],['eta','eta','eta','eta','eta'],["0.0113","0.016","0.03","0.045","Ref"]):
    #   for range, label in zip([[[-np.pi, np.pi],[-2.9, -1.6]],[[-np.pi, np.pi],[1.6, 2.9]]], ["negative_eta","positive_eta"]):
    #     plot.plot_2D_histograms(events, att_eta, "eta","phi", args, triangle, range, True)
    #     plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}_phi.png", dpi=300)
    #     plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}_phi.pdf")
    #     print(f"Saved figure: {output_dir}/distributions/2D_distributions_{triangle}_{label}_phi.png")

    # for events, att_eta, triangle in zip([pair_cluster_0p0113_matched,pair_cluster_0p016_matched,pair_cluster_0p03_matched,pair_cluster_0p045_matched,pair_cluster_Ref_matched],['eta','eta','eta','eta','eta'],["0.0113","0.016","0.03","0.045","Ref"]):
    #     plot.plot_2D_histograms(events, att_eta, "eta","phi", args, triangle, [[-np.pi, np.pi],[-2.9, 2.9]], True)
    #     plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_eta_phi.png", dpi=300)
    #     plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_eta_phi.pdf")
    #     print(f"Saved figure: {output_dir}/distributions/2D_distributions_{triangle}_eta_phi.png")

    # for events, att_eta, triangle in zip([events_0p0113,events_0p016,events_0p03,events_0p045,events_Ref],['eta','eta','eta','eta','eta'],["0.0113","0.016","0.03","0.045","Ref"]):
    #   for range, label in zip([[[-np.pi, np.pi], [0, 100]],[[-np.pi, np.pi], [0, 100]]], ["negative_eta","positive_eta"]):
    #     plot.plot_2D_histograms(events, att_eta, "phi", "pt", args, triangle, range)
    #     plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}_pt_phi.png", dpi=300)
    #     plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_{label}_pt_phi.pdf")
    #     print(f"Saved figure: {output_dir}/distributions/2D_distributions_{triangle}_{label}_pt_phi.png")
    # for events, att_eta, triangle in zip([events_0p0113,events_0p016,events_0p03,events_0p045,events_Ref],['eta','eta','eta','eta','eta'],["0.0113","0.016","0.03","0.045","Ref"]):
    #   plot.plot_2D_histograms(events, att_eta, "phi", "pt", args, triangle, [[-np.pi, np.pi], [0, 100]])
    #   plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_pt_phi.png", dpi=300)
    #   plt.savefig(f"{output_dir}/distributions/2D_distributions_{triangle}_pt_phi.pdf")
    #   print(f"Saved figure: {output_dir}/distributions/2D_distributions_{triangle}_pt_phi.png")

    # '''Plot distributions of pt, eta and phi of the cluster'''

    # for var, bin_size, range_var in zip(['pT', 'eta', 'phi'], [bin_pt, bin_eta, bin_phi], [range_pt, range_eta, range_phi]):
    #   plt.figure(figsize=(10, 10))
    #   plot.comparison_histo_performance(events_0p0113, 'eta', args, var, bin_size, range_var, "0.0113", my_cmap(0 / (n_colors - 1)), legend_handles)
    #   plot.comparison_histo_performance(events_0p016, 'eta', args, var, bin_size, range_var, "0.016", my_cmap(1 / (n_colors - 1)), legend_handles)
    #   plot.comparison_histo_performance(events_0p03, 'eta', args, var, bin_size, range_var, "0.03", my_cmap(2 / (n_colors - 1)), legend_handles)
    #   plot.comparison_histo_performance(events_0p045, 'eta', args, var, bin_size, range_var, "0.045", my_cmap(3 / (n_colors - 1)), legend_handles)
    #   plot.comparison_histo_performance(events_Ref, 'eta', args, var, bin_size, range_var, "Ref", my_cmap(4 / (n_colors - 1)), legend_handles)
    #   plt.tight_layout()
    #   plt.savefig(f"{output_dir}/distributions/Non_matched_{var}_distributions.png", dpi=300)
    #   plt.savefig(f"{output_dir}/distributions/Non_matched_{var}_distributions.pdf")
    #   print(f"Saved figure: {output_dir}/distributions/Non_matched_{var}_distributions.png")
    #   legend_handles = []

    # '''Plot distributions of pt, eta, phi and delta R of matched clusters'''

    # for var, bin_size, range_var in zip(['pT', 'eta', 'phi', 'delta_r'], [bin_pt, bin_eta, bin_phi, bin_deltaR], [range_pt, range_eta, range_phi, range_deltaR]):
    #   plt.figure(figsize=(10, 10))
    #   plot.comparison_histo_performance(pair_cluster_0p0113_matched, 'eta', args, var, bin_size, range_var, "0.0113", my_cmap(0 / (n_colors - 1)), legend_handles, True)
    #   plot.comparison_histo_performance(pair_cluster_0p016_matched, 'eta', args, var, bin_size, range_var, "0.016", my_cmap(1 / (n_colors - 1)), legend_handles, True)
    #   plot.comparison_histo_performance(pair_cluster_0p03_matched, 'eta', args, var, bin_size, range_var, "0.03", my_cmap(2 / (n_colors - 1)), legend_handles, True)
    #   plot.comparison_histo_performance(pair_cluster_0p045_matched, 'eta', args, var, bin_size, range_var, "0.045", my_cmap(3 / (n_colors - 1)), legend_handles, True)
    #   plot.comparison_histo_performance(pair_cluster_Ref_matched, 'eta', args, var, bin_size, range_var, "Ref", my_cmap(4 / (n_colors - 1)), legend_handles, True)
    #   plt.tight_layout()
    #   plt.savefig(f"{output_dir}/distributions/Matched_{var}_distributions.png", dpi=300)
    #   plt.savefig(f"{output_dir}/distributions/Matched_{var}_distributions.pdf")
    #   print(f"Saved figure: {output_dir}/distributions/Matched_{var}_distributions.png")
    #   legend_handles = []

    # if args.pileup=="PU0":
    #   range_n=[0,20]
    #   bin_n=20
    # else:
    #   range_n=[0,50]
    #   bin_n=40

    # '''Plot distributions of the number of reconstructed clusters'''

    # plot.distribution_of_clusters_per_event(events_0p0113, events_gen_filtered_0p0113, 'eta', args, legend_handles, 1 , bin_n, range_n, "0p0113", my_cmap(0 / (n_colors - 1)))
    # plot.distribution_of_clusters_per_event(events_0p016, events_gen_filtered_0p016, 'eta', args, legend_handles, 1 , bin_n, range_n, "0p016", my_cmap(1 / (n_colors - 1)))
    # plot.distribution_of_clusters_per_event(events_0p03, events_gen_filtered_0p03, 'eta', args, legend_handles, 1 , bin_n, range_n, "0p03", my_cmap(2 / (n_colors - 1)))
    # plot.distribution_of_clusters_per_event(events_0p045, events_gen_filtered_0p045, 'eta', args, legend_handles, 1 , bin_n, range_n, "0p045", my_cmap(3 / (n_colors - 1)))
    # plot.distribution_of_clusters_per_event(events_Ref, events_gen_filtered_Ref, 'eta', args, legend_handles, 1 , bin_n, range_n, "Ref", my_cmap(4 / (n_colors - 1)))
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/distributions/Distribution_Nclusters_1.png",dpi=300)
    # plt.savefig(f"{output_dir}/distributions/Distribution_Nclusters_1.pdf")
    # print(f"Saved figure: {output_dir}/distributions/Distribution_Nclusters_1.png")
    # legend_handles = []

    calibrations = f"{output_dir}/calibrated_distributions"
    os.makedirs(calibrations, exist_ok=True)

    '''Plot the distribution of the calibrated energy'''

    for size, tri in zip([pair_cluster_0p0113_matched, pair_cluster_0p016_matched, pair_cluster_0p03_matched, pair_cluster_0p045_matched, pair_cluster_Ref_matched],["0.0113","0.016","0.03","0.045","Ref"]):
      plt.figure(figsize=(10, 10))
      plot.comparison_histo_performance(size, 'Ecalib' , args, 'Ecalib', bin_pt, range_pt, f"Calibrated pT \n for triangle size \n {tri}" , my_cmap(5 / (n_colors - 1)), legend_handles, matched=True)
      plot.comparison_histo_performance(size, 'pT', args, 'pT', bin_pt, range_pt, f"pT for \n triangle size \n {tri}", my_cmap(0 / (n_colors - 1)), legend_handles, matched=True)
      plt.tight_layout()
      plt.savefig(f"{calibrations}/Matched_{tri}_calib_pt_distributions.png", dpi=300)
      plt.savefig(f"{calibrations}/Matched_{tri}_calib_pt_distributions.pdf")
      print(f"Saved figure: {calibrations}/Matched_{tri}_calib_pt_distributions.png")
      legend_handles = []
      plt.close()
  
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

    # for var, bin_var, range_var in zip(['n_cl_pt','n_cl_eta'],[bin_n_cl_pt, bin_n_cl_eta],[range_n_cl_pt, range_n_cl_eta]):
    #   plot.plot_clusters_per_bin(datasets=datasets,bin_n=bin_var, range_=range_var,bin_nb=bin_n, range_nb=range_n,var=var, args=args,output_dir=output_dir,gen_n=1)

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
      bin_n=20
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
    calibration = f"{output_dir}/calibration"
    os.makedirs(calibration, exist_ok=True)
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

    # for var, bin_size, range_var in zip(['pT', 'eta', 'phi'], [bin_pt, bin_eta, bin_phi], [range_pt, range_eta, range_phi]):
    #   plt.figure(figsize=(10, 10))
    #   plot.scale_distribution(pair_cluster_0p0113_matched, pair_gen_masked_0p0113, args, var, bin_size, range_var, "0.0113", my_cmap(0 / (n_colors - 1)), legend_handles)
    #   plot.scale_distribution(pair_cluster_0p016_matched, pair_gen_masked_0p016, args, var, bin_size, range_var, "0.016", my_cmap(1 / (n_colors - 1)), legend_handles)
    #   plot.scale_distribution(pair_cluster_0p03_matched, pair_gen_masked_0p03, args, var, bin_size, range_var, "0.03", my_cmap(2 / (n_colors - 1)), legend_handles)
    #   plot.scale_distribution(pair_cluster_0p045_matched, pair_gen_masked_0p045, args, var, bin_size, range_var, "0.045", my_cmap(3 / (n_colors - 1)), legend_handles)
    #   plot.scale_distribution(pair_cluster_Ref_matched, pair_gen_masked_Ref, args, var, bin_size, range_var, "Ref", my_cmap(4 / (n_colors - 1)), legend_handles)
    #   plt.tight_layout()
    #   plt.savefig(f"{output_dir}/responses/Response_{var}_distributions_linear.png", dpi=300)
    #   plt.savefig(f"{output_dir}/responses/Response_matched_{var}_distributions_linear.pdf")
    #   print(f"Saved figure: {output_dir}/responses/Response_{var}_distributions_linear.png")
    #   legend_handles = []

    
    # for var1, var2 , cluster, gen, tri in zip(['Ecalib','Ecalib','Ecalib', 'Ecalib','Ecalib'],
    #                                             ['pT','pT','pT','pT','pT'],
    #                                             [pair_cluster_0p0113_matched, pair_cluster_0p016_matched, pair_cluster_0p03_matched, pair_cluster_0p045_matched, pair_cluster_Ref_matched],
    #                                             [pair_gen_masked_0p0113,pair_gen_masked_0p016, pair_gen_masked_0p03, pair_gen_masked_0p045, pair_gen_masked_Ref],
    #                                             ["0.0113","0.016","0.03","0.045","Ref"]):
    #   plt.figure(figsize=(10, 10))
    #   plot.scale_distribution(cluster, gen, args, var1, bin_pt, range_pt, f"Calibrated pT \n for triangle size \n {tri}" , my_cmap(5 / (n_colors - 1)), legend_handles)
    #   plot.scale_distribution(cluster, gen, args, var2, bin_pt, range_pt, f"pT for \n triangle size \n {tri}" , my_cmap(0 / (n_colors - 1)), legend_handles)
    #   plt.tight_layout()
    #   plt.savefig(f"{output_dir}/calibration/Response_{var1}_distributions_{tri}_linear.png", dpi=300)
    #   plt.savefig(f"{output_dir}/calibration/Response_matched_{var1}_distributions_{tri}_linear.pdf")
    #   print(f"Saved figure: {output_dir}/calibration/Response_{var1}_distributions_{tri}_linear.png")
    #   legend_handles = []

    # if args.pileup=="PU200":
    #   for var1, var2 , var3, var4, cluster, gen, tri in zip(['Ecalib','Ecalib','Ecalib', 'Ecalib','Ecalib'],
    #                                               ['Ecalib_cal_eta','Ecalib_cal_eta','Ecalib_cal_eta', 'Ecalib_cal_eta','Ecalib_cal_eta'],
    #                                               ['Ecalib_all','Ecalib_all','Ecalib_all', 'Ecalib_all','Ecalib_all'],
    #                                               ['pT','pT','pT','pT','pT'],
    #                                               [pair_cluster_0p0113_matched, pair_cluster_0p016_matched, pair_cluster_0p03_matched, pair_cluster_0p045_matched, pair_cluster_Ref_matched],
    #                                               [pair_gen_masked_0p0113,pair_gen_masked_0p016, pair_gen_masked_0p03, pair_gen_masked_0p045, pair_gen_masked_Ref],
    #                                               ["0.0113","0.016","0.03","0.045","Ref"]):
    #     plt.figure(figsize=(10, 10))
    #     plot.scale_distribution(cluster, gen, args, var1, bin_pt, range_pt, r"Calibrated pT with $w_l$ from PU0"+"\n"+ f" for triangle size \n {tri}" , my_cmap(5 / (n_colors - 1)), legend_handles)
    #     plot.scale_distribution(cluster, gen, args, var2, bin_pt, range_pt, r"Calibrated pT with $w_l$ from PU0 "+ "\n"+ r"$\alpha$ and $\beta$ from PU200"+f" for \n triangle size \n {tri}" , my_cmap(3 / (n_colors - 1)), legend_handles)
    #     plot.scale_distribution(cluster, gen, args, var3, bin_pt, range_pt, r"Calibrated pT with $w_l$, $\alpha$ and $\beta$"+ "\n"+f"from PU200 for triangle size \n {tri}" , my_cmap(2 / (n_colors - 1)), legend_handles)
    #     plot.scale_distribution(cluster, gen, args, var4, bin_pt, range_pt, f"pT for triangle size {tri}" , my_cmap(0 / (n_colors - 1)), legend_handles)
    #     plt.tight_layout()
    #     plt.savefig(f"{output_dir}/calibration/Response_{var1}_distributions_{tri}_comparison_all.png", dpi=300)
    #     plt.savefig(f"{output_dir}/calibration/Response_matched_{var1}_distributions_{tri}_comparison_all.pdf")
    #     print(f"Saved figure: {output_dir}/calibration/Response_{var1}_distributions_{tri}_comparison_all.png")
    #     legend_handles = []

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

    # for var,var2 , bin_size, range_var in zip(['pT','pT_eta','pT_phi'],['Ecalib','Ecalib_eta','Ecalib_phi'], 
    #                                           [bin_pt, bin_eta, bin_phi], [[20,100], range_eta, range_phi]):
    #   for cluster, gen, tri in zip([pair_cluster_0p0113_matched, pair_cluster_0p016_matched, pair_cluster_0p03_matched, pair_cluster_0p045_matched, pair_cluster_Ref_matched],
    #                                 [pair_gen_masked_0p0113,pair_gen_masked_0p016, pair_gen_masked_0p03, pair_gen_masked_0p045, pair_gen_masked_Ref],
    #                                 ["0.0113","0.016","0.03","0.045","Ref"]):
    #     fig, ax = plt.subplots(figsize=(10,10))
    #     plot.plot_responses(cluster, gen, args, var, ax, r"Cluster $p_T$ " + tri, my_cmap(0/ (n_colors - 1)), bin_size,range_var)
    #     plot.plot_responses(cluster, gen, args, var2, ax, r"Calibrated cluster $p_T$ " + tri,  my_cmap(5/ (n_colors - 1)), bin_size,range_var)
    #     if args.response:
    #       plt.savefig(f"{output_dir}/calibration/Response_{var}_differential_{tri}.png",dpi=300)
    #       plt.savefig(f"{output_dir}/calibration/Response_matched_{var}_differential_{tri}.pdf")
    #       print(f"{output_dir}/calibration/Response_{var}_differential_{tri}.png")
    #       plt.close(fig)
    #     elif args.eff_rms:
    #       plt.savefig(f"{output_dir}/calibration/Resolution_{var}_differential_effrms_{tri}.png",dpi=300)
    #       plt.savefig(f"{output_dir}/calibration/Resolution_matched_{var}_differential_effrms_{tri}.pdf")
    #       print(f"{output_dir}/calibration/Resolution_{var}_differential_effrms_{tri}.png")
    #       plt.close(fig)
    #     elif args.resolution:
    #       plt.savefig(f"{output_dir}/calibration/Resolution_{var}_differential_{tri}.png",dpi=300)
    #       plt.savefig(f"{output_dir}/calibration/Resolution_matched_{var}_differential_{tri}.pdf")
    #       print(f"{output_dir}/calibration/Resolution_{var}_differential_{tri}.png")
    #       plt.close(fig)

    # if args.pileup=="PU200":
    #   for var,var2 , var3, var4, bin_size, range_var in zip(['pT','pT_eta','pT_phi'],['Ecalib','Ecalib_eta','Ecalib_phi'], ['Ecalib_cal_eta','Ecalib_eta_eta','Ecalib_eta_phi'],['Ecalib_all_','Ecalib_all_eta_','Ecalib_all_phi_'],
    #                                             [bin_pt, bin_eta, bin_phi], [[20,100], range_eta, range_phi]):
    #     for cluster, gen, tri in zip([pair_cluster_0p0113_matched, pair_cluster_0p016_matched, pair_cluster_0p03_matched, pair_cluster_0p045_matched, pair_cluster_Ref_matched],
    #                                   [pair_gen_masked_0p0113,pair_gen_masked_0p016, pair_gen_masked_0p03, pair_gen_masked_0p045, pair_gen_masked_Ref],
    #                                   ["0.0113","0.016","0.03","0.045","Ref"]):
    #       fig, ax = plt.subplots(figsize=(10,10))
    #       plot.plot_responses(cluster, gen, args, var, ax, r"Cluster $p_T$ " + tri, my_cmap(0/ (n_colors - 1)), bin_size,range_var)
    #       plot.plot_responses(cluster, gen, args, var2, ax, r"Calibrated pT with $w_l$ from PU0"+"\n"+ f" for triangle size \n {tri}",  my_cmap(5/ (n_colors - 1)), bin_size,range_var)
    #       plot.plot_responses(cluster, gen, args, var3, ax, r"Calibrated pT with $w_l$ from PU0 "+ "\n"+ r"$\alpha$ and $\beta$ from PU200"+ f" for triangle size \n {tri}",  my_cmap(3/ (n_colors - 1)), bin_size,range_var)
    #       plot.plot_responses(cluster, gen, args, var4, ax, r"Calibrated pT with $w_l$, $\alpha$ and $\beta$"+ "\n"+f"from PU200 for triangle size \n {tri}",  my_cmap(2/ (n_colors - 1)), bin_size,range_var)
    #       if args.response:
    #         plt.savefig(f"{output_dir}/calibration/Response_{var}_differential_{tri}_eta_calib.png",dpi=300)
    #         plt.savefig(f"{output_dir}/calibration/Response_matched_{var}_differential_{tri}_eta_calib.pdf")
    #         print(f"{output_dir}/calibration/Response_{var}_differential_{tri}_eta_calib.png")
    #         plt.close(fig)
    #       elif args.eff_rms:
    #         plt.savefig(f"{output_dir}/calibration/Resolution_{var}_differential_effrms_{tri}_eta_calib.png",dpi=300)
    #         plt.savefig(f"{output_dir}/calibration/Resolution_matched_{var}_differential_effrms_{tri}_eta_calib.pdf")
    #         print(f"{output_dir}/calibration/Resolution_{var}_differential_effrms_{tri}_eta_calib.png")
    #         plt.close(fig)
    #       elif args.resolution:
    #         plt.savefig(f"{output_dir}/calibration/Resolution_{var}_differential_{tri}_eta_calib.png",dpi=300)
    #         plt.savefig(f"{output_dir}/calibration/Resolution_matched_{var}_differential_{tri}_eta_calib.pdf")
    #         print(f"{output_dir}/calibration/Resolution_{var}_differential_{tri}_eta_calib.png")
    #         plt.close(fig)


    # for var, bin_size, range_var in zip(['pT', 'eta', 'phi', 'pT_eta','pT_phi'], [bin_pt, bin_eta, bin_phi, bin_eta, bin_phi], [range_pt, range_eta, range_phi, range_eta, range_phi]):
    #   fig, ax = plt.subplots(figsize=(10,10))
    #   plot.plot_responses(pair_cluster_0p0113_matched, pair_gen_masked_0p0113, args, var, ax, "0p0113", my_cmap(0/ (n_colors - 1)), bin_size,range_var)
    #   plot.plot_responses(pair_cluster_0p016_matched, pair_gen_masked_0p016, args, var, ax, "0p016", my_cmap(1/ (n_colors - 1)), bin_size,range_var)
    #   plot.plot_responses(pair_cluster_0p03_matched, pair_gen_masked_0p03, args, var, ax, "0p03", my_cmap(2/ (n_colors - 1)), bin_size,range_var)
    #   plot.plot_responses(pair_cluster_0p045_matched, pair_gen_masked_0p045, args, var, ax, "0p045", my_cmap(3/ (n_colors - 1)), bin_size,range_var)
    #   plot.plot_responses(pair_cluster_Ref_matched, pair_gen_masked_Ref, args, var, ax, "Ref", my_cmap(4/ (n_colors - 1)), bin_size,range_var)
    #   if args.response:
    #     if args.fit and var=="pT":
    #       plt.savefig(f"{output_dir}/responses/Response_{var}_differential_fit.png",dpi=300)
    #       plt.savefig(f"{output_dir}/responses/Response_matched_{var}_differential_fit.pdf")
    #       print(f"{output_dir}/responses/Response_{var}_differential_fit.png")
    #       plt.close(fig)
    #     else:
    #       plt.savefig(f"{output_dir}/responses/Response_{var}_differential.png",dpi=300)
    #       plt.savefig(f"{output_dir}/responses/Response_matched_{var}_differential.pdf")
    #       print(f"{output_dir}/responses/Response_{var}_differential.png")
    #       plt.close(fig)
    #   elif (var=="pT" or var=="pT_eta" or var=="pT_phi") and args.eff_rms:
    #     plt.savefig(f"{output_dir}/responses/Resolution_{var}_differential_effrms.png",dpi=300)
    #     plt.savefig(f"{output_dir}/responses/Resolution_matched_{var}_differential_effrms.pdf")
    #     print(f"{output_dir}/responses/Resolution_{var}_differential_effrms.png")
    #     plt.close(fig)
    #   elif args.resolution:
    #     plt.savefig(f"{output_dir}/responses/Resolution_{var}_differential.png",dpi=300)
    #     plt.savefig(f"{output_dir}/responses/Resolution_matched_{var}_differential.pdf")
    #     print(f"{output_dir}/responses/Resolution_{var}_differential.png")
    #     plt.close(fig)
        
    # if not args.eff_rms:
    #   #Number of clusters per event plots for pt and eta
    #   for var, bin, range in zip(['n_cl_pt', 'n_cl_eta'], [bin_n_cl_pt, bin_n_cl_eta], [range_n_cl_pt, range_n_cl_eta]):
    #     figure, ax = plt.subplots(figsize=(10, 10))
    #     plot.number_of_clusters_per_event(events_0p0113, events_gen_filtered_0p0113, 'eta', ax, args, 1,var,bin, range, "0p0113", my_cmap(0 / (n_colors - 1)))
    #     plot.number_of_clusters_per_event(events_0p016, events_gen_filtered_0p016, 'eta', ax, args, 1, var,bin, range, "0p016", my_cmap(1 / (n_colors - 1)))
    #     plot.number_of_clusters_per_event(events_0p03, events_gen_filtered_0p03, 'eta', ax, args, 1, var, bin, range, "0p03", my_cmap(2 / (n_colors - 1)))
    #     plot.number_of_clusters_per_event(events_0p045, events_gen_filtered_0p045, 'eta', ax, args, 1, var,bin, range, "0p045", my_cmap(3 / (n_colors - 1)))
    #     plot.number_of_clusters_per_event(events_Ref, events_gen_filtered_Ref, 'eta', ax, args, 1, var, bin, range, "Ref", my_cmap(4 / (n_colors - 1)))
    #     if args.response:
    #       plt.savefig(f"{output_dir}/responses/Response_{var}_differential_{1}.png",dpi=300)
    #       plt.savefig(f"{output_dir}/responses/Response_{var}_differential_{1}.pdf")
    #       print(f"Saved figure: {output_dir}/responses/Response_{var}_differential_{1}.png")
    #       plt.close(fig)
    #     if args.resolution:
    #       plt.savefig(f"{output_dir}/responses/Resolution_{var}_differential_{1}.png",dpi=300)
    #       plt.savefig(f"{output_dir}/responses/Resolution_{var}_differential_{1}.pdf")
    #       print(f"Saved figure: {output_dir}/responses/Resolution_{var}_differential_{1}.png")
    #       plt.close(fig)
    #     if args.distribution:
    #       plt.savefig(f"{output_dir}/distributions/Distribution_ncl_{var}_{1}.png",dpi=300)
    #       plt.savefig(f"{output_dir}/distributions/Distribution_{var}_{1}.pdf")
    #       print(f"Saved figure: {output_dir}/distributions/Distribution_{var}_{1}.png")
    #       plt.close(fig)

    #   figure, ax = plt.subplots(figsize=(10, 10))
    #   plot.number_of_clusters_per_event(events_0p0113, events_gen_filtered_0p0113, 'eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "0p0113", my_cmap(0 / (n_colors - 1)))
    #   plot.number_of_clusters_per_event(events_0p016, events_gen_filtered_0p016, 'eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "0p016", my_cmap(1 / (n_colors - 1)))
    #   plot.number_of_clusters_per_event(events_0p03, events_gen_filtered_0p03, 'eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "0p03", my_cmap(2 / (n_colors - 1)))
    #   plot.number_of_clusters_per_event(events_0p045, events_gen_filtered_0p045, 'eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "0p045", my_cmap(3 / (n_colors - 1)))
    #   plot.number_of_clusters_per_event(events_Ref, events_gen_filtered_Ref, 'eta', ax, args, 2 ,'n_cl_pt', bin_n_cl_pt, range_n_cl_pt, "Ref", my_cmap(4 / (n_colors - 1)))
    #   if args.response:
    #     plt.savefig(f"{output_dir}/responses/Response_n_cl_pt_differential_2.png",dpi=300)
    #     plt.savefig(f"{output_dir}/responses/Response_matched_n_cl_pt_differential_2.pdf")
    #     print(f"Saved figure: {output_dir}/responses/Response_n_cl_pt_differential_2.png")
    #     plt.close(fig)
    #   if args.resolution:
    #     plt.savefig(f"{output_dir}/responses/Resolution_n_cl_pt_differential_2.png",dpi=300)
    #     plt.savefig(f"{output_dir}/responses/Resolution_matched_n_cl_pt_differential_2.pdf")
    #     print(f"Saved figure: {output_dir}/responses/Resolution_n_cl_pt_differential_2.png")
    #     plt.close(fig)
    #   if args.distribution:
    #       plt.savefig(f"{output_dir}/distributions/Distribution_ncl_{var}_2.png",dpi=300)
    #       plt.savefig(f"{output_dir}/distributions/Distribution_{var}_2.pdf")
    #       print(f"Saved figure: {output_dir}/distributions/Distribution_{var}_2.png")
    #       plt.close(fig)

  if args.calib_test_PU0:
    print("In calib test PU0")
    parquet_dir_wl= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU0_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV'

    weight_0p0113_fit_bounds_0_2 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_2/"+"weight_wl_0p0113.parquet")
    weight_0p016_fit_bounds_0_2 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_2/"+"weight_wl_0p016.parquet")
    weight_0p03_fit_bounds_0_2 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_2/"+"weight_wl_0p03.parquet")
    weight_0p045_fit_bounds_0_2 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_2/"+"weight_wl_0p045.parquet")
    weight_Ref_fit_bounds_0_2 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_2/"+"weight_wl_Ref.parquet")

    weight_0p0113_fit_bounds_0_20 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_20/"+"weight_wl_0p0113.parquet")
    weight_0p016_fit_bounds_0_20 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_20/"+"weight_wl_0p016.parquet")
    weight_0p03_fit_bounds_0_20 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_20/"+"weight_wl_0p03.parquet")
    weight_0p045_fit_bounds_0_20 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_20/"+"weight_wl_0p045.parquet")
    weight_Ref_fit_bounds_0_20 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_20/"+"weight_wl_Ref.parquet")

    weight_0p0113_fit_bounds_0_2_no_layer1 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_2_no_layer1/"+"weight_wl_0p0113.parquet")
    weight_0p016_fit_bounds_0_2_no_layer1 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_2_no_layer1/"+"weight_wl_0p016.parquet")
    weight_0p03_fit_bounds_0_2_no_layer1 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_2_no_layer1/"+"weight_wl_0p03.parquet")
    weight_0p045_fit_bounds_0_2_no_layer1 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_2_no_layer1/"+"weight_wl_0p045.parquet")
    weight_Ref_fit_bounds_0_2_no_layer1 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_2_no_layer1/"+"weight_wl_Ref.parquet")

    weight_0p0113_fit_bounds_0_20_no_layer1 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_20_no_layer1/"+"weight_wl_0p0113.parquet")
    weight_0p016_fit_bounds_0_20_no_layer1 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_20_no_layer1/"+"weight_wl_0p016.parquet")
    weight_0p03_fit_bounds_0_20_no_layer1 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_20_no_layer1/"+"weight_wl_0p03.parquet")
    weight_0p045_fit_bounds_0_20_no_layer1 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_20_no_layer1/"+"weight_wl_0p045.parquet")
    weight_Ref_fit_bounds_0_20_no_layer1 = ak.from_parquet(parquet_dir_wl+"/PU0_bounds_0_20_no_layer1/"+"weight_wl_Ref.parquet")
    
    weight_0p0113_no_bounds = ak.from_parquet(parquet_dir_wl+"/PU0_no_bounds/"+"weight_wl_0p0113.parquet")
    weight_0p016_no_bounds = ak.from_parquet(parquet_dir_wl+"/PU0_no_bounds/"+"weight_wl_0p016.parquet")
    weight_0p03_no_bounds = ak.from_parquet(parquet_dir_wl+"/PU0_no_bounds/"+"weight_wl_0p03.parquet")
    weight_0p045_no_bounds = ak.from_parquet(parquet_dir_wl+"/PU0_no_bounds/"+"weight_wl_0p045.parquet")
    weight_Ref_no_bounds = ak.from_parquet(parquet_dir_wl+"/PU0_no_bounds/"+"weight_wl_Ref.parquet")
    
    weight_0p0113_no_bounds_nolayer1 = ak.from_parquet(parquet_dir_wl+"/PU0_no_bounds_no_layer1/"+"weight_wl_0p0113.parquet")
    weight_0p016_no_bounds_nolayer1 = ak.from_parquet(parquet_dir_wl+"/PU0_no_bounds_no_layer1/"+"weight_wl_0p016.parquet")
    weight_0p03_no_bounds_nolayer1 = ak.from_parquet(parquet_dir_wl+"/PU0_no_bounds_no_layer1/"+"weight_wl_0p03.parquet")
    weight_0p045_no_bounds_nolayer1 = ak.from_parquet(parquet_dir_wl+"/PU0_no_bounds_no_layer1/"+"weight_wl_0p045.parquet")
    weight_Ref_no_bounds_nolayer1 = ak.from_parquet(parquet_dir_wl+"/PU0_no_bounds_no_layer1/"+"weight_wl_Ref.parquet")

    pair_cluster_0p0113_matched= plot.apply_calibration(pair_cluster_0p0113_matched, weight_0p0113_fit_bounds_0_2, "fit_bounds_0_2")
    pair_cluster_0p016_matched= plot.apply_calibration(pair_cluster_0p016_matched, weight_0p016_fit_bounds_0_2, "fit_bounds_0_2")
    pair_cluster_0p03_matched= plot.apply_calibration(pair_cluster_0p03_matched, weight_0p03_fit_bounds_0_2, "fit_bounds_0_2")
    pair_cluster_0p045_matched= plot.apply_calibration(pair_cluster_0p045_matched, weight_0p045_fit_bounds_0_2, "fit_bounds_0_2")
    pair_cluster_Ref_matched= plot.apply_calibration(pair_cluster_Ref_matched, weight_Ref_fit_bounds_0_2, "fit_bounds_0_2")

    pair_cluster_0p0113_matched= plot.apply_calibration(pair_cluster_0p0113_matched, weight_0p0113_fit_bounds_0_20, "fit_bounds_0_20")
    pair_cluster_0p016_matched= plot.apply_calibration(pair_cluster_0p016_matched, weight_0p016_fit_bounds_0_20, "fit_bounds_0_20")
    pair_cluster_0p03_matched= plot.apply_calibration(pair_cluster_0p03_matched, weight_0p03_fit_bounds_0_20, "fit_bounds_0_20")
    pair_cluster_0p045_matched= plot.apply_calibration(pair_cluster_0p045_matched, weight_0p045_fit_bounds_0_20, "fit_bounds_0_20")
    pair_cluster_Ref_matched= plot.apply_calibration(pair_cluster_Ref_matched, weight_Ref_fit_bounds_0_20, "fit_bounds_0_20")

    pair_cluster_0p0113_matched= plot.apply_calibration(pair_cluster_0p0113_matched, weight_0p0113_no_bounds_nolayer1, "fit_bounds_0_2_no_layer1")
    pair_cluster_0p016_matched= plot.apply_calibration(pair_cluster_0p016_matched, weight_0p016_no_bounds_nolayer1, "fit_bounds_0_2_no_layer1")
    pair_cluster_0p03_matched= plot.apply_calibration(pair_cluster_0p03_matched, weight_0p03_no_bounds_nolayer1, "fit_bounds_0_2_no_layer1")
    pair_cluster_0p045_matched= plot.apply_calibration(pair_cluster_0p045_matched, weight_0p045_no_bounds_nolayer1, "fit_bounds_0_2_no_layer1")
    pair_cluster_Ref_matched= plot.apply_calibration(pair_cluster_Ref_matched, weight_Ref_no_bounds_nolayer1, "fit_bounds_0_2_no_layer1")

    pair_cluster_0p0113_matched= plot.apply_calibration(pair_cluster_0p0113_matched, weight_0p0113_fit_bounds_0_20_no_layer1, "fit_bounds_0_20_no_layer1")
    pair_cluster_0p016_matched= plot.apply_calibration(pair_cluster_0p016_matched, weight_0p016_fit_bounds_0_20_no_layer1, "fit_bounds_0_20_no_layer1")
    pair_cluster_0p03_matched= plot.apply_calibration(pair_cluster_0p03_matched, weight_0p03_fit_bounds_0_20_no_layer1, "fit_bounds_0_20_no_layer1")
    pair_cluster_0p045_matched= plot.apply_calibration(pair_cluster_0p045_matched, weight_0p045_fit_bounds_0_20_no_layer1, "fit_bounds_0_20_no_layer1")
    pair_cluster_Ref_matched= plot.apply_calibration(pair_cluster_Ref_matched, weight_Ref_fit_bounds_0_20_no_layer1, "fit_bounds_0_20_no_layer1")

    pair_cluster_0p0113_matched= plot.apply_calibration(pair_cluster_0p0113_matched, weight_0p0113_no_bounds, "no_bounds")
    pair_cluster_0p016_matched= plot.apply_calibration(pair_cluster_0p016_matched, weight_0p016_no_bounds, "no_bounds")
    pair_cluster_0p03_matched= plot.apply_calibration(pair_cluster_0p03_matched, weight_0p03_no_bounds, "no_bounds")
    pair_cluster_0p045_matched= plot.apply_calibration(pair_cluster_0p045_matched, weight_0p045_no_bounds, "no_bounds")
    pair_cluster_Ref_matched= plot.apply_calibration(pair_cluster_Ref_matched, weight_Ref_no_bounds, "no_bounds")

    pair_cluster_0p0113_matched= plot.apply_calibration(pair_cluster_0p0113_matched, weight_0p0113_no_bounds_nolayer1, "no_bounds_nolayer1")
    pair_cluster_0p016_matched= plot.apply_calibration(pair_cluster_0p016_matched, weight_0p016_no_bounds_nolayer1, "no_bounds_nolayer1")
    pair_cluster_0p03_matched= plot.apply_calibration(pair_cluster_0p03_matched, weight_0p03_no_bounds_nolayer1, "no_bounds_nolayer1")
    pair_cluster_0p045_matched= plot.apply_calibration(pair_cluster_0p045_matched, weight_0p045_no_bounds_nolayer1, "no_bounds_nolayer1")
    pair_cluster_Ref_matched= plot.apply_calibration(pair_cluster_Ref_matched, weight_Ref_no_bounds_nolayer1, "no_bounds_nolayer1")

    name = ["fit_bounds_0_2","fit_bounds_0_20","fit_bounds_0_2_no_layer1","fit_bounds_0_20_no_layer1", "no_bounds", "no_bounds_nolayer1"]
    # name = ["fit_bounds_0_2", "fit_bounds_0_20"]

    print(pair_cluster_0p0113_matched.Ecalib_fit_bounds_0_2)
    print(pair_cluster_0p0113_matched.Ecalib_fit_bounds_0_20)
    print(pair_cluster_0p0113_matched.Ecalib_fit_bounds_0_2_no_layer1)
    print(pair_cluster_0p0113_matched.Ecalib_fit_bounds_0_20_no_layer1)
    print(pair_cluster_0p0113_matched.Ecalib_no_bounds)
    print(pair_cluster_0p0113_matched.Ecalib_no_bounds_nolayer1)

    print(np.max(np.abs(pair_cluster_0p0113_matched.Ecalib_no_bounds - pair_cluster_0p0113_matched.Ecalib_no_bounds_nolayer1)))
    print(np.abs(pair_cluster_0p0113_matched.Ecalib_no_bounds - pair_cluster_0p0113_matched.Ecalib_no_bounds_nolayer1))
    print(np.max(np.abs(pair_cluster_0p0113_matched.Ecalib_fit_bounds_0_2 - pair_cluster_0p0113_matched.Ecalib_fit_bounds_0_2_no_layer1)))
    print(np.abs(pair_cluster_0p0113_matched.Ecalib_fit_bounds_0_2 - pair_cluster_0p0113_matched.Ecalib_fit_bounds_0_2_no_layer1))

    triangles = ["0p0113", "0p016", "0p03", "0p045", "Ref"]
    # bin_pt= 10
    # for tri in triangles:
    #   fig, ax = plt.subplots(figsize=(10,10))
    #   plot.plot_responses(locals()[f"pair_cluster_{tri}_matched"], locals()[f"pair_gen_masked_{tri}"], args, "pT",ax, f"pT for {tri}", my_cmap(0 / (n_colors - 1)), bin_pt, [20,100])
    #   for n, i in zip(name, range(len(name))):
    #     plot.plot_responses(locals()[f"pair_cluster_{tri}_matched"], locals()[f"pair_gen_masked_{tri}"], args, f"Ecalib_{n}",ax, f"{n}", my_cmap((i+1) / (n_colors - 1)),  bin_pt, [20,100], f"{n}")
    #   plt.savefig(f"{output_dir}/calibration/Response_pT_differential_{tri}_calib_comparison.png",dpi=300)
    #   plt.savefig(f"{output_dir}/calibration/Response_pT_differential_{tri}_calib_comparison.pdf")
    #   print(f"{output_dir}/calibration/Response_pT_differential_{tri}_calib_comparison.png")
    #   print(f"{output_dir}/calibration/Response_pT_differential_{tri}_calib_comparison.pdf")
      # plt.close()

    range_eta = [1.6, 2.9]
    range_phi = [-3.14, 3.14]
    bin_eta=10
    bin_phi=10
    for tri in triangles:
      fig, ax = plt.subplots(figsize=(10,10))
      for var, bin, range_ in zip(["eta", "phi"], [bin_eta, bin_phi], [range_eta, range_phi]):
        plot.plot_responses(locals()[f"pair_cluster_{tri}_matched"], locals()[f"pair_gen_masked_{tri}"], args, f"pT_{var}",ax, f"pT for {tri}", my_cmap(0 / (n_colors - 1)), bin, range_)
        for n, i in zip(name, range(len(name))):
          plot.plot_responses(locals()[f"pair_cluster_{tri}_matched"], locals()[f"pair_gen_masked_{tri}"], args, f"Ecalib_{var}",ax, f"{n}", my_cmap((i+1) / (n_colors - 1)),  bin, range_, f"{n}")
        plt.savefig(f"{output_dir}/calibration/Response_pT_{var}_differential_{tri}_calib_comparison.png",dpi=300)
        plt.savefig(f"{output_dir}/calibration/Response_pT_{var}_differential_{tri}_calib_comparison.pdf")
        print(f"{output_dir}/calibration/Response_pT_{var}_differential_{tri}_calib_comparison.png")
        # print(f"{output_dir}/calibration/Response_pT_{var}_differential_{tri}_calib_comparison.pdf")
        plt.close()


    os.makedirs(f"{output_dir}/calibration", exist_ok=True)

    

  if args.calib_test:
    print("In calib test")
    # ------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------
    os.makedirs(f"{output_dir}/calibration", exist_ok=True)
    parquet_dir_wl= f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU0_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV'


    # ------------------------------------------------------------
    # Configuration definitions
    # ------------------------------------------------------------
    calibration_configs = {
        "fit_bounds_0_2": "PU0_bounds_0_2",
        "fit_bounds_0_20": "PU0_bounds_0_20",
        "fit_bounds_0_2_no_layer1": "PU0_bounds_0_2_no_layer1",
        "fit_bounds_0_20_no_layer1": "PU0_bounds_0_20_no_layer1",
        "no_bounds": "PU0_no_bounds",
        "no_bounds_nolayer1": "PU0_no_bounds_no_layer1",
    }

    triangles = ["0p0113", "0p016", "0p03", "0p045", "Ref"]

    # ------------------------------------------------------------
    # Store clusters in dictionary
    # ------------------------------------------------------------
    clusters = {
        "0p0113": pair_cluster_0p0113_matched,
        "0p016": pair_cluster_0p016_matched,
        "0p03": pair_cluster_0p03_matched,
        "0p045": pair_cluster_0p045_matched,
        "Ref": pair_cluster_Ref_matched,
    }

    # ------------------------------------------------------------
    # Store gen objects in dictionary
    # ------------------------------------------------------------
    gen_masked = {
        "0p0113": pair_gen_masked_0p0113,
        "0p016": pair_gen_masked_0p016,
        "0p03": pair_gen_masked_0p03,
        "0p045": pair_gen_masked_0p045,
        "Ref": pair_gen_masked_Ref,
    }

    # ------------------------------------------------------------
    # Load all weights into nested dictionary
    # ------------------------------------------------------------
    weights = {}

    for calib_name, folder in calibration_configs.items():
        weights[calib_name] = {}
        for tri in triangles:
            path = f"{parquet_dir_wl}/{folder}/weight_wl_{tri}.parquet"
            weights[calib_name][tri] = ak.from_parquet(path)

    # ------------------------------------------------------------
    # Apply all calibrations automatically
    # ------------------------------------------------------------
    for tri in triangles:
        for calib_name in calibration_configs.keys():
            clusters[tri] = plot.apply_calibration(
                clusters[tri],
                weights[calib_name][tri],
                calib_name
            )

    # ------------------------------------------------------------
    # Optional: print comparison checks (for debugging)
    # ------------------------------------------------------------
    print("debuggging checks")
    print(np.max(np.abs(
        clusters["0p0113"].Ecalib_no_bounds -
        clusters["0p0113"].Ecalib_no_bounds_nolayer1
    )))

    print(np.max(np.abs(
        clusters["0p0113"].Ecalib_fit_bounds_0_2 -
        clusters["0p0113"].Ecalib_fit_bounds_0_2_no_layer1
    )))

    # ------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------
    range_eta = [1.6, 2.9]
    range_phi = [-3.14, 3.14]
    bin_eta = 10
    bin_phi = 10

    n_colors = len(calibration_configs) + 1

    for tri in triangles:

        for var, bin_, range_ in zip(
            ["eta", "phi"],
            [bin_eta, bin_phi],
            [range_eta, range_phi]
        ):

            fig, ax = plt.subplots(figsize=(10, 10))

            # Plot raw pT
            plot.plot_responses(
                clusters[tri],
                gen_masked[tri],
                args,
                f"pT_{var}",
                ax,
                f"pT for {tri}",
                my_cmap(0 / (n_colors - 1)),
                bin_,
                range_
            )

            # Plot all calibrations
            for i, calib_name in enumerate(calibration_configs.keys()):
                plot.plot_responses(
                    clusters[tri],
                    gen_masked[tri],
                    args,
                    f"Ecalib_{var}",
                    ax,
                    calib_name,
                    my_cmap((i + 1) / (n_colors - 1)),
                    bin_,
                    range_,
                    calib_name
                )

            # Save
            plt.savefig(
                f"{output_dir}/calibration/Response_pT_{var}_differential_{tri}_calib_comparison.png",
                dpi=300
            )
            plt.savefig(
                f"{output_dir}/calibration/Response_pT_{var}_differential_{tri}_calib_comparison.pdf"
            )

            print(f"{output_dir}/calibration/Response_pT_{var}_differential_{tri}_calib_comparison.png")

            plt.close()
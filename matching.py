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

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Stage-2 Emulator Parameters')

  parser.add_argument('-n',          type=int, default=1,         help='Provide the number of events')
  parser.add_argument('--particles', type=str, default='photons', help='Choose the particle sample')
  parser.add_argument('--pileup',    type=str, default='PU0',     help='Choose the pileup - PU0 or PU200')
  parser.add_argument('--base_path', type=str, default='root://eoscms.cern.ch//store/group/dpg_hgcal/comm_hgcal/TPG/stage2_emulator_ntuples_semiemulator_2Passes/double')
  parser.add_argument('--parquet_output', type=str, default="/data_CMS/cms/amella/HGCAL_samples/")
  parser.add_argument('--name_tree',    type=str, default='l1tHGCalTriggerNtuplizer/HGCalTriggerNtuple')
  parser.add_argument('--pt_cut',    type=float, default=0,         help='Provide the cut for the cluster pt')
  parser.add_argument('--parquet',    action='store_true', help='Produce parquet file to plot')

  parser.add_argument('--tag',       type=str, default='',        help='Name to make unique json files')
  parser.add_argument('--var',       type=str, default='pT',        help='Name of the variable to be plotted')
  

  parser.add_argument('--gen_pt_cut',    type=float, default=0,         help='Provide the cut for the gen pt')
  parser.add_argument('--PU0calibration',   action='store_true',         help='Provide the calibration factor')
  parser.add_argument('--PU200calibration',   action='store_true',         help='Provide the calibration factor')
  parser.add_argument('--PileUpcalibration',   action='store_true',         help='Provide the calibration factor')
  parser.add_argument('--PileUpcalibrationFinal',   action='store_true',         help='Provide the calibration factor')
  parser.add_argument('--matching',   action='store_true',         help='Provide the calibration factor')
  parser.add_argument('--total_efficiency',     action='store_true', help='Compute the total efficiency for each emulation test')

  args = parser.parse_args()

  ###########################

  output_dir = f"plots/results_performance_plots_{args.particles}_{args.pileup}_{args.tag}"
  os.makedirs(output_dir, exist_ok=True)

  if args.pt_cut:
    output_dir = f"plots/results_performance_plots_{args.particles}_{args.pileup}_cut_{args.pt_cut}_GeV_{args.tag}"
    os.makedirs(output_dir, exist_ok=True)

  if args.gen_pt_cut:
    output_dir = f"plots/results_performance_plots_{args.particles}_{args.pileup}_gen_cut_{args.gen_pt_cut}_GeV_{args.tag}"
    os.makedirs(output_dir, exist_ok=True)

  if args.gen_pt_cut and args.pt_cut:
    output_dir = f"plots/results_performance_plots_{args.particles}_{args.pileup}_gen_cut_{args.gen_pt_cut}_GeV_cut_{args.pt_cut}_GeV_{args.tag}"
    os.makedirs(output_dir, exist_ok=True)

 ########################

  # Since the dataset is too heavy to load the events every time, we just need to load them once and save them as parquet files

  if args.parquet:
    print(args.base_path)
    events_gen, events_0p0113, events_0p016, events_0p03 , events_0p045, events_Ref =provide_events_performaces(args.n, args.base_path, args.particles, args.pileup, args.pt_cut)

    output_dir = args.parquet_output + args.particles + "_" + args.pileup + "_new_branch"
    if args.gen_pt_cut:
        output_dir += f"gen_pt_cut{args.gen_pt_cut}_GeV"
    os.makedirs(output_dir, exist_ok=True)

    ak.to_parquet(events_gen, os.path.join(output_dir, "events_gen.parquet"))
    ak.to_parquet(events_0p0113, os.path.join(output_dir, "events_0p0113.parquet"))
    ak.to_parquet(events_0p016, os.path.join(output_dir, "events_0p016.parquet"))
    ak.to_parquet(events_0p03, os.path.join(output_dir, "events_0p03.parquet"))
    ak.to_parquet(events_0p045, os.path.join(output_dir, "events_0p045.parquet"))
    ak.to_parquet(events_Ref, os.path.join(output_dir, "events_Ref.parquet"))

  # Once the events have been skimmed and saved in the parquet files we can read them from there
  parquet_dir='/data_CMS/cms/amella/HGCAL_samples/parquet_files/'+args.particles+'_'+args.pileup+'_new_branch/'

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

  if args.matching:
        if args.gen_pt_cut !=0 and args.pt_cut==0:
            parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/'
            os.makedirs(parquet_dir, exist_ok=True)

        if args.pt_cut !=0 and args.gen_pt_cut==0:
            parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/cluster_pt_cut{args.pt_cut}_GeV/'
            os.makedirs(parquet_dir, exist_ok=True)

        if args.pt_cut !=0 and args.gen_pt_cut !=0:
            parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/cluster_pt_cut{args.pt_cut}_GeV_gen_pt_cut{args.gen_pt_cut}_GeV/'
            os.makedirs(parquet_dir, exist_ok=True)

        #Apply matching of clusters and gen partciles requiring DeltaR <0.1 and taking the highest pt cluster
        pair_cluster_0p0113_matched, pair_gen_masked_0p0113, events_0p0113, events_gen_filtered_0p0113 = apply_matching(events_0p0113, 'cl3d_p0113Tri_eta','cl3d_p0113Tri_phi', events_gen, args, deltaR=0.1)
        pair_cluster_0p016_matched, pair_gen_masked_0p016, events_0p016, events_gen_filtered_0p016 = apply_matching(events_0p016, 'cl3d_p016Tri_eta','cl3d_p016Tri_phi', events_gen, args, deltaR=0.1)
        pair_cluster_0p03_matched, pair_gen_masked_0p03, events_0p03, events_gen_filtered_0p03 = apply_matching(events_0p03, 'cl3d_p03Tri_eta','cl3d_p03Tri_phi', events_gen, args, deltaR=0.1)
        pair_cluster_0p045_matched, pair_gen_masked_0p045, events_0p045, events_gen_filtered_0p045 = apply_matching(events_0p045, 'cl3d_p045Tri_eta','cl3d_p045Tri_phi', events_gen, args, deltaR=0.1)
        pair_cluster_Ref_matched, pair_gen_masked_Ref, events_Ref, events_gen_filtered_Ref = apply_matching(events_Ref, 'cl3d_Ref_eta','cl3d_Ref_phi', events_gen, args, deltaR=0.1)

        ak.to_parquet(pair_cluster_0p0113_matched, parquet_dir+"pair_cluster_0p0113_matched.parquet")
        ak.to_parquet(pair_cluster_0p016_matched, parquet_dir+"pair_cluster_0p016_matched.parquet")
        ak.to_parquet(pair_cluster_0p03_matched, parquet_dir+"pair_cluster_0p03_matched.parquet")
        ak.to_parquet(pair_cluster_0p045_matched, parquet_dir+"pair_cluster_0p045_matched.parquet")
        ak.to_parquet(pair_cluster_Ref_matched, parquet_dir+"pair_cluster_Ref_matched.parquet")

        ak.to_parquet(pair_gen_masked_0p0113, parquet_dir+"pair_gen_masked_0p0113.parquet")
        ak.to_parquet(pair_gen_masked_0p016, parquet_dir+"pair_gen_masked_0p016.parquet")
        ak.to_parquet(pair_gen_masked_0p03, parquet_dir+"pair_gen_masked_0p03.parquet")
        ak.to_parquet(pair_gen_masked_0p045, parquet_dir+"pair_gen_masked_0p045.parquet")
        ak.to_parquet(pair_gen_masked_Ref, parquet_dir+"pair_gen_masked_Ref.parquet")

        ak.to_parquet(events_0p0113, parquet_dir+"events_0p0113_filtered.parquet")
        ak.to_parquet(events_0p016, parquet_dir+"events_0p016_filtered.parquet")
        ak.to_parquet(events_0p03, parquet_dir+"events_0p03_filtered.parquet")
        ak.to_parquet(events_0p045, parquet_dir+"events_0p045_filtered.parquet")
        ak.to_parquet(events_Ref, parquet_dir+"events_Ref_filtered.parquet")

        ak.to_parquet(events_gen_filtered_0p0113, parquet_dir+"events_gen_filtered_0p0113.parquet")
        ak.to_parquet(events_gen_filtered_0p016, parquet_dir+"events_gen_filtered_0p016.parquet")
        ak.to_parquet(events_gen_filtered_0p03, parquet_dir+"events_gen_filtered_0p03.parquet")
        ak.to_parquet(events_gen_filtered_0p045, parquet_dir+"events_gen_filtered_0p045.parquet")
        ak.to_parquet(events_gen_filtered_Ref, parquet_dir+"events_gen_filtered_Ref.parquet")
  else:
        parquert_dir='/data_CMS/cms/amella/HGCAL_samples/parquet_files/'+args.particles+'_'+args.pileup+'_new_branch/'  
        if args.gen_pt_cut !=0 and args.pt_cut==0:
            parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/'

        pair_cluster_0p0113_matched = ak.from_parquet(parquet_dir+"pair_cluster_0p0113_matched.parquet")
        pair_cluster_0p016_matched = ak.from_parquet(parquet_dir+"pair_cluster_0p016_matched.parquet")
        pair_cluster_0p03_matched = ak.from_parquet(parquet_dir+"pair_cluster_0p03_matched.parquet")
        pair_cluster_0p045_matched = ak.from_parquet(parquet_dir+"pair_cluster_0p045_matched.parquet")
        pair_cluster_Ref_matched = ak.from_parquet(parquet_dir+"pair_cluster_Ref_matched.parquet")

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


        # events_gen_filtered_0p0113 = ak.from_parquet(parquet_dir+"events_gen_filtered_0p0113.parquet")
        # events_gen_filtered_0p016 = ak.from_parquet(parquet_dir+"events_gen_filtered_0p016.parquet")
        # events_gen_filtered_0p03 = ak.from_parquet(parquet_dir+"events_gen_filtered_0p03.parquet")
        # events_gen_filtered_0p045 = ak.from_parquet(parquet_dir+"events_gen_filtered_0p045.parquet")
        # events_gen_filtered_Ref = ak.from_parquet(parquet_dir+"events_gen_filtered_Ref.parquet")

        # for i in [events_gen_filtered_0p0113, events_gen_filtered_0p016, events_gen_filtered_0p03, events_gen_filtered_0p045, events_gen_filtered_Ref]:
        #     print("Length of events:", len(i))
        #     print(min(ak.flatten(i.pt,axis=-1)), max(ak.flatten(i.pt,axis=-1)))

        # for i in [events_0p0113, events_0p016, events_0p03, events_0p045, events_Ref]:
        #     print("Length of events:", len(i))
        #     print(min(ak.flatten(i.pt,axis=-1)), max(ak.flatten(i.pt,axis=-1)))

  if args.PU0calibration:
        # We wannt to derive a calibration given by: E_calib= sum w_l E_l - (a eta + b)
        # Where E-L is the energy per layer and the wl are the calib factors to apply to each layer
        # In the following we only apply the calibration in the CE-E layers(first 13 trigger layers)
        weights=[]
        for events, gen in zip([pair_cluster_0p0113_matched,pair_cluster_0p016_matched,
                                pair_cluster_0p03_matched,pair_cluster_0p045_matched,
                                pair_cluster_Ref_matched],
                                [pair_gen_masked_0p0113, pair_gen_masked_0p016, 
                                pair_gen_masked_0p03,pair_gen_masked_0p045, 
                                pair_gen_masked_Ref]):
            weight= plot.derive_calibration(events, gen, args)
            weights.append(weight)

        weight_wl_0p0113 = weights[0]
        weight_wl_0p016 = weights[1]
        weight_wl_0p03 = weights[2]
        weight_wl_0p045 = weights[3]
        weight_wl_Ref = weights[4]

        print("Derived calibration weights wl:")
        print("0.0113:", weight_wl_0p0113)
        print("0.016:", weight_wl_0p016)
        print("0.03:", weight_wl_0p03)
        print("0.045:", weight_wl_0p045)
        print("Ref:", weight_wl_Ref)

        print("Shape of weight_wl_0p0113:", weight_wl_0p0113.shape)

        # parquet_dir='/data_CMS/cms/amella/HGCAL_samples/parquet_files/'+args.particles+'_'+args.pileup+'_new_branch/'
        # parquet_dir='/data_CMS/cms/amella/HGCAL_samples/parquet_files/'+args.particles+'_'+args.pileup+'/'
        if args.tag and args.gen_pt_cut!=0:
            parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/{args.tag}/'
            os.makedirs(parquet_dir, exist_ok=True)
        print(parquet_dir)
        if os.path.exists(parquet_dir):
            print("Opening", parquet_dir)
            ak.to_parquet(weight_wl_0p0113, os.path.join(parquet_dir, "weight_wl_0p0113.parquet"))
            ak.to_parquet(weight_wl_0p016, os.path.join(parquet_dir, "weight_wl_0p016.parquet"))
            ak.to_parquet(weight_wl_0p03, os.path.join(parquet_dir, "weight_wl_0p03.parquet"))
            ak.to_parquet(weight_wl_0p045, os.path.join(parquet_dir, "weight_wl_0p045.parquet"))
            ak.to_parquet(weight_wl_Ref, os.path.join(parquet_dir, "weight_wl_Ref.parquet"))

  if args.PU200calibration:
    # We want to derive a calibration given by: E_calib= sum w_l E_l - (a eta + b)
    # Where E-L is the energy per layer and the wl are the calib factors to apply to each layer
    # In the following we only apply the calibration in the CE-E layers(first 13 trigger layers)
    # we derive the thriteen wl factors and the two parameters a and b of the eta dependent term together in a single step by minimizing the resolution on the E_gen-E_calib distribution
    weights=[]
    if args.tag and args.gen_pt_cut!=0:
        parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/{args.tag}/'
        os.makedirs(parquet_dir, exist_ok=True)

    print(parquet_dir)
    for events, gen in zip([pair_cluster_0p0113_matched,pair_cluster_0p016_matched,
                            pair_cluster_0p03_matched,pair_cluster_0p045_matched,
                            pair_cluster_Ref_matched],
                            [pair_gen_masked_0p0113, pair_gen_masked_0p016, 
                            pair_gen_masked_0p03,pair_gen_masked_0p045, 
                            pair_gen_masked_Ref]):
        weight= plot.derive_calibration(events, gen, args, 0,20)
        weights.append(weight)

    weight_wl_0p0113 = weights[0]
    weight_wl_0p016 = weights[1]
    weight_wl_0p03 = weights[2]
    weight_wl_0p045 = weights[3]
    weight_wl_Ref = weights[4]

    print("Derived calibration weights wl, alpha and beta:")
    print("0.0113:", weight_wl_0p0113)
    print("0.016:", weight_wl_0p016)
    print("0.03:", weight_wl_0p03)
    print("0.045:", weight_wl_0p045)
    print("Ref:", weight_wl_Ref)
    print("Shape of weight_wl_0p0113:", weight_wl_0p0113.shape)

    # parquet_dir='/data_CMS/cms/amella/HGCAL_samples/parquet_files/'+args.particles+'_'+args.pileup+'_new_branch/'
    # parquet_dir='/data_CMS/cms/amella/HGCAL_samples/parquet_files/'+args.particles+'_'+args.pileup+'/'
    if os.path.exists(parquet_dir):
        print("Opening", parquet_dir)
        ak.to_parquet(weight_wl_0p0113, os.path.join(parquet_dir, "weight_w_0p0113_PU200.parquet"))
        ak.to_parquet(weight_wl_0p016, os.path.join(parquet_dir, "weight_w_0p016_PU200.parquet"))
        ak.to_parquet(weight_wl_0p03, os.path.join(parquet_dir, "weight_w_0p03_PU200.parquet"))
        ak.to_parquet(weight_wl_0p045, os.path.join(parquet_dir, "weight_w_0p045_PU200.parquet"))
        ak.to_parquet(weight_wl_Ref, os.path.join(parquet_dir, "weight_w_Ref_PU200.parquet"))


    pair_cluster_0p0113_matched= plot.apply_calibration_all_weights(pair_cluster_0p0113_matched, weight_wl_0p0113, args.tag)
    pair_cluster_0p016_matched= plot.apply_calibration_all_weights(pair_cluster_0p016_matched, weight_wl_0p016, args.tag)
    pair_cluster_0p03_matched= plot.apply_calibration_all_weights(pair_cluster_0p03_matched, weight_wl_0p03, args.tag)
    pair_cluster_0p045_matched= plot.apply_calibration_all_weights(pair_cluster_0p045_matched, weight_wl_0p045, args.tag)
    pair_cluster_Ref_matched= plot.apply_calibration_all_weights(pair_cluster_Ref_matched, weight_wl_Ref, args.tag)

    print("Applied pileup calibration")
    # print(pair_cluster_0p0113_matched.Ecalib_all)
    print(getattr(pair_cluster_0p0113_matched, f"Ecalib_all_{args.tag}"))
    print(pair_gen_masked_0p0113.pt)
    

  if args.PileUpcalibration:
    if args.tag and args.gen_pt_cut!=0:
        parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/{args.tag}/'
        os.makedirs(parquet_dir, exist_ok=True)
    # We wannt to derive a calibration given by: E_calib= sum w_l E_l - (a eta + b)
    # In the following we derive the coefficients a and b

    if args.gen_pt_cut !=0 and args.pt_cut==0:
        parquet_dir_PU0=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU0_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/'
    else:
        parquet_dir_PU0=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU0_new_branch/'
  
    # weight_dir = "PU0_bounds_0_20/"
    name="fit_no_bounds"
    # weight_dir = "PU0_bounds_0_20_no_layer1/"
    weight_dir = "PU0_no_bounds/"
    # weight_dir = "PU0_no_bounds_no_layer1/"
    if os.path.exists(parquet_dir_PU0+"/weight_wl_0p0113.parquet"):
        weight_0p0113 = ak.from_parquet(parquet_dir_PU0+ weight_dir+"weight_wl_0p0113.parquet")
        weight_0p016 = ak.from_parquet(parquet_dir_PU0+ weight_dir+"weight_wl_0p016.parquet")
        weight_0p03 = ak.from_parquet(parquet_dir_PU0+ weight_dir+"weight_wl_0p03.parquet")
        weight_0p045 = ak.from_parquet(parquet_dir_PU0+ weight_dir+"weight_wl_0p045.parquet")
        weight_Ref = ak.from_parquet(parquet_dir_PU0+ weight_dir+"weight_wl_Ref.parquet")

        print("Length of weight_0p0113:", len(weight_0p0113))

        # Apply calibration
        pair_cluster_0p0113_matched= plot.apply_calibration(pair_cluster_0p0113_matched, weight_0p0113, name)
        pair_cluster_0p016_matched= plot.apply_calibration(pair_cluster_0p016_matched, weight_0p016, name)
        pair_cluster_0p03_matched= plot.apply_calibration(pair_cluster_0p03_matched, weight_0p03, name)
        pair_cluster_0p045_matched= plot.apply_calibration(pair_cluster_0p045_matched, weight_0p045, name)
        pair_cluster_Ref_matched= plot.apply_calibration(pair_cluster_Ref_matched, weight_Ref, name)

    weights=[]
    for events, gen in zip([pair_cluster_0p0113_matched,pair_cluster_0p016_matched,
                            pair_cluster_0p03_matched,pair_cluster_0p045_matched,
                            pair_cluster_Ref_matched],
                            [pair_gen_masked_0p0113, pair_gen_masked_0p016, 
                            pair_gen_masked_0p03,pair_gen_masked_0p045, 
                            pair_gen_masked_Ref]):
        weight= plot.derive_calibration(events, gen, args, -np.inf,np.inf, name)
        weights.append(weight)

    weight_a_0p0113 = weights[0]
    weight_a_0p016 = weights[1]
    weight_a_0p03 = weights[2]
    weight_a_0p045 = weights[3]
    weight_a_Ref = weights[4]

    print("Derived calibration factors a:")
    print("0.0113:", weight_a_0p0113)
    print("0.016:", weight_a_0p016)
    print("0.03:", weight_a_0p03)   
    print("0.045:", weight_a_0p045)
    print("Ref:", weight_a_Ref)


    # parquet_dir='/data_CMS/cms/amella/HGCAL_samples/parquet_files/'+args.particles+'_'+args.pileup+'_new_branch/'
    # parquet_dir='/data_CMS/cms/amella/HGCAL_samples/parquet_files/'+args.particles+'_'+args.pileup+'/'
    if os.path.exists(parquet_dir):
        print("Opening", parquet_dir)
        ak.to_parquet(ak.Array([weight_a_0p0113]), os.path.join(parquet_dir, "weight_a_0p0113.parquet"))
        ak.to_parquet(ak.Array([weight_a_0p016]),   os.path.join(parquet_dir, "weight_a_0p016.parquet"))
        ak.to_parquet(ak.Array([weight_a_0p03]),    os.path.join(parquet_dir, "weight_a_0p03.parquet"))
        ak.to_parquet(ak.Array([weight_a_0p045]),   os.path.join(parquet_dir, "weight_a_0p045.parquet"))
        ak.to_parquet(ak.Array([weight_a_Ref]),     os.path.join(parquet_dir, "weight_a_Ref.parquet"))
    
    pair_cluster_0p0113_matched= plot.apply_calibration_eta(pair_cluster_0p0113_matched, weight_a_0p0113, name)
    pair_cluster_0p016_matched= plot.apply_calibration_eta(pair_cluster_0p016_matched, weight_a_0p016, name)
    pair_cluster_0p03_matched= plot.apply_calibration_eta(pair_cluster_0p03_matched, weight_a_0p03, name)
    pair_cluster_0p045_matched= plot.apply_calibration_eta(pair_cluster_0p045_matched, weight_a_0p045, name)
    pair_cluster_Ref_matched= plot.apply_calibration_eta(pair_cluster_Ref_matched, weight_a_Ref, name)

    print("Applied pileup calibration")
    print(getattr(pair_cluster_0p0113_matched, f"Ecalib_PU_term{name}"))
    print(getattr(pair_cluster_0p0113_matched, f"Ecalib_{name}"))
    print(pair_gen_masked_0p0113.pt)


if args.PileUpcalibrationFinal:
    if args.tag and args.gen_pt_cut!=0:
        parquet_dir=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_{args.pileup}_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/{args.tag}/'
        os.makedirs(parquet_dir, exist_ok=True)
    # We wannt to derive a calibration given by: E_calib= sum w_l E_l - (a eta + b)
    # In the following we derive the coefficients a and b

    if args.gen_pt_cut !=0 and args.pt_cut==0:
        parquet_dir_PU0=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU0_new_branch/gen_pt_cut{args.gen_pt_cut}_GeV/'
    else:
        parquet_dir_PU0=f'/data_CMS/cms/amella/HGCAL_samples/parquet_files/{args.particles}_PU0_new_branch/'
  
    # weight_dir = "PU0_bounds_0_20/"
    name="fit_bounds_0_20"
    # weight_dir = "PU0_bounds_0_20_no_layer1/"
    weight_dir = "PU0_bounds_0_20/"
    # weight_dir = "PU0_no_bounds_no_layer1/"

    if os.path.exists(parquet_dir_PU0 + weight_dir +"/weight_wl_0p0113.parquet"):
        weight_0p0113 = ak.from_parquet(parquet_dir_PU0+ weight_dir+"weight_wl_0p0113.parquet")
        weight_0p016 = ak.from_parquet(parquet_dir_PU0+ weight_dir+"weight_wl_0p016.parquet")
        weight_0p03 = ak.from_parquet(parquet_dir_PU0+ weight_dir+"weight_wl_0p03.parquet")
        weight_0p045 = ak.from_parquet(parquet_dir_PU0+ weight_dir+"weight_wl_0p045.parquet")
        weight_Ref = ak.from_parquet(parquet_dir_PU0+ weight_dir+"weight_wl_Ref.parquet")

        print("Length of weight_0p0113:", len(weight_0p0113))

        # Apply calibration
        pair_cluster_0p0113_matched= plot.apply_calibration(pair_cluster_0p0113_matched, weight_0p0113, name)
        pair_cluster_0p016_matched= plot.apply_calibration(pair_cluster_0p016_matched, weight_0p016, name)
        pair_cluster_0p03_matched= plot.apply_calibration(pair_cluster_0p03_matched, weight_0p03, name)
        pair_cluster_0p045_matched= plot.apply_calibration(pair_cluster_0p045_matched, weight_0p045, name)
        pair_cluster_Ref_matched= plot.apply_calibration(pair_cluster_Ref_matched, weight_Ref, name)

    weights=[]
    for events, gen in zip([pair_cluster_0p0113_matched,pair_cluster_0p016_matched,
                            pair_cluster_0p03_matched,pair_cluster_0p045_matched,
                            pair_cluster_Ref_matched],
                            [pair_gen_masked_0p0113, pair_gen_masked_0p016, 
                            pair_gen_masked_0p03,pair_gen_masked_0p045, 
                            pair_gen_masked_Ref]):
        weight= plot.derive_calibration(events, gen, args, -np.inf,np.inf, name)
        weights.append(weight)

    weight_a_0p0113 = weights[0]
    weight_a_0p016 = weights[1]
    weight_a_0p03 = weights[2]
    weight_a_0p045 = weights[3]
    weight_a_Ref = weights[4]

    print("Derived calibration factors a:")
    print("0.0113:", weight_a_0p0113)
    print("0.016:", weight_a_0p016)
    print("0.03:", weight_a_0p03)   
    print("0.045:", weight_a_0p045)
    print("Ref:", weight_a_Ref)


    # parquet_dir='/data_CMS/cms/amella/HGCAL_samples/parquet_files/'+args.particles+'_'+args.pileup+'_new_branch/'
    # parquet_dir='/data_CMS/cms/amella/HGCAL_samples/parquet_files/'+args.particles+'_'+args.pileup+'/'
    if os.path.exists(parquet_dir):
        print("Opening", parquet_dir)
        ak.to_parquet(ak.Array([weight_a_0p0113]), os.path.join(parquet_dir, "weight_a_0p0113.parquet"))
        ak.to_parquet(ak.Array([weight_a_0p016]),   os.path.join(parquet_dir, "weight_a_0p016.parquet"))
        ak.to_parquet(ak.Array([weight_a_0p03]),    os.path.join(parquet_dir, "weight_a_0p03.parquet"))
        ak.to_parquet(ak.Array([weight_a_0p045]),   os.path.join(parquet_dir, "weight_a_0p045.parquet"))
        ak.to_parquet(ak.Array([weight_a_Ref]),     os.path.join(parquet_dir, "weight_a_Ref.parquet"))
    
    pair_cluster_0p0113_matched= plot.apply_calibration_eta(pair_cluster_0p0113_matched, weight_a_0p0113, name)
    pair_cluster_0p016_matched= plot.apply_calibration_eta(pair_cluster_0p016_matched, weight_a_0p016, name)
    pair_cluster_0p03_matched= plot.apply_calibration_eta(pair_cluster_0p03_matched, weight_a_0p03, name)
    pair_cluster_0p045_matched= plot.apply_calibration_eta(pair_cluster_0p045_matched, weight_a_0p045, name)
    pair_cluster_Ref_matched= plot.apply_calibration_eta(pair_cluster_Ref_matched, weight_a_Ref, name)

    print("Applied pileup calibration")
    print(getattr(pair_cluster_0p0113_matched, f"Ecalib_PU_term_"))
    print(getattr(pair_cluster_0p0113_matched, f"Ecalib_{name}"))
    print(pair_gen_masked_0p0113.pt)




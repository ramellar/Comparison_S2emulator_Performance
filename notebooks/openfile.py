#test
import os
import argparse
import awkward as ak
from   data_handling.event_performances import apply_matching
from data_handling.utils import build_parquet_dir
from configs.config import PARQUET_BASE, EMU_CONFIG
import data_handling.files as io
from data_handling.efficiency import compute_efficiencies

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Stage-2 Emulator Parameters')

    parser.add_argument('--particles', type=str, default='photons', help='Choose the particle sample')
    parser.add_argument('--pileup',    type=str, default='PU0',     help='Choose the pileup - PU0 or PU200')
    parser.add_argument('--pt_cut',    type=float, default=0,         help='Provide the cut for the cluster pt')
    parser.add_argument('--gen_pt_cut',    type=float, default=0,         help='Provide the cut for the gen pt')
    parser.add_argument('--deltaR',    type=float, default=0.1,         help='DeltaR')
    parser.add_argument('--total_efficiency',     action='store_true', help='Compute the total efficiency for each emulation test')

    args = parser.parse_args()

    base = f"{PARQUET_BASE}{args.particles}_{args.pileup}_new_branch/"
    print(base)
    events, events_gen = io.load_events(base)
    output_dir= build_parquet_dir(args)
    
    #print(events_gen.pt)

    results = {}
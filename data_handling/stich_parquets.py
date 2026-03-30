import awkward as ak
import glob
import argparse
import os
from configs.config import EMU_CONFIG, PARQUET_BASE, EVENT_NAMES

def merge_results(args):
    for name in EVENT_NAMES:
        print(f"--- Processing {name} ---")
        
        # Find all files for this specific category (part_0 to part_9)
        search_pattern = os.path.join(BASE_DIR, f"{name}_part_*.parquet")
        print(search_pattern)
        files = sorted(glob.glob(search_pattern))
        
        if not files:
            print(f"No files found for {name}. Skipping...")
            continue
            
        print(f" Found {len(files)} parts. Merging...")
        
        # Load and concatenate all parts into one Awkward Array
        try:
            combined = ak.concatenate([ak.from_parquet(f) for f in files])
            output_path = os.path.join(PARQUET_BASE + args.particles + "_" + args.pileup + "_new_branch/", f"{name}.parquet")
            ak.to_parquet(combined, output_path)
            
            print(f"Successfully created: {output_path}")
            print(f"Total events: {len(combined)}")

            for f in files: os.remove(f) 
            print(f"Deleted {len(files)} part files.")
        except Exception as e:
            print(f"Error merging {name}: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Stage-2 Emulator Parameters')

    parser.add_argument('-n',          type=int, default=1,         help='Provide the number of events')
    parser.add_argument('--particles', type=str, default='Photon', help='Choose the particle sample')
    parser.add_argument('--pileup',    type=str, default='PU0',     help='Choose the pileup - PU0 or PU200')
    parser.add_argument('--base_path', type=str, default='root://eoscms.cern.ch//store/group/dpg_hgcal/comm_hgcal/TPG/stage2_emulator_ntuples_semiemulator_2Passes/double')
    parser.add_argument('--name_tree',    type=str , default='l1tHGCalTriggerNtuplizer/HGCalTriggerNtuple')
    parser.add_argument('--pt_cut',    type=float, default=0,         help='Provide the cut for the cluster pt')
    parser.add_argument('--n_files',    type=float, default=976,         help='Provide the cut for the cluster pt')
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()

    BASE_DIR = PARQUET_BASE+ args.particles + "_" + args.pileup + "_new_branch/"  +"single_jobs/"
    
    merge_results(args)

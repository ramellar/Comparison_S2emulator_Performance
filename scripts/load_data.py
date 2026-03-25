import os
import argparse
import awkward as ak
from data_handling.event_performances import provide_events_performaces
from configs.config import PARQUET_BASE, EVENT_NAMES

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Stage-2 Emulator Parameters')

  parser.add_argument('-n',          type=int, default=1,         help='Provide the number of events')
  parser.add_argument('--particles', type=str, default='photons', help='Choose the particle sample')
  parser.add_argument('--pileup',    type=str, default='PU0',     help='Choose the pileup - PU0 or PU200')
  parser.add_argument('--base_path', type=str, default='root://eoscms.cern.ch//store/group/dpg_hgcal/comm_hgcal/TPG/stage2_emulator_ntuples_semiemulator_2Passes/double')
  parser.add_argument('--name_tree',    type=str , default='l1tHGCalTriggerNtuplizer/HGCalTriggerNtuple')
  parser.add_argument('--pt_cut',    type=float, default=0,         help='Provide the cut for the cluster pt')
  parser.add_argument('--n_files',    type=float, default=976,         help='Provide the cut for the cluster pt')
  parser.add_argument('--job_id', type=int, default=0)
  parser.add_argument('--n_jobs', type=int, default=1)
  args = parser.parse_args()

 # Since the dataset is too heavy to load the events every time, we just need to load them once and save them as parquet files

  print(args.base_path)
  events_gen, events_0p0113, events_0p016, events_0p03 , events_0p045, events_Ref =provide_events_performaces(args.n, args.base_path, args.particles, args.pileup, args.n_files, args.pt_cut, args.job_id, args.n_jobs)
  events= [events_gen, events_0p0113, events_0p016, events_0p03 , events_0p045, events_Ref]

  output_dir = PARQUET_BASE + "/" + args.particles + "_" + args.pileup + "_new_branch/"  + "single_jobs/"
  
  os.makedirs(output_dir, exist_ok=True)

  for event, name in zip(events, EVENT_NAMES):
    ak.to_parquet(event, output_dir + f"/{name}_part_{args.job_id}.parquet")
    print(f"Saved chunk {args.job_id} to {output_dir}")

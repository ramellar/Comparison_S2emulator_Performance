import argparse

from io import load_events, save_matching_results, load_matching_results
from matching import run_matching
from efficiency import compute_efficiencies
from utils import build_parquet_dir


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--particles', default="photons")
    parser.add_argument('--pileup', default="PU0")
    parser.add_argument('--pt_cut', type=float, default=0)
    parser.add_argument('--gen_pt_cut', type=float, default=0)

    parser.add_argument('--matching', action="store_true")
    parser.add_argument('--total_efficiency', action="store_true")

    args = parser.parse_args()

    parquet_dir = build_parquet_dir(args)

    events, events_gen = load_events(parquet_dir)

    if args.total_efficiency:

        compute_efficiencies(events, events_gen, args)

    if args.matching:

        results = run_matching(events, events_gen, args)

        save_matching_results(results, events, parquet_dir)

    else:

        results = load_matching_results(parquet_dir)


if __name__ == "__main__":
    main()

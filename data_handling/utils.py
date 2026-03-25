import os
from configs.config import PARQUET_BASE


def build_parquet_dir(args):

    base = f"{PARQUET_BASE}{args.particles}_{args.pileup}_new_branch"

    if args.pt_cut and args.gen_pt_cut:
        base += f"/cluster_pt_cut{args.pt_cut}_GeV_gen_pt_cut{args.gen_pt_cut}_GeV"
    elif args.pt_cut:
        base += f"/cluster_pt_cut{args.pt_cut}_GeV"
    elif args.gen_pt_cut:
        base += f"/gen_pt_cut{args.gen_pt_cut}_GeV"

    os.makedirs(base, exist_ok=True)

    return base + "/"


def build_plotting_dir(args):

    base = f"plots_{args.pileup}"

    if args.matched:
        base += f"/matched_cluster"
    elif args.events:
        base += f"/bare_cluster"
    elif args.filtered_events:
        base += f"/filetered_cluster"

    if args.pt_cut and args.gen_pt_cut:
        base += f"/cluster_pt_cut{args.pt_cut}_GeV_gen_pt_cut{args.gen_pt_cut}_GeV"
    elif args.pt_cut:
        base += f"/cluster_pt_cut{args.pt_cut}_GeV"
    elif args.gen_pt_cut:
        base += f"/gen_pt_cut{args.gen_pt_cut}_GeV"

    os.makedirs(base, exist_ok=True)

    return base + "/"

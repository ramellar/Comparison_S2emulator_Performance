"""Merge HTCondor chunks produced by ``scripts.matching_test``.

The input chunks are expected in ``<matching output>/parts`` with names such
as ``pair_cluster_0p0113_matched_part_0.parquet``.  The merged files are
written to the matching output directory, using the standard names consumed
by ``load_matching_results`` and the plotting scripts.
"""

import argparse
import glob
import os

import awkward as ak

from configs.config import EMU_CONFIG
from data_handling.utils import build_parquet_dir


RESULT_FILES = {
    "pair_cluster": "pair_cluster_{key}_matched{matching_object}_part_*.parquet",
    "pair_gen": "pair_gen_masked_{key}{matching_object}_part_*.parquet",
    "events_filtered": "events_{key}_filtered{matching_object}_part_*.parquet",
    "events_gen_filtered": "events_gen_filtered_{key}{matching_object}_part_*.parquet",
    "events_gen_denominator": "events_gen_denominator_{key}{matching_object}_part_*.parquet",
    "antikt_diagnostics": "antikt_diagnostics_{key}{matching_object}_part_*.parquet",
}

OUTPUT_FILES = {
    "pair_cluster": "pair_cluster_{key}_matched{matching_object}.parquet",
    "pair_gen": "pair_gen_masked_{key}{matching_object}.parquet",
    "events_filtered": "events_{key}_filtered{matching_object}.parquet",
    "events_gen_filtered": "events_gen_filtered_{key}{matching_object}.parquet",
    "events_gen_denominator": "events_gen_denominator_{key}{matching_object}.parquet",
    "antikt_diagnostics": "antikt_diagnostics_{key}{matching_object}.parquet",
    "taucalib_fit": "taucalib_fit_{key}.parquet",
}


def merge_matching_parts(output_dir, matching_object, overwrite=False):
    """Concatenate all matching chunks found in ``output_dir/parts``."""
    parts_dir = os.path.join(output_dir, "parts")

    if not os.path.isdir(parts_dir):
        raise RuntimeError("Error: No chunks to stitch")

    # Check before writing anything.  This also makes a second stitch attempt
    # fail immediately after the chunks have been removed.
    chunk_files = sorted({
        path
        for key in EMU_CONFIG
        for template in RESULT_FILES.values()
        for path in glob.glob(os.path.join(parts_dir, template.format(key=key, matching_object=matching_object)))
    })
    if not chunk_files:
        raise RuntimeError("Error: No chunks to stitch")

    for key in EMU_CONFIG:
        for result_name, input_template in RESULT_FILES.items():
            input_pattern = os.path.join(
                parts_dir,
                input_template.format(key=key, matching_object=matching_object),
            )
            input_files = sorted(glob.glob(input_pattern))

            if not input_files:
                if result_name in {"taucalib_fit", "antikt_diagnostics"}:
                    continue
                raise FileNotFoundError(
                    f"No chunks found for {result_name}, triangle {key}: "
                    f"{input_pattern}"
                )

            output_path = os.path.join(
                output_dir,
                OUTPUT_FILES[result_name].format(key=key, matching_object=matching_object),
            )

            if os.path.exists(output_path) and not overwrite:
                raise FileExistsError(
                    f"Output already exists: {output_path}. "
                    "Use --overwrite to replace it."
                )

            print(
                f"Merging {len(input_files)} chunks for "
                f"{result_name}, triangle {key}"
            )
            merged = ak.concatenate([ak.from_parquet(path) for path in input_files])
            ak.to_parquet(merged, output_path)
            print(f"  wrote {len(merged)} events: {output_path}")

    # Remove chunks only after every merged file was written successfully.
    for path in chunk_files:
        os.remove(path)
    print(f"Removed {len(chunk_files)} chunk files from {parts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge matching parquet chunks produced by HTCondor"
    )
    parser.add_argument("--particles", type=str, default="photons")
    parser.add_argument("--pileup", type=str, default="PU0")
    parser.add_argument("--pt_cut", type=float, default=0)
    parser.add_argument("--gen_pt_cut", type=float, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing merged parquet files",)
    parser.add_argument("--matching_type", type=str, choices=["gen_cluster", "antikt_jets"], default="gen_cluster", help="Choose the matching strategy: 1-to-1 gen-cluster or matching with anti-kt reconstructed jets.")
    
    args = parser.parse_args()
    
    matching_object = {"gen_cluster": "_cluster", "antikt_jets": "_jet"}[args.matching_type]

    output_dir = build_parquet_dir(args)
    try:
        merge_matching_parts(output_dir, matching_object=matching_object, overwrite=args.overwrite)
    except RuntimeError as error:
        raise SystemExit(str(error))

"""Apply the tau calibration to the matched raw jets."""

import argparse
import os

import awkward as ak

from configs.config import EMU_CONFIG
from data_handling.event_performances import taucalib_function_offset, taucalib_function_MC
from data_handling.utils import build_parquet_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the tau calibration to matched raw jets")
    parser.add_argument("--particles", default="TauTau")
    parser.add_argument("--pileup", default="PU200")
    parser.add_argument("--pt_cut", type=float, default=0)
    parser.add_argument("--gen_pt_cut", type=float, default=20.0)
    parser.add_argument("--calibration", choices=["offset", "MC"], default="offset", help="Choose the calibration method: offset or offset+MC.")
    args = parser.parse_args()

    parquet_dir = build_parquet_dir(args)

    for key in EMU_CONFIG:
        # Load only the two matched raw files needed for the calibration.
        cluster = ak.from_parquet(os.path.join(parquet_dir, f"pair_cluster_{key}_matched_jet.parquet"))
        gen = ak.from_parquet(os.path.join(parquet_dir, f"pair_gen_masked_{key}_jet.parquet"))

        calibrated, fit = taucalib_function_offset(cluster, gen)
        
        if args.calibration == "MC":
            calibrated, mc_fit, response_2d = taucalib_function_MC(calibrated, gen)

        # Write back the matched array, keeping pt raw and adding pt_corrected.
        ak.to_parquet(
            calibrated,
            os.path.join(parquet_dir, f"pair_cluster_{key}_matched_jet.parquet"),
        )

        # Awkward array with beta(|eta|) points and the cubic-fit parameters.
        ak.to_parquet(
            ak.Array([fit]),
            os.path.join(parquet_dir, f"taucalib_fit_{key}.parquet"),
        )
        
        if args.calibration == "MC":
            ak.to_parquet(
                ak.Array([mc_fit]),
                os.path.join(parquet_dir, f"taucalib_mc_fit_{key}.parquet"),
            )
            ak.to_parquet(
                response_2d,
                os.path.join(parquet_dir, f"taucalib_mc_response_{key}.parquet"),
            )
        
        print(f"Added pt_corrected to the matched array for triangle {key}")

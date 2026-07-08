#!/usr/bin/env python3
"""
Scan logs/*.err files for failed jobs and optionally resubmit them via HTCondor.

Usage:
  python3 check_and_resubmit.py           # just report
  python3 check_and_resubmit.py --submit  # report + submit failed jobs
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "logs"
SUBMIT_FILE = Path(__file__).parent / "submit_load.sub"
PROXY_PATH = "/home/llr/cms/amella/.globus/user_proxy.pem"

# Keywords that indicate a job failed
ERROR_PATTERNS = re.compile(r"Traceback|Error|Exception|FATAL|Segmentation fault", re.IGNORECASE)
# Filename pattern: load_{Part}_{PU}_{JobID}.err
FILE_PATTERN = re.compile(r"load_([A-Za-z]+)_(PU\d+)_(\d+)\.err$")


def load_job_params():
    """Parse submit_load.sub and return {(Part, PU, JobID): (NJobs, NFiles)}."""
    params = {}
    if not SUBMIT_FILE.exists():
        return params
    in_queue = False
    for line in SUBMIT_FILE.read_text().splitlines():
        stripped = line.strip()
        if re.match(r"queue\b", stripped, re.IGNORECASE) and "from" in stripped.lower():
            in_queue = True
            continue
        if in_queue:
            if stripped == ")":
                break
            if not stripped or stripped.startswith("#"):
                continue
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) >= 5:
                part, pu, job_id = parts[0], parts[1], int(parts[2])
                n_jobs, n_files = int(parts[3]), int(parts[4])
                params[(part, pu, job_id)] = (n_jobs, n_files)
    return params


def check_err_files():
    failed = []
    ok = []

    err_files = sorted(LOGS_DIR.glob("*.err"))
    if not err_files:
        print(f"No .err files found in {LOGS_DIR}")
        sys.exit(0)

    for path in err_files:
        m = FILE_PATTERN.match(path.name)
        if not m:
            continue  # skip files with unexpected names (e.g. load_Photon_PU200.err without job_id)
        part, pu, job_id = m.group(1), m.group(2), int(m.group(3))
        content = path.read_text(errors="replace")
        if ERROR_PATTERNS.search(content):
            # Extract the first error line for display
            first_error = next(
                (l.strip() for l in content.splitlines() if ERROR_PATTERNS.search(l)),
                "unknown error"
            )
            failed.append((part, pu, job_id, path.name, first_error))
        else:
            ok.append((part, pu, job_id, path.name))

    return failed, ok


def make_resubmit_file(failed, job_params):
    """Write a condor submit file targeting only the failed jobs."""
    lines = [
        "# Auto-generated resubmit file for failed jobs",
        "universe = vanilla",
        "executable = run_load_data.sh",
        "",
        "arguments = $(ProxyPath) -n 9999999 --particles $(Part) --pileup $(PU) --job_id $(JobID) --n_jobs $(NJobs) --n_files $(NFiles)",
        "",
        "transfer_input_files = scripts/, data_handling/, configs/",
        "",
        "request_memory = 8G",
        "request_cpus = 1",
        "",
        "T3Queue = short",
        "WNTag = el9",
        "include : /opt/exp_soft/cms/t3/t3queue |",
        "",
        f"ProxyPath = {PROXY_PATH}",
        "",
        "output = logs/load_$(Part)_$(PU)_$(JobID).out",
        "error  = logs/load_$(Part)_$(PU)_$(JobID).err",
        "log    = logs/load_$(Part)_$(PU)_$(JobID).log",
        "",
        "queue Part, PU, JobID, NJobs, NFiles from (",
    ]
    missing = []
    for part, pu, job_id, *_ in failed:
        key = (part, pu, job_id)
        if key in job_params:
            n_jobs, n_files = job_params[key]
            lines.append(f"  {part}, {pu}, {job_id}, {n_jobs}, {n_files}")
        else:
            missing.append(key)
            print(f"  WARNING: no entry for {key} in {SUBMIT_FILE.name} — skipping")
    lines.append(")")

    out_path = Path(__file__).parent / "resubmit_failed.sub"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path, missing


def main():
    parser = argparse.ArgumentParser(description="Check .err logs and optionally resubmit failed jobs.")
    parser.add_argument("--submit", action="store_true", help="Submit the failed jobs after reporting.")
    args = parser.parse_args()

    job_params = load_job_params()
    if not job_params:
        print(f"WARNING: could not parse {SUBMIT_FILE.name} — NJobs/NFiles will be missing from resubmit file")

    failed, ok = check_err_files()

    print(f"\n{'='*60}")
    print(f"  OK jobs    : {len(ok)}")
    print(f"  Failed jobs: {len(failed)}")
    print(f"{'='*60}\n")

    if not failed:
        print("All jobs completed successfully.")
        return

    print("Failed jobs:")
    for part, pu, job_id, fname, first_error in failed:
        n_jobs, n_files = job_params.get((part, pu, job_id), ("?", "?"))
        print(f"  [{part} {pu} job={job_id:>2} n_jobs={n_jobs} n_files={n_files}]  {fname}")
        print(f"      -> {first_error[:120]}")

    sub_path, missing = make_resubmit_file(failed, job_params)
    print(f"\nResubmit file written to: {sub_path}")

    if missing:
        print(f"\nWARNING: {len(missing)} job(s) were skipped because they have no entry in {SUBMIT_FILE.name}.")

    if args.submit:
        print("\nSubmitting failed jobs...")
        result = subprocess.run(["condor_submit", str(sub_path)], capture_output=True, text=True,
                                cwd=str(sub_path.parent))
        print(result.stdout)
        if result.returncode != 0:
            print("ERROR during submission:", result.stderr, file=sys.stderr)
            sys.exit(result.returncode)
    else:
        print(f"Run with --submit to resubmit, or manually:  condor_submit {sub_path.name}")


if __name__ == "__main__":
    main()

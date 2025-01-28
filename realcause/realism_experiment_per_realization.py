#!/usr/bin/env python3

import os
import argparse
import subprocess
import ast
import numpy as np
import json


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments and aggregate results."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="List of integer seeds (e.g., --seeds 1 42 100).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where all experiments will be stored.",
    )
    args = parser.parse_args()

    # Fields of interest in the summary.txt of each experiment
    fields = ["ate_bias_noisy", "ate_bias_not_noisy", "pehe_noisy", "pehe_not_noisy"]

    # Create the main output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Dictionary to hold data for each realization across seeds
    realization_data = {i: {field: [] for field in fields} for i in range(1, 101)}

    for seed in args.seeds:
        print(f"Running experiments for seed: {seed}")
        seed_dir = os.path.join(args.out_dir, str(seed))
        os.makedirs(seed_dir, exist_ok=True)

        # Run 100 times for the current seed (this is the number of realizations in IHDP)
        for run_index in range(1, 101):
            print(f"  Run index: {run_index}")
            run_dir = os.path.join(seed_dir, str(run_index))
            os.makedirs(run_dir, exist_ok=True)

            # Command to call train_generator.py
            cmd = [
                "python",
                "train_generator.py",
                "--data",
                "ihdp",
                "--saveroot",
                run_dir,
                "--dist",
                "FactorialGaussian",
                "--model_type",
                "tarnet",
                "--n_hidden_layers",
                "1",
                "--dim_h",
                "64",
                "--batch_size",
                "64",
                "--num_epochs",
                "100",
                "--lr",
                "0.001",
                "--w_transform",
                "Standardize",
                "--y_transform",
                "Normalize",
                "--early_stop",
                "False",
                "--patience",
                "10",
                "--train",
                "True",
                "--eval",
                "True",
                "--realization_idx",
                str(run_index),
                "--seed",
                str(seed),
            ]

            # Run the command, if one of them fails then this raises a CalledProcessError
            subprocess.run(cmd, check=True)

            # Read the summary file
            summary_file = os.path.join(run_dir, "summary.txt")
            with open(summary_file, "r") as f:
                summary_str = f.read()
            summary_dict = ast.literal_eval(summary_str)
            for field in fields:
                realization_data[run_index][field].append(summary_dict[field])

    # Compute per-realization statistics
    realization_stats = {}
    global_data = {field: [] for field in fields}
    for realization, data in realization_data.items():
        realization_stats[realization] = {}
        for field in fields:
            arr = np.array(data[field])
            realization_mean = arr.mean()
            realization_std = arr.std()
            realization_stats[realization][f"{field}_mean"] = float(realization_mean)
            realization_stats[realization][f"{field}_std"] = float(realization_std)

            # Add realization-level data to global list for global stats
            global_data[field].extend(arr.tolist())

    # Compute global statistics across all realizations and seeds
    global_stats = {}
    for field in fields:
        global_arr = np.array(global_data[field])
        global_stats[f"{field}_mean"] = float(global_arr.mean())
        global_stats[f"{field}_std"] = float(global_arr.std())

    # Save everything into <out_dir>/summary.txt
    results = {
        "realization_stats": realization_stats,
        "global_stats": global_stats,
    }
    summary_out_file = os.path.join(args.out_dir, "summary.txt")
    with open(summary_out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nDone! Summary stored in {summary_out_file}")


if __name__ == "__main__":
    main()

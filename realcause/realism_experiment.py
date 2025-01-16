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

    # Dictionary to hold stats for each seed
    seed_stats = {}
    # Dictionary to collect data across all seeds
    all_data = {field: [] for field in fields}

    for seed in args.seeds:
        print(f"Running experiments for seed: {seed}")
        seed_dir = os.path.join(args.out_dir, str(seed))
        os.makedirs(seed_dir, exist_ok=True)

        # Collect data for each field
        seed_data = {field: [] for field in fields}

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
                seed_data[field].append(summary_dict[field])

        # Compute per-seed statistics
        seed_stats[seed] = {}
        for field in fields:
            arr = np.array(seed_data[field])
            seed_mean = arr.mean()
            seed_std = arr.std()
            seed_stats[seed][f"{field}_mean"] = float(seed_mean)
            seed_stats[seed][f"{field}_std"] = float(seed_std)

            # Add to global lists for overall stats
            all_data[field].extend(arr.tolist())

    # Compute overall statistics across *all* seeds
    overall_stats = {}
    for field in fields:
        field_arr = np.array(all_data[field])
        overall_stats[f"{field}_mean"] = float(field_arr.mean())
        overall_stats[f"{field}_std"] = float(field_arr.std())

    # Save everything into <out_dir>/summary.txt
    results = {"seed_stats": seed_stats, "overall_stats": overall_stats}
    summary_out_file = os.path.join(args.out_dir, "summary.txt")
    with open(summary_out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nDone! Summary stored in {summary_out_file}")


if __name__ == "__main__":
    main()

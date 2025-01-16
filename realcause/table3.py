import json
import numpy as np
from train_generator import get_data, main as tg_main
from loading import load_from_folder, load_gen
from pathlib import Path
import torch
from train_generator import get_args, main
import os
from run_metrics import evaluate_directory
import pprint
from data.ihdp import load_ihdp

MODEL_DIRECTORY = "./eval/ihdp/exp1/model"


def main():
    model, args = load_gen(saveroot=MODEL_DIRECTORY, dataroot="./datasets/")

    pehe_list = []
    true_ate_list = []
    estimated_ate_list = []
    ate_absolute_bias_list = []

    # Get an average over all realizations
    for i in range(101):
        # Get data
        # ites_true, ate_true, w, t, y = get_data(args, i=i) # no i=i argument
        d = load_ihdp(return_ate=True, return_ites=True, i=i)
        w, t, y, ate_true, ites_true = d["w"], d["t"], d["y"], d["ate"], d["ites"]

        # Compute estimated ITEs using the pretrained model
        estimated_ites = model.ite(w=w, noisy=True)

        # Calculate PEHE
        pehe = np.sqrt(np.mean((ites_true - estimated_ites) ** 2))
        # pehe = np.sqrt(np.median(np.square(ites_true - estimated_ites)))

        # Estimate the ATE using the model's method
        ate_est = model.noisy_ate(
            # w=w, # ATE is always the same
            # transform_w=True,
        )

        # Compute the ATE absolute bias
        ate_abs_bias = abs(ate_est - ate_true)

        pehe_list.append(pehe)
        true_ate_list.append(ate_true)
        estimated_ate_list.append(ate_est)
        ate_absolute_bias_list.append(ate_abs_bias)

    pehe = (np.mean(pehe_list), np.std(pehe_list))
    ate_true = (np.mean(true_ate_list), np.std(true_ate_list))
    ate_est = (np.mean(estimated_ate_list), np.std(estimated_ate_list))
    ate_abs_bias = (np.mean(ate_absolute_bias_list), np.std(ate_absolute_bias_list))

    def format_tuple(x):
        return f"({x[0]:.4f}, {x[1]:.4f})"

    print("\n")
    print("----------")
    print(f"PEHE: {format_tuple(pehe)}")
    print(f"True ATE: {format_tuple(ate_true)}")
    print(f"Estimated ATE: {format_tuple(ate_est)}")
    print(f"ATE Absolute Bias: {format_tuple(ate_abs_bias)}")
    print("----------")

    # Load the summary file and read metrics
    summary_path = Path(MODEL_DIRECTORY) / "summary.txt"
    with open(summary_path, "r") as file:
        summary_data = json.load(file)
    ate_exact = summary_data["ate_exact"]
    ate_noisy = summary_data["ate_noisy"]
    ate_abs_bias_summary = abs(ate_noisy - ate_exact)
    print("----------")
    print(f"ATE Exact (from summary file): {ate_exact:.4f}")
    print(f"ATE Noisy (from summary file): {ate_noisy:.4f}")
    print(f"ATE Absolute Bias: {ate_abs_bias_summary:.4f}")
    print("----------")


if __name__ == "__main__":
    evaluate_directory(
        checkpoint_dir="./eval",
        num_tests=100,  # Increase for robust metrics
        n_uni=100,  # Number of univariate samples
        n_multi=1000,  # Number of multivariate samples
        results_dir="./results",
    )
    results_path = Path("./results") / "results.json"
    with open(results_path, "r") as file:
        results_data = json.load(file)
    print()
    main()
    print()
    pprint.pprint(results_data)

import os
import json
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

base_dir = "my_experiments_3_per_realization"
n_seeds = 20
n_realizations = 100

data = []
n_significant = 0

for realization in range(1, 1 + n_realizations):
    all_biases = []
    filtered_biases = []

    for seed in range(1, 1 + n_seeds):
        summary_path = os.path.join(
            base_dir, str(seed), str(realization), "summary.txt"
        )
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            bias = summary["ate_bias_noisy"]
            all_biases.append(bias)

            if (
                summary["q30_y_pval"] > 0.05
                and summary["q30_t_pval"] > 0.05
                # and summary["avg_y_pval"] > 0.05
                # and summary["avg_t_pval"] > 0.05
            ):
                filtered_biases.append(bias)
        except Exception as e:
            print(f"Error reading {summary_path}: {e}")

    def compute_stats(biases):
        if len(biases) == 0:
            return np.nan, np.nan, np.nan
        return np.mean(biases), np.std(biases), np.max(biases)

    all_mean, all_std, all_max = compute_stats(all_biases)
    filt_mean, filt_std, filt_max = compute_stats(filtered_biases)

    # Welch's t-test (unequal variances)
    ttest_pval = np.nan
    if len(filtered_biases) >= 2:
        try:
            _, ttest_pval = ttest_ind(all_biases, filtered_biases, equal_var=False)
            if ttest_pval < 0.05:
                n_significant += 1
        except Exception as e:
            print(f"T-test failed for realization {realization}: {e}")

    data.append(
        {
            "realization": realization,
            "all_mean": all_mean,
            "all_std": all_std,
            "all_max": all_max,
            "filt_mean": filt_mean,
            "filt_std": filt_std,
            "filt_max": filt_max,
            "n_filtered": len(filtered_biases),
            "ttest_pval": ttest_pval,
        }
    )

df = pd.DataFrame(data)

# Show table
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 0)  # Auto-detect width
pd.set_option("display.max_colwidth", None)  # Show full content of each column

print(df)

# Final summary
print(
    f"\n📊 Out of {n_realizations} realizations, {n_significant} showed a statistically significant difference in ATE bias (Welch's t-test, p < 0.05) between all seeds and the filtered subset (where avg_y_pval > 0.05 and avg_t_pval > 0.05).\n"
)
print(
    "This means that in these cases, the ATE bias under 'statistically insignificant' p-values differs meaningfully from the full population--suggesting a possible relation between p-values and bias."
)

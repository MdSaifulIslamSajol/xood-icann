#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 00:51:19 2025

@author: saiful
"""

import pandas as pd
import numpy as np
import glob

# Assuming your result files are named like: results/svhn_densenet_imagenet_advers_seed_*/summary_x-ood-lr.csv
result_files = glob.glob("results/aggregate_svhn_densenet/svhn_densenet_seed_*/summary_x-ood-lr.csv")

all_seeds = []
for file in result_files:
    each_seed_df = pd.read_csv(file, index_col=0)
    all_seeds.append(each_seed_df)

# Stack all runs into 3D array
stacked = np.stack([each_seed_df.to_numpy() for each_seed_df in all_seeds], axis=0)

# Compute mean, std, and 95% CI
mean = np.mean(stacked, axis=0)
std = np.std(stacked, axis=0)
ci95 = 1.96 * std / np.sqrt(len(all_seeds))

# Use first index to get row/column names
df_mean = pd.DataFrame(mean, index=all_seeds[0].index, columns=all_seeds[0].columns)
df_std = pd.DataFrame(std, index=all_seeds[0].index, columns=all_seeds[0].columns)
df_ci = pd.DataFrame(ci95, index=all_seeds[0].index, columns=all_seeds[0].columns)


# # Save output
df_mean.to_csv("results/aggregate_svhn_densenet/mean_std_ci/mean.csv")
df_std.to_csv("results/aggregate_svhn_densenet/mean_std_ci/std.csv")
df_ci.to_csv("results/aggregate_svhn_densenet/mean_std_ci/ci95.csv")

# Now create formatted table: "mean ± std"
df_formatted = df_mean.copy()

for row in df_mean.index:
    for col in df_mean.columns:
        mean_val = df_mean.loc[row, col]
        std_val = df_std.loc[row, col]
        # Format with 3 decimal places
        df_formatted.loc[row, col] = f"{mean_val:.4f} ± {std_val:.4f}"

# Save final formatted CSV
df_formatted.to_csv("results/aggregate_svhn_densenet/mean_std_ci/final_mean_std.csv")

print("Formatted aggregation file saved successfully.")

print("Aggregation complete.")

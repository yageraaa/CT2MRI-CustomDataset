import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import wandb


def log_raw_data(CT_data, MR_data, idx, output_dir="debug"):
    os.makedirs(output_dir, exist_ok=True)
    slice_idx = CT_data.shape[2] // 2

    plt.imsave(os.path.join(output_dir, f"raw_CT_{idx}.png"), CT_data[:, :, slice_idx], cmap='gray')
    plt.imsave(os.path.join(output_dir, f"raw_MR_{idx}.png"), MR_data[:, :, slice_idx], cmap='gray')

def log_transposed_data(CT_data, MR_data, idx, output_dir="debug"):
    os.makedirs(output_dir, exist_ok=True)
    slice_idx = CT_data.shape[2] // 2

    plt.imsave(os.path.join(output_dir, f"transposed_CT_{idx}.png"), CT_data[:, :, slice_idx], cmap='gray')
    plt.imsave(os.path.join(output_dir, f"transposed_MR_{idx}.png"), MR_data[:, :, slice_idx], cmap='gray')

def log_filtered_data(CT_data, MR_data, idx, output_dir="debug"):
    os.makedirs(output_dir, exist_ok=True)

    plt.imsave(os.path.join(output_dir, f"filtered_CT_{idx}.png"), CT_data[:, :, 0], cmap='gray')
    plt.imsave(os.path.join(output_dir, f"filtered_MR_{idx}.png"), MR_data[:, :, 0], cmap='gray')

def log_histograms(normalized_histograms, cum_hist, hist_diff, sub_name, output_dir="./debug"):
    print(f"Creating directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    normalized_path = os.path.join(output_dir, f"{sub_name}_normalized_hist.png")
    print(f"Saving normalized histogram to: {normalized_path}")
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(normalized_histograms)), normalized_histograms, width=3)
    plt.title(f"Normalized Histogram: {sub_name}")
    plt.savefig(normalized_path)
    plt.close()

    cum_path = os.path.join(output_dir, f"{sub_name}_cumulative_hist.png")
    print(f"Saving cumulative histogram to: {cum_path}")
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(cum_hist)), cum_hist, width=3)
    plt.title(f"Cumulative Histogram: {sub_name}")
    plt.savefig(cum_path)
    plt.close()

    diff_path = os.path.join(output_dir, f"{sub_name}_hist_diff.png")
    print(f"Saving histogram diff to: {diff_path}")
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(hist_diff)), hist_diff, width=3)
    plt.title(f"Histogram Diff: {sub_name}")
    plt.savefig(diff_path)
    plt.close()

def log_data_loader_samples(dataset, num_samples=3, output_dir="debug"):
    os.makedirs(output_dir, exist_ok=True)
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        (x, x_name), (x_cond, x_cond_name), *context = sample

        plt.imsave(os.path.join(output_dir, f"sample_{i}_x.png"), x.numpy()[0], cmap='gray')
        plt.imsave(os.path.join(output_dir, f"sample_{i}_x_cond.png"), x_cond.numpy()[0], cmap='gray')

        try:
            wandb.log({
                f"sample_{i}/x": wandb.Image(x.numpy()[0], caption=f"x: {x_name}"),
                f"sample_{i}/x_cond": wandb.Image(x_cond.numpy()[0], caption=f"x_cond: {x_cond_name}"),
            })
        except:
            print(f"Could not log sample {i} to wandb")
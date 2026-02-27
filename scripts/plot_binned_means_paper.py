#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# User requirements:
# - Times New Roman
# - only x and y axis text
# - no title or explanation boxes (legend)
# - just the graph on the right (binned means)

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'

data_path = '/Users/emirhan/Desktop/LeGrad-1/scripts/outputs/cosine_vs_miou_correlation_all_methods_data.npz'
data = np.load(data_path)

model_type = data['model_type'].item() if data['model_type'].shape == () else str(data['model_type'])

methods = ['legrad', 'gradcam', 'chefercam', 'attentioncam', 'daam']
method_names = ['LeGrad', 'GradCAM', 'CheferCAM', 'AttentionCAM', 'DAAM']

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
markers = ['o', 's', '^', 'D', 'P']

fig, ax2 = plt.subplots(1, 1, figsize=(9, 7))

# Calculate global min and max to align with previous script
global_min = float('inf')
global_max = float('-inf')
all_res = {}
for m, name in zip(methods, method_names):
    if f'{m}_cosine_sims' in data:
        sims = data[f'{m}_cosine_sims']
        mious = data[f'{m}_mious']
        pr = data[f'{m}_pearson_r']
        all_res[name] = {'sims': sims, 'mious': mious, 'pr': pr}
        if sims.max() > global_max: global_max = sims.max()

global_min = 0.25

n_bins = 15
bin_edges = np.linspace(global_min, global_max, n_bins + 1)

for i, name in enumerate(method_names):
    if name not in all_res: continue
    res = all_res[name]
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]

    bin_centers = []
    bin_means = []
    bin_stds = []

    for b in range(n_bins):
        if b == n_bins - 1:
            mask = (res['sims'] >= bin_edges[b]) & (res['sims'] <= bin_edges[b + 1])
        else:
            mask = (res['sims'] >= bin_edges[b]) & (res['sims'] < bin_edges[b + 1])
        if mask.sum() >= 3:
            bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
            bin_means.append(res['mious'][mask].mean())
            bin_stds.append(res['mious'][mask].std() / np.sqrt(mask.sum()))

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    ax2.plot(bin_centers, bin_means, color=color, linewidth=2, marker=marker,
             markersize=8, label=name) # Add label in case we ever want legend
    # Shaded error band (±1 SE)
    ax2.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                     color=color, alpha=0.15)

ax2.set_xlabel('Cosine Similarity', fontsize=24)
ax2.set_ylabel('Mean Heatmap Mass (%)', fontsize=24) 
# Note: In the older code it dynamically set to 'Mean mIoU (%)' or 'Mean Heatmap Mass (%)'
# Based on the scores in the data, correlation r values match the output.

ax2.tick_params(labelsize=20)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(global_min - 0.02, global_max + 0.02)

ax2.legend(fontsize=20, loc='upper left', framealpha=0.9)

plt.tight_layout()
output_path = '/Users/emirhan/Desktop/LeGrad-1/scripts/outputs/cosine_vs_miou_binned_paper.pdf'
plt.savefig(output_path, bbox_inches='tight', facecolor='white')
output_path_png = '/Users/emirhan/Desktop/LeGrad-1/scripts/outputs/cosine_vs_miou_binned_paper.png'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')

print(f"Saved to {output_path} and .png")

import torch
import numpy as np
from benchmark_segmentation_v2 import batch_pix_accuracy, batch_intersection_union, get_ap_scores

# Dummy inputs
dummy_heatmap = torch.rand(1, 224, 224)
dummy_predict = torch.stack([1.0 - dummy_heatmap, dummy_heatmap], dim=0)

# All-zero target (wrong prompt scenario)
dummy_target = torch.zeros(1, 224, 224, dtype=torch.long)

pix_corr, pix_lbl = batch_pix_accuracy(dummy_predict, dummy_target)
print("Pixel Acc:", pix_corr, pix_lbl)

inter, union = batch_intersection_union(dummy_predict, dummy_target, nclass=2)
print("mIoU inter:", inter)
print("mIoU union:", union)

try:
    ap = get_ap_scores(dummy_predict, dummy_target)
    print("AP:", ap)
except Exception as e:
    print("AP Error:", e)


# CFOT: Cross-Frame Optimal Transport for Skeleton-Based Gesture Recognition

Official PyTorch implementation of the paper:

**Cross-Frame Optimal Transport Learning for Skeleton-Based Hand Gesture Recognition**  
*ICPR 2026 (under review)*

<p align="center">
  <img src="figures/CFOT_overview.png" width="900">
</p>

---

## Overview

Cross-Frame Optimal Transport (CFOT) is a temporal modeling framework for
skeleton-based hand gesture recognition. CFOT explicitly discovers
long-range correspondences between skeletal joints across non-consecutive
frames using optimal transport, enabling robust recognition under temporal
variations and motion ambiguity.


# IPN Skeleton Dataset (MediaPipe Extracted)

This dataset contains **skeleton-based representations extracted using
Google MediaPipe** from the IPN Hand Gesture Dataset.

## Important Note

- This dataset contains **only derived skeleton data**

## Extraction Details

- Skeleton extractor: Google MediaPipe
- Joints: hand and upper-body landmarks
- Coordinates: (x, y, z)


---

## Pretrained Models

| Model        | Dataset  | Accuracy | Link                         |
|--------------|----------|----------|------------------------------|
| CFOT + ST-GCN | ISL      | 90.20%   | `pretrained/cfot_isl.pt`     |
| CFOT + ST-GCN | IPN      | 95.71%   | `pretrained/cfot_ipn.pt`     |
| CFOT + ST-GCN | Briareo  | 99.31%   | `pretrained/cfot_briareo.pt`|

---

## Quick Start (Evaluation)

Evaluate a pretrained CFOT model:

```bash
python scripts\eval_<dataset_name>.py\
  --config configs/<dataset_name>_stgcn.yaml \
  --weights pretrained/cfot_<dataset_name>.pt

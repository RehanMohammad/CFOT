# CFOT: Cross-Frame Optimal Transport for Skeleton-Based Gesture Recognition

Official PyTorch implementation of the paper:

**Cross-Frame Optimal Transport Learning for Skeleton-Based Hand Gesture Recognition**  
*ICPR 2026 (under review)*


<p align="center">
  <img src="figures/pipeline.pdf" width="900">
</p>


## Overview

Cross-Frame Optimal Transport (CFOT) is a temporal modeling framework for
skeleton-based hand gesture recognition. CFOT explicitly discovers
long-range correspondences between skeletal joints across non-consecutive
frames using optimal transport, enabling robust recognition under temporal
variations and motion ambiguity.



## Pretrained Models

| Model | Dataset | Accuracy | Link |
|------|--------|----------|------|
| CFOT + ST-GCN | ISL     | 90.20% | `pretrained/cfot_isl.pt` |
| CFOT + ST-GCN | IPN 	  | 95.71% | `pretrained/cfot_ipn.pt` |
| CFOT + ST-GCN | Briareo | 99.31% | `pretrained/cfot_ipn.pt` |


## Quick Start (Evaluation)

Evaluate a pretrained CFOT model:

```bash
python eval_<dataset_name:isl||ipn||briareo>.py \
  --config configs/<dataset_name:isl||ipn||briareo>_stgcn.yaml \
  --weights pretrained/cfot_<dataset_name:isl||ipn||briareo>.pt



\## Installation

```bash

pip install -r requirements.txt



```markdown
## Dataset

### ISL Skeleton Dataset

- 50 isolated ISL gestures
- 3 subjects
- 10 repetitions per gesture
- Skeletons extracted using MediaPipe
- Wrist-centered normalization

📁 Dataset: `ISL_dataset/ISL_dataset.zip`

⚠️ **License**: Research-only, non-commercial  
See `ISL_dataset/LICENSE.txt`

```markdown
## License

- Code: MIT License
- Dataset: Research-only, non-commercial license
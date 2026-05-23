# ZeroAD Zero-Shot: CLIP-Based Anomaly Detection

Official submission for VAND 4.0 Challenge - Industrial Track (Zero-Shot Category).

## Method Overview

ZeroAD is a parameter-efficient zero-shot anomaly detection framework that leverages frozen CLIP ViT-L/14@336px features. The approach:

1. **Frozen Feature Extraction**: Extract multi-scale features from 5 intermediate layers (12, 15, 18, 21, 24) of CLIP ViT-L/14@336px
2. **Lightweight Scorers**: Train only 10 small linear layers (768→1 each) for anomaly scoring:
   - 5 image-level classifiers (one per layer)
   - 5 pixel-level detectors (one per layer)
3. **Multi-Layer Fusion**: Average anomaly scores across all 5 feature layers
4. **Dual-Branch Inference**: Combine image-level classification and pixel-level segmentation scores

**Total Trainable Parameters**: 7,690 (0.0077M)

## Zero-Shot Declaration

✓ **No MVTec AD 2 images used during training**
- Pretraining: CLIP ViT-L/14@336px (OpenAI, trained on LAION-400M / OpenAI WebImageText)
- Training: MVTec AD (test split) or VisA (test split) only
- Hyperparameters: Fixed (no tuning on MVTec AD 2)

## Environment Setup

### Requirements
- Python 3.9+
- CUDA 11.8+ (or CPU mode)
- GPU: Recommended (RTX 3090 / A100 or similar for inference)

### Installation

```bash
git clone https://github.com/clear-able/ZeroAD-VAND2026.git
cd zero-shot
pip install -r requirements.txt
```

The CLIP backbone model (~890 MB) will be automatically downloaded to `~/.cache/clip/` on first run.

## Quick Start

### Test on MVTec2 (Compute SegF1)

```bash
# Using MVTec-trained weights (recommended)
python test.py \
  --weight weights/uniadet_mvtec_final.pth \
  --data_dir /path/to/mvtec2_dataset \
  --image_size 518

# Output: Per-category and mean SegF1 scores
```

### Generate Submission Files

```bash
# Run inference on test_private and test_private_mixed, save float16 .tiff anomaly maps
python submit.py \
  --weight weights/uniadet_mvtec_final.pth \
  --data_dir /path/to/mvtec2_dataset \
  --output_dir ./submission

# Output structure: submission/anomaly_images/{category}/{split}/*.tiff
# Compress and upload: tar -czf submission.tar.gz -C submission anomaly_images/
```

### Train from Scratch (Optional)

```bash
# Train on MVTec AD (test split)
python train.py \
  --dataset mvtec \
  --test_dataset mvtec \
  --data_dir /path/to/datasets \
  --epoch 15 \
  --learning_rate 0.001 \
  --batch_size 8

# Train on VisA (test split)
python train.py \
  --dataset visa \
  --test_dataset mvtec \
  --data_dir /path/to/datasets \
  --epoch 15 \
  --learning_rate 0.001 \
  --batch_size 8
```

## Results

### Evaluated on MVTec2 test_public (SegF1)

| Training Data | Mean SegF1 | Best Category | Worst Category |
|---|---|---|---|
| MVTec AD (test) | **22.9%** | rice (43.8%) | can (0.4%) |
| VisA (test) | 4.6% | vial (13.9%) | can (0.1%) |

### Model Performance Details

**Image-Level Anomaly Detection** (using MVTec-trained weights):
- I-AUROC: 89.5% (MVTec2 test_public)
- I-AP: 94.3%

**Pixel-Level Anomaly Localization** (SegF1 at optimal threshold):
- SegF1: 22.9% (Mean across 8 categories)
- Robust pixel-level detection with Gaussian smoothing post-processing

## Architecture Details

### Feature Extraction
- **Backbone**: CLIP ViT-L/14@336px (frozen)
- **Layers Used**: [12, 15, 18, 21, 24]
- **Feature Dimension**: 768 (after projection)
- **Patch Grid**: 37×37 = 1,369 patches per image

### Scoring Modules
Each of 5 feature layers has:
```
AnomalyScorer_CLS:  Linear(768 → 1)  # Image-level binary classifier
AnomalyScorer_SEG:  Linear(768 → 1)  # Per-patch anomaly scorer
```

### Inference Pipeline
1. Extract features from 5 layers
2. L2-normalize CLS and patch features
3. Classify each layer independently
4. Generate pixel-level anomaly maps (upsampled to 518×518)
5. Average scores across 5 layers
6. Fuse image-level and pixel-level scores: `(1-0.5)×cls + 0.5×max(seg_map)`
7. Apply Gaussian smoothing (σ=4) for robustness

## File Structure

```
zero-shot/
├── README.md                      # This file
├── LICENSE                        # CC BY-NC 4.0
├── requirements.txt               # Python dependencies
├── model.py                       # ZeroAD model definition
├── train.py                       # Training & evaluation entrypoint
├── test.py                        # Evaluation on MVTec2 test_public
├── submit.py                      # Submission file generation
├── clip/                          # CLIP visual encoder
│   ├── __init__.py
│   ├── clip.py                    # CLIP loader (auto-downloads ViT-L/14@336px)
│   └── model.py                   # CLIP architecture with intermediate layer extraction
└── weights/
    ├── README.md                  # Weight descriptions
    ├── uniadet_mvtec_final.pth    # MVTec-trained weights (38 KB)
    └── uniadet_visa_final.pth     # VisA-trained weights (38 KB)
```

## Configuration

### Adjustable Parameters

#### test.py
- `--weight`: Path to checkpoint file
- `--image_size`: Input image size (default: 518)
- `--sigma`: Gaussian smoothing kernel (default: 4)
- `--data_dir`: Dataset root directory

#### train.py
- `--dataset`: Training dataset (`mvtec` or `visa`)
- `--test_dataset`: Test dataset (`mvtec` or `visa`)
- `--epoch`: Training epochs (default: 15)
- `--learning_rate`: Adam learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 8)
- `--save_freq`: Save frequency (default: every 5 epochs)

## Dataset Compatibility

- **Tested**: MVTec AD 2, MVTec AD, VisA
- **Required Format**: Standard anomaly detection directory structure
  ```
  {dataset_name}/
  ├── {category}/
  │   ├── train/good/
  │   ├── test/good/ & test/{defect_type}/
  │   └── ground_truth/{defect_type}/
  ```

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- torch ≥ 2.0.0
- torchvision ≥ 0.15.0
- scipy, scikit-image, scikit-learn
- PIL, tqdm, tifffile, tabulate

## References

- CLIP: https://github.com/openai/CLIP
- AnomalyCLIP: https://arxiv.org/abs/2310.18961
- MVTec AD 2 Challenge: https://sites.google.com/view/vand4-cvpr2026/challenge

## Citation

If you use this method in your research, please cite:

```bibtex
@inproceedings{vand2026_zeroshot,
  title={ZeroAD: Zero-Shot Anomaly Detection via Lightweight CLIP Feature Scoring},
  author={Chen, Long},
  booktitle={CVPR 2026 VAND Challenge},
  year={2026}
}
```

## License

This code is released under the **CC BY-NC 4.0 license**. See `LICENSE` for details.

The CLIP model is released under the MIT license by OpenAI.

## Contact

Long Chen  
Guangdong Artificial Intelligence and Digital Economy Laboratory (Shenzhen)  
Shenzhen University, Shenzhen, China  
chenlong@gml.ac.cn

# Model Weights

This directory contains the anomaly detection model weights trained for the zero-shot track.

## Pretrained Weights

Two versions of trained weights are provided:

### uniadet_mvtec_final.pth (38 KB)
Model trained on MVTec AD (test split only) for 15 epochs.
- Recommended for evaluating on datasets with industrial defects similar to MVTec
- SegF1 on MVTec2 test_public: 22.9% (Mean)

### uniadet_visa_final.pth (38 KB)
Model trained on VisA (test split only) for 15 epochs.
- Recommended as a general baseline
- SegF1 on MVTec2 test_public: 4.6% (Mean)

## CLIP Backbone

The underlying visual encoder is **CLIP ViT-L/14@336px**, which is automatically downloaded from OpenAI's CDN on first run to `~/.cache/clip/ViT-L-14-336px.pt` (~890 MB). This is a frozen pretrained model and is not included in the submission.

## How to Use

```bash
# Test with MVTec-trained weights
python test.py --weight weights/uniadet_mvtec_final.pth --data_dir /path/to/mvtec2 --output_dir ./results

# Generate submission with MVTec-trained weights
python submit.py --weight weights/uniadet_mvtec_final.pth --data_dir /path/to/mvtec2 --output_dir ./submission
```

## Zero-Shot Declaration

These models follow strict zero-shot protocol:
- **No images from MVTec AD 2 dataset were used during training**
- Only CLIP ViT-L/14@336px pretrained weights (from LAION-400M / OpenAI WebImageText) are used
- Model weights contain only 7,690 trainable parameters (10 × Linear(768→1) modules for 5 feature layers)

# MobileNet + Gabor Unmasked Pipeline

This pipeline is optimized for fast unmasked face recognition using MobileNetV2 and Gabor filters.

## Features
- **Backbone**: MobileNetV2
- **Texture Features**: Gabor Filters (16-D)
- **Dataset**: Unmasked Faces
- **Similarity**: Cosine Similarity (threshold: 0.55)
- **Speed**: Optimized by skipping mask detection and filtering

## Usage
```bash
python train_mobilenet_gabor_unmasked.py
```

# MobileNet + Gabor Combined Pipeline

This pipeline is designed to handle both masked and unmasked faces by training on a combined dataset.

## Features
- **Backbone**: MobileNetV2
- **Texture Features**: Gabor Filters (16-D)
- **Dataset**: Combined (Masked + Unmasked)
- **Similarity**: Cosine Similarity (threshold: 0.55)
- **Versatility**: Capable of identifying people regardless of whether they are wearing a mask.

## Usage
```bash
python train_mobilenet_gabor_both.py
```

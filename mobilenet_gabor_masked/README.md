# MobileNet + Gabor Masked Pipeline

This pipeline is optimized for masked face recognition using a MobileNetV2 backbone for embeddings and Gabor filters for texture feature extraction.

## Features
- **Backbone**: MobileNetV2
- **Texture Features**: Gabor Filters (16-D)
- **Dataset**: Masked Faces
- **Similarity**: Cosine Similarity (threshold: 0.55)
- **Preprocessing**: Background removal (optional), Gaussian filtering
- **Segmentation**: YOLO-based face and mask detection

## Usage
To train the pipeline:
```bash
python train_mobilenet_gabor_masked.py
```

## Structure
- `src_mobilenet_gabor_masked/`: Core modules
- `train_mobilenet_gabor_masked.py`: Training script

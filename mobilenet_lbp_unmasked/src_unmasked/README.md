# src_unmasked - Face Recognition Pipeline for Unmasked Dataset

This folder contains the face recognition pipeline adapted for the **unmasked dataset** (Dataset-Without mask) from the Proposed dataset folder.

## Key Differences from `src` folder

### 1. **No Filtering Applied**
- The `src` folder includes a filtering step (Gaussian/Median filtering) before feature extraction
- The `src_unmasked` folder **skips the filtering step entirely**
- Face images are processed directly after segmentation

### 2. **No Mask Detection**
- The `segmentation.py` module does not include mask detection logic
- The `segment_face()` method returns only face regions without mask information
- The pipeline does not check for `is_masked` or `mask_confidence` fields

### 3. **Simplified Pipeline**
The processing flow in `pipeline.py`:
1. **Preprocessing** - Background removal and resizing
2. **Segmentation** - Face detection only (no mask detection)
3. **LBP Extraction** - Texture features (no filtering applied beforehand)
4. **Embedding Extraction** - Deep learning features
5. **Classification** - Person identification

## Files Created

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization |
| `detector.py` | Person identification classifier (SVM/RF/KNN) |
| `embedding.py` | Deep learning embeddings using MobileNetV2 |
| `lbp_extractor.py` | Local Binary Pattern feature extraction |
| `preprocessing.py` | Background removal and image resizing |
| `segmentation.py` | Face detection (without mask detection) |
| `pipeline.py` | Complete pipeline (without filtering step) |

## Usage

```python
from src_unmasked.pipeline import FaceRecognitionPipeline

# Initialize pipeline
pipeline = FaceRecognitionPipeline(
    target_size=(256, 256),
    remove_bg=True,
    detector_type='yolo',
    embedding_dim=128,
    classifier_type='svm'
)

# Process single image
result = pipeline.process_image(image_path='path/to/image.jpg')

# Train on unmasked dataset
pipeline.train(train_dir='Proposed dataset/Dataset-Without mask')

# Save trained model
pipeline.save_pipeline('models/unmasked_model')
```

## Dataset Structure

Expected directory structure for training:
```
Proposed dataset/
└── Dataset-Without mask/
    ├── person1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── person2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
```

## Comparison with `src` folder

| Feature | `src` (Masked) | `src_unmasked` (Unmasked) |
|---------|----------------|---------------------------|
| Filtering | ✅ Gaussian/Median | ❌ No filtering |
| Mask Detection | ✅ Included | ❌ Not included |
| Face Detection | ✅ YOLO/MTCNN/MediaPipe | ✅ YOLO/MTCNN/MediaPipe |
| LBP Features | ✅ Included | ✅ Included |
| Deep Embeddings | ✅ MobileNetV2 | ✅ MobileNetV2 |
| Classification | ✅ SVM/RF/KNN | ✅ SVM/RF/KNN |

## Notes

- This pipeline is specifically designed for the unmasked dataset
- No filtering is applied to preserve original face features
- Mask detection logic has been completely removed
- All other components (embeddings, LBP, classification) remain the same

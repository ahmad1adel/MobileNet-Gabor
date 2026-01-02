# Face Recognition System with Mask Detection

A comprehensive face recognition system that can identify individuals whether they are wearing a mask or not. The system implements a complete pipeline from image preprocessing to person identification.

## Features

1. **Image Preprocessing**
   - Background removal using deep learning
   - Image resizing to uniform dimensions

2. **Face Segmentation**
   - Face detection using MTCNN or MediaPipe
   - Mask detection capability

3. **Image Filtering**
   - Gaussian Blur for noise reduction
   - Median Filter alternative

4. **LBP Feature Extraction**
   - Local Binary Pattern texture features
   - Multiscale and spatial LBP options

5. **Deep Learning Embeddings**
   - Face embeddings using MobileNetV2-based architecture
   - Fine-tunable embedding model

6. **Person Detection/Identification**
   - Cosine Similarity (threshold 0.55) ⭐
   - Unknown person detection
   - Hybrid features (embeddings + LBP)

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
frp/
├── data/
│   ├── raw/              # Raw input images
│   ├── processed/        # Preprocessed images
│   ├── train/            # Training data (person folders)
│   ├── val/              # Validation data
│   └── test/             # Test data
├── models/               # Saved models
├── outputs/              # Output results
├── src/
│   ├── preprocessing.py  # Image preprocessing
│   ├── segmentation.py   # Face detection & segmentation
│   ├── filtering.py      # Image filtering
│   ├── lbp_extractor.py  # LBP feature extraction
│   ├── embedding.py      # Deep learning embeddings
│   ├── detector.py       # Cosine similarity identification ⭐
│   └── pipeline.py       # End-to-end pipeline
├── train.py              # Training script
├── inference.py          # Inference script
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Usage

### Training

Prepare your training data in the following structure:
```
data/train/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── image1.jpg
│   └── ...
└── ...
```

Run training:
```bash
python train.py --train_dir data/train --val_dir data/val --model_dir models
```

Training options:
- `--train_dir`: Training data directory (required)
- `--val_dir`: Validation data directory (optional)
- `--model_dir`: Directory to save models (default: models)
- `--target_size`: Image size as width height (default: 256 256)
- `--remove_bg`: Remove background (default: True)
- `--filter_type`: gaussian or median (default: gaussian)
- `--detector_type`: yolo, mtcnn, or mediapipe (default: yolo)
- `--similarity_threshold`: threshold for identification (default: 0.55)
- `--embedding_dim`: Embedding dimension (default: 128)

### Inference

Run inference on a single image:
```bash
python inference.py --image path/to/image.jpg --model_dir models --output result.jpg
```

Inference options:
- `--image`: Input image path (required)
- `--model_dir`: Directory containing trained models (default: models)
- `--output`: Path to save visualization (optional)
- Other options same as training

### Using the Pipeline in Code

```python
from src.pipeline import FaceRecognitionPipeline

# Initialize pipeline
pipeline = FaceRecognitionPipeline(
    target_size=(256, 256),
    remove_bg=True,
    filter_type='gaussian',
    detector_type='yolo',
    similarity_threshold=0.55
)

# Load trained models
pipeline.load_pipeline('models')

# Process an image
result = pipeline.process_image(image_path='path/to/image.jpg')

# Access results
if result['success']:
    for face_data in result['faces']:
        print(f"Person: {face_data['prediction']}")
        print(f"Confidence: {face_data['confidence']}")
        print(f"Masked: {face_data['is_masked']}")
```

## Pipeline Steps

1. **Preprocessing**: Removes background and resizes images
2. **Segmentation**: Detects faces and determines mask status
3. **Filtering**: Applies noise reduction (Gaussian/Median)
4. **LBP Extraction**: Extracts texture features
5. **Embedding**: Generates deep learning features
6. **Identification**: Compares signatures using cosine similarity (threshold 0.55)

## Model Components

- **Face Embedder**: MobileNetV2-based architecture for generating face embeddings
- **LBP Extractor**: Local Binary Pattern features for texture analysis
- **Detection**: Cosine similarity for person identification using combined features (0.55 threshold)

## Notes

- The system works with both masked and unmasked faces
- Background removal requires the `rembg` library
- MTCNN requires TensorFlow
- For best results, ensure good lighting and clear face visibility in training images
- The mask detection uses a heuristic approach; for production, consider training a dedicated mask detection model

## Troubleshooting

- If background removal fails, ensure `rembg` is properly installed
- For MTCNN issues, try using MediaPipe detector instead
- If memory issues occur, reduce `target_size` or batch processing
- Ensure training data has multiple images per person for better accuracy

## License

This project is provided as-is for educational and research purposes.


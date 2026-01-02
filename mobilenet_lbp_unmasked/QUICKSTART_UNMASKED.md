# Quick Start Guide - Unmasked Dataset Pipeline

## 🚀 How to Run

### Option 1: Simple Training (Recommended)

Train the model on the unmasked dataset with default settings:

```bash
python train_unmasked.py
```

This will:
- Use the default dataset path: `Proposed dataset/Dataset-Without mask`
- Train using YOLO detector and Cosine Similarity identification (threshold 0.55)
- Save the model to `models/unmasked_model/`

### Option 2: Custom Training

Train with custom parameters:

```bash
python train_unmasked.py --train_dir "Proposed dataset/Dataset-Without mask" --model_dir "models/my_model" --detector_type mtcnn --classifier_type rf
```

**Available options:**
- `--train_dir`: Path to training data (default: `Proposed dataset/Dataset-Without mask`)
- `--val_dir`: Path to validation data (optional)
- `--model_dir`: Where to save models (default: `models/unmasked_model`)
- `--detector_type`: Face detector (`yolo`, `mtcnn`, or `mediapipe`)
- `--classifier_type`: Classifier (`svm`, `rf`, or `knn`)
- `--target_size`: Image size (default: `256 256`)
- `--embedding_dim`: Embedding dimension (default: `128`)

### Option 3: Run Examples

See various usage examples:

```bash
python example_usage_unmasked.py
```

### Option 4: Custom Script

Create your own script:

```python
from src_unmasked.pipeline import FaceRecognitionPipeline

# Initialize
pipeline = FaceRecognitionPipeline(
    target_size=(256, 256),
    remove_bg=True,
    detector_type='yolo',
    similarity_threshold=0.55
)

# Train
pipeline.train(train_dir='Proposed dataset/Dataset-Without mask')

# Save
pipeline.save_pipeline('models/unmasked_model')

# Use for prediction
result = pipeline.process_image(image_path='test.jpg')
if result['success']:
    for face in result['faces']:
        print(f"Person: {face['prediction']}")
        print(f"Confidence: {face['confidence']:.2%}")
```

## 📁 Dataset Structure Required

Your dataset should be organized like this:

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
    └── person3/
        └── ...
```

## 🔧 Prerequisites

Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow
- opencv-python
- numpy
- scikit-learn
- scikit-image
- mtcnn
- mediapipe
- ultralytics (for YOLO)
- rembg
- pillow

## 📊 After Training

Once training is complete, you'll find:
- `models/unmasked_model/embedder.keras` - Deep learning model
- `models/unmasked_model/detector.pkl` - Cosine similarity signatures

## 🎯 Using Trained Model

```python
from src_unmasked.pipeline import FaceRecognitionPipeline

# Load trained model
pipeline = FaceRecognitionPipeline()
pipeline.load_pipeline('models/unmasked_model')

# Predict on new image
result = pipeline.process_image(image_path='new_person.jpg')
print(result)
```

## ⚡ Quick Commands Cheat Sheet

| Task | Command |
|------|---------|
| Train with defaults | `python train_unmasked.py` |
| Train with MTCNN | `python train_unmasked.py --detector_type mtcnn` |
| Train with Random Forest | `python train_unmasked.py --classifier_type rf` |
| Custom output dir | `python train_unmasked.py --model_dir models/my_model` |
| See examples | `python example_usage_unmasked.py` |

## 🆚 Differences from Masked Pipeline

| Feature | Masked (`src`) | Unmasked (`src_unmasked`) |
|---------|----------------|---------------------------|
| Filtering | ✅ Applied | ❌ Not applied |
| Mask Detection | ✅ Included | ❌ Not included |
| Training Script | `train.py` | `train_unmasked.py` |
| Example Script | `example_usage.py` | `example_usage_unmasked.py` |

## 💡 Tips

1. **Start small**: Test with a subset of your data first
2. **YOLO is faster**: Use YOLO detector for speed, MTCNN for accuracy
3. **Similarity works well**: Cosine similarity (threshold 0.55) gives robust results
4. **Monitor training**: Watch the identification accuracy during training
5. **Save your models**: Always save after successful training

## 🐛 Troubleshooting

**Issue**: `No faces detected`
- Try different detector types (YOLO, MTCNN, MediaPipe)
- Check if images contain visible faces
- Verify image paths are correct

**Issue**: `Out of memory`
- Reduce batch size
- Use smaller target_size
- Process fewer images at once

**Issue**: `Import errors`
- Install missing packages: `pip install -r requirements.txt`
- Check Python version (3.8+ recommended)

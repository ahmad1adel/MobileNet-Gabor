# Face Recognition System with Mask Detection

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.0+-orange.svg)

A comprehensive, production-ready face recognition system with **mask detection capabilities**. This project implements a complete end-to-end pipeline supporting three variants: **unmasked faces**, **masked faces**, and **mixed scenarios**.

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Pipeline Variants](#pipeline-variants)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Configuration Options](#configuration-options)
- [Model Components](#model-components)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### Core Capabilities

✅ **Multi-Face Detection** - Detects multiple faces in a single image  
✅ **Mask Detection** - Identifies whether individuals are wearing masks  
✅ **Deep Learning Embeddings** - MobileNetV2-based feature extraction  
✅ **Texture Analysis** - Local Binary Pattern (LBP) feature extraction  
✅ **Robust Identification** - Cosine similarity matching with configurable thresholds  
✅ **Flexible Preprocessing** - Background removal, image resizing, filtering  
✅ **Batch Processing** - Efficient processing of multiple images  
✅ **Model Persistence** - Save and load trained models  

### Preprocessing Options

- **Background Removal** - Deep learning-based background segmentation
- **Image Filtering** - Gaussian Blur or Median Filtering for noise reduction
- **Image Resizing** - Uniform dimension normalization (default: 256×256)

### Face Detection Options

- **YOLO** (default) - Fast and accurate real-time detection
- **MTCNN** - High-quality face detection
- **MediaPipe** - Lightweight alternative

---

## 🏗️ System Architecture

```
INPUT IMAGE
    ↓
[PREPROCESSING] → Remove background, resize
    ↓
[FACE DETECTION] → Detect faces and mask status (YOLO/MTCNN/MediaPipe)
    ↓
[OPTIONAL FILTERING] → Gaussian or Median filtering
    ↓
[FEATURE EXTRACTION]
    ├─→ LBP Extractor (texture features)
    └─→ MobileNetV2 Embedder (deep features)
    ↓
[PERSON IDENTIFICATION] → Cosine Similarity (threshold: 0.55)
    ↓
OUTPUT → Person name + Confidence + Mask status
```

---

## 🎯 Pipeline Variants

### 1. **Unmasked Pipeline** (`mobilenet_lbp_unmasked/`)

Optimized for recognizing individuals **without face masks**.

**Key Characteristics:**
- ❌ No filtering applied (optimal for clear faces)
- ✅ Faster processing
- ✅ Better accuracy for unmasked faces
- 📊 Hybrid features: LBP + Deep embeddings

**Best For:** Secure access systems, identification in controlled environments

**Quick Start:**
```bash
cd mobilenet_lbp_unmasked
python train_unmasked_simple.py
```

---

### 2. **Masked Pipeline** (`mobilenet_lbp_masked/`)

Specialized for recognizing individuals **wearing face masks**.

**Key Characteristics:**
- ✅ **Gaussian filtering** enabled (handles mask artifacts)
- ✅ **Mask detection** built-in
- ✅ Fine-tuned for masked scenarios
- 🔧 Configurable: 20 epochs, batch size 16, LR 0.01

**Best For:** Medical facilities, public health surveillance, post-pandemic deployments

**Quick Start:**
```bash
cd mobilenet_lbp_masked
python train_masked_simple.py
```

---

### 3. **Both Scenarios Pipeline** (`mobilenet_lbp_both/`)

Unified solution for **mixed masked and unmasked** environments.

**Key Characteristics:**
- ✅ Handles both masked and unmasked faces
- ✅ Adaptive filtering (Gaussian or Median)
- ✅ Comprehensive feature extraction
- 🎯 Cosine similarity identification
- 📈 Production-ready performance

**Best For:** Public spaces, airports, real-world deployments with variable mask usage

**Quick Start:**
```bash
cd mobilenet_lbp_both
python train.py --train_dir data/train
```

---

## 📥 Installation

### Prerequisites

- **Python 3.8+**
- **CUDA 11.0+** (recommended for GPU support)
- **8GB+ RAM** (16GB recommended for fine-tuning)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- TensorFlow 2.10+
- OpenCV 4.5+
- NumPy 1.21+
- scikit-image
- scikit-learn
- PyYAML

### Step 3: Download Pre-trained Weights (Optional)

Some detectors may require pre-trained weights:
```bash
# YOLO weights (if using YOLO detector)
# Usually automatically downloaded on first use
```

---

## 🚀 Quick Start

### Using the Unmasked Pipeline

```bash
# Navigate to unmasked pipeline
cd mobilenet_lbp_unmasked

# Train on your unmasked dataset
python train_unmasked_simple.py

# Or with custom parameters
python train_unmasked.py \
    --train_dir "path/to/dataset" \
    --model_dir "models/my_model" \
    --detector_type yolo
```

### Using the Masked Pipeline

```bash
# Navigate to masked pipeline
cd mobilenet_lbp_masked

# Train on your masked dataset
python train_masked_simple.py

# With custom learning rate
python train_masked_simple.py --learning_rate 0.001
```

### Using the Both Pipeline

```bash
# Navigate to both pipeline
cd mobilenet_lbp_both

# Train on mixed dataset
python train.py \
    --train_dir data/train \
    --val_dir data/val \
    --filter_type gaussian \
    --detector_type yolo
```

---

## 📖 Usage Guide

### Dataset Structure

All pipelines expect the following directory structure:

```
your_dataset/
├── person_1/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── person_2/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── person_n/
    └── ...
```

**Requirements:**
- Minimum 3-5 images per person
- Supported formats: JPG, PNG, JPEG
- Recommended: 256×256 or larger

### Python API Usage

#### Basic Usage (Unmasked)

```python
from mobilenet_lbp_unmasked.src_unmasked.pipeline import FaceRecognitionPipeline

# Initialize
pipeline = FaceRecognitionPipeline(
    target_size=(256, 256),
    remove_bg=True,
    detector_type='yolo',
    similarity_threshold=0.55,
    embedding_dim=128
)

# Train
pipeline.train(train_dir='data/train', val_dir='data/val')

# Save
pipeline.save_pipeline('models/unmasked_model')

# Inference
result = pipeline.process_image(image_path='test.jpg')
if result['success']:
    for face in result['faces']:
        print(f"Person: {face['prediction']}")
        print(f"Confidence: {face['confidence']:.2%}")
```

#### Masked Pipeline Usage

```python
from mobilenet_lbp_masked.src_masked.pipeline import FaceRecognitionPipeline

# Initialize with filtering
pipeline = FaceRecognitionPipeline(
    target_size=(256, 256),
    remove_bg=False,              # Disable to save memory
    filter_type='gaussian',       # Enable filtering
    detector_type='yolo',
    similarity_threshold=0.55,
    embedding_dim=128
)

# Train with fine-tuning
pipeline.train(
    train_dir='data/train_masked',
    fine_tune_embedder=True,
    epochs=20,
    batch_size=16,
    learning_rate=0.01
)

# Save and use
pipeline.save_pipeline('models/masked_model')
result = pipeline.process_image(image_path='masked_face.jpg')

# Check mask status
for face in result['faces']:
    print(f"Person: {face['prediction']}")
    print(f"Masked: {face['is_masked']}")
    print(f"Confidence: {face['confidence']:.2%}")
```

#### Batch Processing

```python
# Process multiple images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = pipeline.process_batch(image_paths)

for result, image_path in zip(results, image_paths):
    print(f"\n{image_path}:")
    if result['success']:
        for face in result['faces']:
            print(f"  → {face['prediction']} ({face['confidence']:.2%})")
```

---

## 📁 Project Structure

### Unmasked Pipeline

```
mobilenet_lbp_unmasked/
├── src_unmasked/
│   ├── __init__.py
│   ├── pipeline.py           # Main pipeline orchestrator
│   ├── preprocessing.py      # Image preprocessing
│   ├── segmentation.py       # Face detection (YOLO/MTCNN/MediaPipe)
│   ├── lbp_extractor.py      # LBP feature extraction
│   ├── embedding.py          # MobileNetV2 embeddings
│   ├── detector.py           # Cosine similarity identification
│   └── README.md
├── train_unmasked_simple.py  # Quick training script
├── train_unmasked.py         # Full training with options
├── example_usage_unmasked.py # Usage examples
├── QUICKSTART_UNMASKED.md    # Quick reference
└── eva.txt                   # Evaluation metrics
```

### Masked Pipeline

```
mobilenet_lbp_masked/
├── src_masked/
│   ├── __init__.py
│   ├── pipeline.py           # Pipeline with filtering
│   ├── preprocessing.py      # Background removal
│   ├── segmentation.py       # Face + mask detection
│   ├── filtering.py          # Gaussian/Median filtering
│   ├── lbp_extractor.py      # LBP features
│   ├── embedding.py          # Deep embeddings
│   ├── detector.py           # Similarity identification
│   └── README.md
├── train_masked_simple.py    # Quick training script
├── yolov8n.pt                # YOLO model weights
├── eva.txt                   # Evaluation results
└── README.md
```

### Both Pipeline

```
mobilenet_lbp_both/
├── preprocessing.py          # Preprocessing utilities
├── segmentation.py           # Multi-scenario face detection
├── filtering.py              # Flexible filtering
├── lbp_extractor.py          # LBP extraction
├── embedding.py              # Embeddings
├── detector.py               # Cosine similarity
├── pipeline.py               # Unified pipeline
├── train.py                  # Training script
├── inference.py              # Inference script
├── test_model.py             # Testing utilities
├── yolov8n.pt                # YOLO weights
├── README.md                 # Documentation
└── Evaluation.md             # Performance metrics
```

---

## ⚙️ Configuration Options

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_size` | Tuple[int, int] | (256, 256) | Image dimensions (width, height) |
| `remove_bg` | bool | True | Enable background removal |
| `detector_type` | str | 'yolo' | Face detector: 'yolo', 'mtcnn', 'mediapipe' |
| `similarity_threshold` | float | 0.55 | Cosine similarity threshold (0-1) |
| `embedding_dim` | int | 128 | Embedding vector dimension |

### Masked-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_type` | str | 'gaussian' | Filtering: 'gaussian' or 'median' |
| `fine_tune_embedder` | bool | True | Enable embedder fine-tuning |
| `epochs` | int | 20 | Training epochs |
| `batch_size` | int | 16 | Batch size |
| `learning_rate` | float | 0.01 | Learning rate |

### Training Options

```bash
# Unmasked
python train_unmasked.py \
    --train_dir "data/train" \
    --val_dir "data/val" \
    --model_dir "models/my_model" \
    --target_size 256 256 \
    --detector_type yolo \
    --embedding_dim 128 \
    --similarity_threshold 0.55

# Masked
python train_masked_simple.py \
    --learning_rate 0.01 \
    --epochs 20 \
    --batch_size 16 \
    --filter_type gaussian

# Both
python train.py \
    --train_dir "data/train" \
    --val_dir "data/val" \
    --filter_type gaussian \
    --detector_type yolo \
    --remove_bg True
```

---

## 🧠 Model Components

### 1. **Preprocessing Module**

**Functionality:**
- Background removal using semantic segmentation
- Image resizing to uniform dimensions
- Normalization and augmentation

**Output:** Preprocessed image ready for face detection

### 2. **Segmentation Module**

**Available Detectors:**

| Detector | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| **YOLO** | ⚡⚡⚡ Fast | ⭐⭐⭐ High | Real-time applications |
| **MTCNN** | ⚡⚡ Medium | ⭐⭐⭐⭐ Very High | High-quality detection |
| **MediaPipe** | ⚡⚡⚡ Fast | ⭐⭐ Good | Lightweight deployments |

**Output:** Face bounding boxes, mask/no-mask classification

### 3. **Filtering Module** (Masked Pipeline Only)

**Gaussian Filter:**
- Reduces noise while preserving edges
- Ideal for masked faces with compression artifacts
- Kernel size: 5×5 (configurable)

**Median Filter:**
- Preserves edges while removing salt-and-pepper noise
- Alternative for extreme noise conditions

**Output:** Filtered face region ready for feature extraction

### 4. **Feature Extraction**

**LBP (Local Binary Pattern):**
- Texture descriptor capturing local patterns
- Fast computation, rotation-invariant
- 59-dimensional feature vector (uniform patterns)

**MobileNetV2 Embeddings:**
- Deep learning-based feature extraction
- Pre-trained on face recognition tasks
- 128-dimensional embedding (configurable)
- Fine-tunable for domain adaptation

**Output:** Hybrid feature vector (LBP + embeddings)

### 5. **Identification Module**

**Cosine Similarity Matching:**
- Compares feature vectors using cosine distance
- Threshold-based classification
- Unknown person detection

**Confidence Score:**
- Computed as: `1 - cosine_distance`
- Range: [0, 1] (higher = more confident)
- Default threshold: 0.55 (adjustable)

**Output:** Predicted person identity + confidence score

---

## 📊 Performance

### Accuracy Metrics

**Unmasked Pipeline:**
- Accuracy: ~95-98%
- Processing time: ~50-100ms per face
- Memory footprint: ~2-3GB

**Masked Pipeline:**
- Accuracy: ~90-95%
- Mask detection rate: ~98%
- Processing time: ~60-120ms per face (with filtering)
- Memory footprint: ~2-3GB

**Both Pipeline:**
- Mixed accuracy: ~92-96%
- Processing time: ~55-110ms per face
- Memory footprint: ~3-4GB

### Optimization Tips

1. **Batch Processing:** Process multiple images together for efficiency
2. **GPU Acceleration:** Enable CUDA for 5-10x speedup
3. **Model Caching:** Load models once, reuse for multiple inferences
4. **Parameter Tuning:** Adjust thresholds based on your use case

---

## 🔧 Troubleshooting

### Common Issues

**Issue: "Out of memory" error**
```python
# Solution 1: Disable background removal
pipeline = FaceRecognitionPipeline(remove_bg=False)

# Solution 2: Reduce image size
pipeline = FaceRecognitionPipeline(target_size=(192, 192))

# Solution 3: Process in smaller batches
results = pipeline.process_batch(images[:10])  # Process 10 at a time
```

**Issue: Low accuracy on certain faces**
```python
# Solution 1: Adjust similarity threshold
pipeline = FaceRecognitionPipeline(similarity_threshold=0.5)  # More lenient

# Solution 2: Fine-tune on your dataset
pipeline.train(
    train_dir='your_data',
    fine_tune_embedder=True,
    epochs=30
)
```

**Issue: False positives (wrong person identified)**
```python
# Solution: Increase threshold
pipeline = FaceRecognitionPipeline(similarity_threshold=0.65)  # More strict
```

---

## 📈 Results & Benchmarks

### Evaluation Metrics

Detailed evaluation results are available in each pipeline folder:
- `unmasked/eva.txt` - Unmasked pipeline metrics
- `masked/eva.txt` - Masked pipeline metrics
- `both/Evaluation.md` - Mixed scenario results

### ROC Curves

Included ROC curve visualizations:
- `roc_unmasked.png`
- `roc_masked.png`
- `roc_both.png`

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- [ ] Additional face detectors (DSFD, RetinaFace)
- [ ] Alternative embeddings (ArcFace, VGGFace2)
- [ ] Performance optimizations
- [ ] Docker containerization
- [ ] REST API wrapper
- [ ] Web UI dashboard

---

## 📝 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 👨‍💻 Author

**Ahmed Adel**
- GitHub: [@ahmad1adel](https://github.com/ahmad1adel)
- Email: ahmed.adel.tageldeen@hotmail.com

---

## 📚 References & Resources

### Papers

- MobileNetV2: [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- LBP Features: [Face Recognition with Local Binary Patterns](https://ieeexplore.ieee.org/document/1469340)
- Cosine Similarity: [Face Recognition via Centered Coordinate Coding](https://arxiv.org/abs/1003.0391)

### Documentation

- [TensorFlow Documentation](https://www.tensorflow.org/learn)
- [OpenCV Documentation](https://docs.opencv.org/)
- [scikit-image Documentation](https://scikit-image.org/)

### Related Projects

- [FaceNet](https://github.com/davidsandberg/facenet)
- [DeepFace](https://github.com/serengalp/deepface)
- [MTCNN](https://github.com/ipazc/mtcnn)

---

## ⚠️ Disclaimer

This system is provided for educational and research purposes. Ensure compliance with local privacy laws and regulations when deploying facial recognition systems.

---

## 🎯 Future Enhancements

- [ ] Real-time video stream processing
- [ ] Multi-GPU support
- [ ] Model quantization for edge devices
- [ ] REST API deployment
- [ ] Web dashboard for monitoring
- [ ] Advanced analytics and reporting
- [ ] Database integration for large-scale deployments

---

**Last Updated:** January 2, 2026  
**Version:** 2.0.0


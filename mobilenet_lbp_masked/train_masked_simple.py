"""
Quick training script for masked dataset with your specified parameters:
- YOLO detector only
- 20 epochs
- Batch size 16
- Learning rate 0.01
- Fine-tuning enabled
- Gaussian filtering enabled
"""

from src_masked.pipeline import FaceRecognitionPipeline
import os

def main():
    print("=" * 70)
    print("Face Recognition Training - Masked Dataset")
    print("YOLO | 20 Epochs | Batch 16 | LR 0.01 | WITH FILTERING")
    print("=" * 70)
    
    # Configuration
    train_dir = r'..\self-built-masked-face-recognition-dataset\AFDB_masked_face_dataset'  # Relative path from masked folder
    model_dir = 'models/masked_model'
    
    # Check dataset
    if not os.path.exists(train_dir):
        print(f"\nError: Dataset not found at '{train_dir}'")
        return
    
    print(f"\nDataset: {train_dir}")
    print(f"Output: {model_dir}")
    print("\nConfiguration:")
    print("  - Detector: YOLO (fixed)")
    print("  - Identification: Cosine Similarity (threshold: 0.55)")
    print("  - Fine-tuning: ENABLED")
    print("  - Epochs: 20")
    print("  - Batch Size: 16")
    print("  - Learning Rate: 0.01")
    print("  - Filtering: Gaussian (ENABLED)")
    print("  - Mask Detection: ENABLED")
    print("=" * 70)
    
    # Initialize pipeline
    print("\n[1/3] Initializing pipeline...")
    pipeline = FaceRecognitionPipeline(
        target_size=(256, 256),
        remove_bg=False,           # Disabled to avoid memory issues
        filter_type='gaussian',    # FILTERING ENABLED
        detector_type='yolo',      # YOLO only
        similarity_threshold=0.55,  # Cosine similarity threshold
        embedding_dim=128
    )
    print("✓ Pipeline initialized with filtering and cosine similarity")
    
    # Train with fine-tuning
    print("\n[2/3] Training with fine-tuning and filtering...")
    print("-" * 70)
    try:
        pipeline.train(
            train_dir=train_dir,
            val_dir=None,
            fine_tune_embedder=True,  # Enable fine-tuning
            epochs=20,                 # 20 epochs as requested
            batch_size=16,             # Batch size 16 as requested
            learning_rate=0.01         # Learning rate 0.01 as requested
        )
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save models
    print("\n[3/3] Saving models...")
    os.makedirs(model_dir, exist_ok=True)
    pipeline.save_pipeline(model_dir)
    
    print("\n" + "=" * 70)
    print("✓ Training completed successfully!")
    print(f"✓ Models saved to: {model_dir}")
    print("=" * 70)
    print("\nYou can now use the trained model for predictions.")
    print("This model includes:")
    print("  ✓ Cosine similarity for person identification")
    print("  ✓ Gaussian filtering for noise reduction")
    print("  ✓ Mask detection capability")
    print("  ✓ Fine-tuned embedder for masked faces")

if __name__ == '__main__':
    main()

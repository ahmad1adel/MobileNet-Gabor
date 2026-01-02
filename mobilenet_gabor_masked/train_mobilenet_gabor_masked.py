from src_mobilenet_gabor_masked import FaceRecognitionPipeline
import os

def train_pipeline():
    # Initialize pipeline
    pipeline = FaceRecognitionPipeline(
        target_size=(256, 256),
        remove_bg=False, # Set to False for faster training
        detector_type='yolo',
        similarity_threshold=0.55
    )
    
    # Path to masked dataset (assuming same structure as base)
    train_dir = 'dataset/masked/train'
    val_dir = 'dataset/masked/val'
    
    if not os.path.exists(train_dir):
        print(f"Training directory {train_dir} not found. Please ensure dataset is prepared.")
        return

    print("Starting training for MobileNet + Gabor (Masked)...")
    pipeline.train(train_dir, val_dir)
    
    # Save the trained pipeline
    pipeline.save_pipeline('models/mobilenet_gabor_masked')
    print("Training complete and model saved.")

if __name__ == "__main__":
    train_pipeline()

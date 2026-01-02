from src_mobilenet_gabor_unmasked import FaceRecognitionPipeline
import os

def train_pipeline():
    pipeline = FaceRecognitionPipeline(
        remove_bg=False,
        similarity_threshold=0.55
    )
    
    # Path to unmasked dataset
    train_dir = 'dataset/unmasked/train'
    val_dir = 'dataset/unmasked/val'
    
    if not os.path.exists(train_dir):
        print(f"Training directory {train_dir} not found.")
        return

    print("Starting training for MobileNet + Gabor (Unmasked)...")
    pipeline.train(train_dir, val_dir)
    pipeline.save_pipeline('models/mobilenet_gabor_unmasked')
    print("Training complete.")

if __name__ == "__main__":
    train_pipeline()

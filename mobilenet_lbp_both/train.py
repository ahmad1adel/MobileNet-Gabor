"""
Training Script for Face Recognition Model
"""

import argparse
import os
import sys
from src.pipeline import FaceRecognitionPipeline


def main():
    parser = argparse.ArgumentParser(description='Train Face Recognition Model')
    parser.add_argument('--train_dir', type=str, required=True,
                       help='Directory containing training data (person folders)')
    parser.add_argument('--val_dir', type=str, default=None,
                       help='Directory containing validation data (optional)')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256],
                       help='Target image size (width height)')
    parser.add_argument('--remove_bg', action='store_true', default=False,
                       help='Remove background from images (disabled by default for memory)')
    parser.add_argument('--filter_type', type=str, default='gaussian',
                       choices=['gaussian', 'median'],
                       help='Filter type for noise reduction')
    parser.add_argument('--detector_type', type=str, default='yolo',
                       choices=['yolo', 'mtcnn', 'mediapipe'],
                       help='Face detector type')
    parser.add_argument('--similarity_threshold', type=float, default=0.55,
                       help='Cosine similarity threshold for person identification')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    
    args = parser.parse_args()
    
    # Check if training directory exists
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory '{args.train_dir}' does not exist")
        sys.exit(1)
    
    # Initialize pipeline
    print("======================================================================")
    print("Face Recognition Training - Base Pipeline")
    print(f"Identification: Cosine Similarity (threshold: {args.similarity_threshold})")
    print("======================================================================")
    
    print("\n[1/3] Initializing pipeline...")
    pipeline = FaceRecognitionPipeline(
        target_size=tuple(args.target_size),
        remove_bg=args.remove_bg,
        filter_type=args.filter_type,
        detector_type=args.detector_type,
        embedding_dim=args.embedding_dim,
        similarity_threshold=args.similarity_threshold
    )
    
    # Train model
    print("\n[2/3] Extracting features and training detector...")
    pipeline.train(args.train_dir, args.val_dir)
    
    # Save pipeline
    print("\n[3/3] Saving trained models...")
    os.makedirs(args.model_dir, exist_ok=True)
    pipeline.save_pipeline(args.model_dir)
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()


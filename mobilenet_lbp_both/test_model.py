"""
Test the trained face recognition model
"""

import argparse
import os
import sys
from src.pipeline import FaceRecognitionPipeline
import cv2


def test_on_image(pipeline, image_path):
    """Test model on a single image"""
    print(f"\nTesting on: {image_path}")
    print("-" * 50)
    
    result = pipeline.process_image(image_path=image_path)
    
    if result['success']:
        print(f"✓ Detected {result['num_faces']} face(s)\n")
        
        for i, face_data in enumerate(result['faces']):
            print(f"Face {i+1}:")
            print(f"  - Masked: {face_data['is_masked']} (confidence: {face_data['mask_confidence']:.2f})")
            
            if face_data.get('prediction'):
                print(f"  - Identity: {face_data['prediction']}")
                print(f"  - Confidence: {face_data['confidence']:.2f}")
            else:
                print(f"  - Identity: Not predicted (model not trained)")
            print()
    else:
        print(f"✗ {result['message']}")


def test_on_directory(pipeline, test_dir):
    """Test model on all images in a directory"""
    print(f"\nTesting on directory: {test_dir}")
    print("=" * 50)
    
    total_images = 0
    successful = 0
    correct_predictions = 0
    
    # Get person folders
    for person_name in os.listdir(test_dir):
        person_dir = os.path.join(test_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        print(f"\nTesting {person_name}...")
        person_correct = 0
        person_total = 0
        
        # Test images in person folder
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            img_path = os.path.join(person_dir, img_file)
            total_images += 1
            person_total += 1
            
            try:
                result = pipeline.process_image(image_path=img_path)
                
                if result['success'] and len(result['faces']) > 0:
                    successful += 1
                    face_data = result['faces'][0]
                    
                    if face_data.get('prediction') == person_name:
                        correct_predictions += 1
                        person_correct += 1
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
        
        if person_total > 0:
            accuracy = person_correct / person_total * 100
            print(f"  Accuracy for {person_name}: {accuracy:.1f}% ({person_correct}/{person_total})")
    
    print("\n" + "=" * 50)
    print("Overall Results:")
    print(f"  Total images: {total_images}")
    print(f"  Successfully processed: {successful}")
    print(f"  Correct predictions: {correct_predictions}")
    if successful > 0:
        print(f"  Overall accuracy: {correct_predictions/successful*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Test Face Recognition Model')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single test image')
    parser.add_argument('--test_dir', type=str, default='data/test',
                       help='Directory containing test images')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256],
                       help='Target image size (width height)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' does not exist")
        print("Please train the model first using: python train.py")
        sys.exit(1)
    
    # Initialize pipeline
    print("Loading Face Recognition Pipeline...")
    pipeline = FaceRecognitionPipeline(
        target_size=tuple(args.target_size),
        remove_bg=True,
        filter_type='gaussian',
        detector_type='mtcnn',
        classifier_type='svm'
    )
    
    # Load trained models
    try:
        pipeline.load_pipeline(args.model_dir)
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)
    
    # Test on single image or directory
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image '{args.image}' does not exist")
            sys.exit(1)
        test_on_image(pipeline, args.image)
    else:
        if not os.path.exists(args.test_dir):
            print(f"Error: Test directory '{args.test_dir}' does not exist")
            sys.exit(1)
        test_on_directory(pipeline, args.test_dir)


if __name__ == '__main__':
    main()


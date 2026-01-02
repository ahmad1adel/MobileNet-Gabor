"""
Example usage of the Face Recognition Pipeline for Unmasked Dataset
"""

from src_unmasked.pipeline import FaceRecognitionPipeline
import cv2


# Example 1: Initialize and use pipeline
def example_basic_usage():
    """Basic usage example"""
    print("Example 1: Basic Pipeline Usage (Unmasked Dataset)")
    print("-" * 50)
    
    # Initialize pipeline (no filtering for unmasked dataset)
    pipeline = FaceRecognitionPipeline(
        target_size=(256, 256),
        remove_bg=True,
        detector_type='yolo',  # or 'mtcnn' or 'mediapipe'
        classifier_type='svm'
    )
    
    # Process an image (replace with your image path)
    # result = pipeline.process_image(image_path='path/to/your/image.jpg')
    
    # if result['success']:
    #     print(f"Detected {result['num_faces']} face(s)")
    #     for i, face_data in enumerate(result['faces']):
    #         print(f"\nFace {i+1}:")
    #         print(f"  Bounding Box: {face_data['bbox']}")
    #         if face_data.get('prediction'):
    #             print(f"  Identity: {face_data['prediction']}")
    #             print(f"  Confidence: {face_data['confidence']:.2f}")
    
    print("Pipeline initialized successfully!")


# Example 2: Training on Unmasked Dataset
def example_training():
    """Training example for unmasked dataset"""
    print("\nExample 2: Training on Unmasked Dataset")
    print("-" * 50)
    
    pipeline = FaceRecognitionPipeline(
        target_size=(256, 256),
        remove_bg=True,
        detector_type='yolo',
        classifier_type='svm'
    )
    
    # Train on unmasked dataset
    # pipeline.train(
    #     train_dir='Proposed dataset/Dataset-Without mask',
    #     val_dir=None  # Optional validation directory
    # )
    
    # Save the trained model
    # pipeline.save_pipeline('models/unmasked_model')
    
    print("Training would be done here...")


# Example 3: Process batch of images
def example_batch_processing():
    """Example of batch processing"""
    print("\nExample 3: Batch Processing")
    print("-" * 50)
    
    pipeline = FaceRecognitionPipeline()
    
    # Process multiple images
    # image_paths = [
    #     'Proposed dataset/Dataset-Without mask/person1/img1.jpg',
    #     'Proposed dataset/Dataset-Without mask/person1/img2.jpg',
    #     'Proposed dataset/Dataset-Without mask/person2/img1.jpg',
    # ]
    
    # results = pipeline.process_batch(image_paths)
    
    # for result in results:
    #     if result['success']:
    #         print(f"Image: {result['image_path']}")
    #         print(f"  Faces detected: {result['num_faces']}")
    
    print("Batch processing example ready...")


# Example 4: Load pre-trained model and predict
def example_inference():
    """Example of using pre-trained model"""
    print("\nExample 4: Inference with Pre-trained Model")
    print("-" * 50)
    
    pipeline = FaceRecognitionPipeline()
    
    # Load pre-trained model
    # pipeline.load_pipeline('models/unmasked_model')
    
    # Process new image
    # result = pipeline.process_image(image_path='test_image.jpg')
    
    # if result['success'] and len(result['faces']) > 0:
    #     face = result['faces'][0]
    #     print(f"Predicted Identity: {face['prediction']}")
    #     print(f"Confidence: {face['confidence']:.2%}")
    
    print("Inference example ready...")


# Example 5: Using individual components
def example_individual_components():
    """Example using individual pipeline components"""
    print("\nExample 5: Using Individual Components")
    print("-" * 50)
    
    from src_unmasked.preprocessing import ImagePreprocessor
    from src_unmasked.segmentation import FaceSegmenter
    from src_unmasked.lbp_extractor import LBPExtractor
    from src_unmasked.embedding import FaceEmbedder
    
    # Initialize components
    preprocessor = ImagePreprocessor(target_size=(256, 256), remove_bg=True)
    segmenter = FaceSegmenter(detector_type='yolo')
    lbp_extractor = LBPExtractor()
    embedder = FaceEmbedder()
    
    # Use components individually
    # image = cv2.imread('path/to/image.jpg')
    # preprocessed = preprocessor.preprocess(image=image)
    # faces = segmenter.segment_face(preprocessed)
    # 
    # if faces['num_faces'] > 0:
    #     face_img = faces['faces'][0]['face_image']
    #     lbp_features = lbp_extractor.extract_features(face_img, use_spatial=True)
    #     embedding = embedder.extract_embedding(face_img)
    #     print(f"LBP features shape: {lbp_features.shape}")
    #     print(f"Embedding shape: {embedding.shape}")
    
    print("Individual components can be used separately...")


if __name__ == '__main__':
    print("=" * 50)
    print("Face Recognition Pipeline - Unmasked Dataset")
    print("=" * 50)
    
    example_basic_usage()
    example_training()
    example_batch_processing()
    example_inference()
    example_individual_components()
    
    print("\n" + "=" * 50)
    print("Key Differences from Masked Pipeline:")
    print("  - No filtering step applied")
    print("  - No mask detection")
    print("  - Optimized for unmasked faces")
    print("\nSee src_unmasked/README.md for details")
    print("=" * 50)

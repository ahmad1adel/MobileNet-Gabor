"""
Complete Face Recognition Pipeline - Masked Dataset
End-to-end processing from image to person identification (WITH filtering and mask detection)
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

from .preprocessing import ImagePreprocessor
from .segmentation import FaceSegmenter
from .filtering import ImageFilter
from .lbp_extractor import LBPExtractor
from .embedding import FaceEmbedder
from .detector import PersonDetector


class FaceRecognitionPipeline:
    """Complete face recognition pipeline for masked dataset"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (256, 256),
                 remove_bg: bool = True,
                 filter_type: str = 'gaussian',
                 detector_type: str = 'yolo',
                 embedding_dim: int = 128,
                 similarity_threshold: float = 0.55):
        """
        Initialize pipeline with cosine similarity detector
        
        Args:
            target_size: Target image size
            remove_bg: Whether to remove background
            filter_type: Filter type ('gaussian' or 'median')
            detector_type: Face detector type ('yolo', 'mtcnn' or 'mediapipe')
            embedding_dim: Embedding dimension
            similarity_threshold: Minimum similarity score for identification (0-1)
        """
        # Initialize components (WITH FILTERING for masked dataset)
        self.preprocessor = ImagePreprocessor(target_size=target_size, remove_bg=remove_bg)
        self.segmenter = FaceSegmenter(detector_type=detector_type)
        self.filter = ImageFilter(filter_type=filter_type, kernel_size=5)  # ADD FILTERING
        self.lbp_extractor = LBPExtractor(num_points=8, radius=1, method='uniform')
        # Convert target_size from (width, height) to (height, width) for TensorFlow
        self.embedder = FaceEmbedder(input_size=(target_size[1], target_size[0]), embedding_dim=embedding_dim)
        self.detector = PersonDetector(similarity_threshold=similarity_threshold, combine_features=True)
    
    def process_image(self, image_path: Optional[str] = None,
                     image: Optional[np.ndarray] = None) -> Dict:
        """
        Process single image through complete pipeline (WITH filtering and mask detection)
        
        Args:
            image_path: Path to image
            image: Image as numpy array
            
        Returns:
            Dictionary with processing results
        """
        # Step 1: Preprocessing
        preprocessed = self.preprocessor.preprocess(image_path=image_path, image=image)
        
        # Step 2: Segmentation (WITH mask detection)
        segmentation_result = self.segmenter.segment_face(preprocessed, return_mask_info=True)
        
        if segmentation_result['num_faces'] == 0:
            return {
                'success': False,
                'message': 'No faces detected',
                'preprocessed_image': preprocessed
            }
        
        # Process each detected face
        results = []
        for face_data in segmentation_result['faces']:
            try:
                face_img = face_data['face_image']
                
                # Validate face image
                if face_img is None or face_img.size == 0:
                    print(f"Warning: Skipping invalid face image")
                    continue
                
                if len(face_img.shape) != 3 or face_img.shape[2] != 3:
                    print(f"Warning: Skipping face with invalid shape: {face_img.shape}")
                    continue
                
                # Step 2.5: FOCUS ON UPPER FACE (Eyes/Forehead)
                # Crop to upper 60% of the face for recognition focus
                h_orig, w_orig = face_img.shape[:2]
                face_img = face_img[0:int(h_orig * 0.6), :]
                
                # APPLY FILTERING STEP (key difference from unmasked)
                filtered_face = self.filter.apply_filter(face_img)
                
                # Validate filtered face
                if filtered_face is None or filtered_face.size == 0:
                    print(f"Warning: Filtering resulted in invalid image")
                    continue
                
                # Resize filtered face to target size if needed
                if filtered_face.shape[:2] != self.preprocessor.target_size[::-1]:
                    filtered_face = cv2.resize(
                        filtered_face, 
                        self.preprocessor.target_size, 
                        interpolation=cv2.INTER_AREA
                    )
                
                # Validate resized face
                if filtered_face.shape[0] < 32 or filtered_face.shape[1] < 32:
                    print(f"Warning: Face too small after resize: {filtered_face.shape}")
                    continue
                
                # Step 3: LBP extraction
                lbp_features = self.lbp_extractor.extract_features(
                    filtered_face, 
                    use_multiscale=False,
                    use_spatial=True
                )
                
                # Step 4: Embedding extraction
                embedding = self.embedder.extract_embedding(filtered_face)
                
                # Step 5: Prediction (if detector is trained)
                prediction = None
                confidence = None
                if self.detector.is_trained:
                    try:
                        prediction, confidence = self.detector.predict_single(embedding, lbp_features)
                    except Exception as e:
                        print(f"Prediction error: {e}")
                
                results.append({
                    'face_image': filtered_face,
                    'bbox': face_data['bbox'],
                    'is_masked': face_data.get('is_masked', False),
                    'mask_confidence': face_data.get('mask_confidence', 0.0),
                    'lbp_features': lbp_features,
                    'embedding': embedding,
                    'prediction': prediction,
                    'confidence': confidence
                })
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        return {
            'success': True,
            'num_faces': len(results),
            'faces': results,
            'preprocessed_image': preprocessed
        }
    
    def process_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of processing results
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.process_image(image_path=img_path)
                result['image_path'] = img_path
                results.append(result)
            except Exception as e:
                results.append({
                    'success': False,
                    'image_path': img_path,
                    'error': str(e)
                })
        
        return results
    
    def prepare_training_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from directory structure
        Expected structure: data_dir/person_name/image_files
        
        Args:
            data_dir: Root directory containing person folders
            
        Returns:
            Tuple of (embeddings, lbp_features, labels)
        """
        embeddings_list = []
        lbp_features_list = []
        labels_list = []
        
        # Iterate through person folders
        for person_name in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            print(f"Processing {person_name}...")
            
            # Process images in person folder
            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                img_path = os.path.join(person_dir, img_file)
                
                try:
                    # Process image
                    result = self.process_image(image_path=img_path)
                    
                    if result['success'] and len(result['faces']) > 0:
                        # Use first face
                        face_data = result['faces'][0]
                        embeddings_list.append(face_data['embedding'])
                        lbp_features_list.append(face_data['lbp_features'])
                        labels_list.append(person_name)
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        embeddings = np.array(embeddings_list)
        lbp_features = np.array(lbp_features_list)
        
        return embeddings, lbp_features, labels_list
    
    def train(self, train_dir: str, val_dir: Optional[str] = None,
              fine_tune_embedder: bool = False, epochs: int = 20, 
              batch_size: int = 16, learning_rate: float = 0.01):
        """
        Train the detector and optionally fine-tune the embedder
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory (optional)
            fine_tune_embedder: Whether to fine-tune the embedding model
            epochs: Number of epochs for fine-tuning (default: 20)
            batch_size: Batch size for fine-tuning (default: 16)
            learning_rate: Learning rate for fine-tuning (default: 0.01)
        """
        print("Preparing training data...")
        
        # If fine-tuning embedder, we need to collect images and labels
        if fine_tune_embedder:
            print("\n" + "="*60)
            print("Fine-tuning embedder model...")
            print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
            print("="*60)
            
            # Collect images for fine-tuning
            train_images = []
            train_image_labels = []
            label_map = {}
            current_label_id = 0
            
            for person_name in os.listdir(train_dir):
                person_dir = os.path.join(train_dir, person_name)
                if not os.path.isdir(person_dir):
                    continue
                
                if person_name not in label_map:
                    label_map[person_name] = current_label_id
                    current_label_id += 1
                
                print(f"Loading images for {person_name}...")
                
                for img_file in os.listdir(person_dir):
                    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        continue
                    
                    img_path = os.path.join(person_dir, img_file)
                    try:
                        # Load and preprocess image
                        preprocessed = self.preprocessor.preprocess(image_path=img_path)
                        segmentation_result = self.segmenter.segment_face(preprocessed, return_mask_info=True)
                        
                        if segmentation_result['num_faces'] > 0:
                            face_img = segmentation_result['faces'][0]['face_image']
                            
                            # Apply filtering
                            filtered_face = self.filter.apply_filter(face_img)
                            
                            # Resize to target size
                            if filtered_face.shape[:2] != self.preprocessor.target_size[::-1]:
                                filtered_face = cv2.resize(
                                    filtered_face, 
                                    self.preprocessor.target_size, 
                                    interpolation=cv2.INTER_AREA
                                )
                            
                            train_images.append(filtered_face)
                            train_image_labels.append(label_map[person_name])
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
            
            print(f"\nCollected {len(train_images)} images for fine-tuning")
            
            # Fine-tune embedder
            train_images_array = np.array(train_images)
            train_labels_array = np.array(train_image_labels)
            
            self.embedder.fine_tune_model(
                train_images_array, 
                train_labels_array,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            print("\n✓ Embedder fine-tuning completed!")
            print("="*60 + "\n")
        
        # Now extract features with the (possibly fine-tuned) embedder
        print("Extracting features for cosine similarity detector training...")
        train_embeddings, train_lbp, train_labels = self.prepare_training_data(train_dir)
        
        print(f"\nTraining cosine similarity detector on {len(train_labels)} samples from {len(set(train_labels))} persons")
        
        # Train detector
        self.detector.train(train_embeddings, train_lbp, train_labels)
        
        # Evaluate on validation set if provided
        if val_dir:
            print("\nPreparing validation data...")
            val_embeddings, val_lbp, val_labels = self.prepare_training_data(val_dir)
            
            print("Evaluating on validation set...")
            metrics = self.detector.evaluate(val_embeddings, val_lbp, val_labels)
            print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
            print(f"Validation F1-Score: {metrics['f1_score']:.4f}")
    
    def save_pipeline(self, model_dir: str):
        """Save all pipeline components"""
        os.makedirs(model_dir, exist_ok=True)
        
        self.embedder.save_model(os.path.join(model_dir, 'embedder.keras'))
        self.detector.save(os.path.join(model_dir, 'detector.pkl'))
        
        print(f"Pipeline saved to {model_dir}")
    
    def load_pipeline(self, model_dir: str):
        """Load pipeline components"""
        # Try .keras first, then .h5 for backward compatibility
        embedder_path = os.path.join(model_dir, 'embedder.keras')
        if not os.path.exists(embedder_path):
            embedder_path = os.path.join(model_dir, 'embedder.h5')
        
        # Load embedder (will rebuild if there are issues)
        try:
            self.embedder.load_model(embedder_path)
            print("✓ Embedder loaded")
        except Exception as e:
            print(f"Warning: Embedder load failed: {e}")
            print("Using new embedder with pretrained weights")
        
        # Load detector
        detector_path = os.path.join(model_dir, 'detector.pkl')
        if os.path.exists(detector_path):
            try:
                self.detector.load(detector_path)
                print("✓ Detector loaded")
            except Exception as e:
                print(f"Warning: Detector load failed: {e}")
        
        print(f"Pipeline loaded from {model_dir}")


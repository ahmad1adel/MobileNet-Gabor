"""
Complete Face Recognition Pipeline
End-to-end processing from image to person identification using MobileNet and Gabor features
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

from .preprocessing import ImagePreprocessor
from .segmentation import FaceSegmenter
from .filtering import ImageFilter
from .gabor_extractor import GaborExtractor
from .embedding import FaceEmbedder
from .detector import PersonDetector


class FaceRecognitionPipeline:
    """Complete face recognition pipeline"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (256, 256),
                 remove_bg: bool = True,
                 filter_type: str = 'gaussian',
                 detector_type: str = 'yolo',
                 embedding_dim: int = 128,
                 similarity_threshold: float = 0.55):
        """
        Initialize pipeline
        
        Args:
            target_size: Target image size
            remove_bg: Whether to remove background
            filter_type: Filter type ('gaussian' or 'median')
            detector_type: Face detector type ('yolo', 'mtcnn' or 'mediapipe')
            embedding_dim: Embedding dimension
            similarity_threshold: Cosine similarity threshold (0-1)
        """
        # Initialize components
        self.preprocessor = ImagePreprocessor(target_size=target_size, remove_bg=remove_bg)
        self.segmenter = FaceSegmenter(detector_type=detector_type)
        self.filter = ImageFilter(filter_type=filter_type, kernel_size=5)
        self.gabor_extractor = GaborExtractor(orientations=4, scales=2)
        # Convert target_size from (width, height) to (height, width) for TensorFlow
        self.embedder = FaceEmbedder(input_size=(target_size[1], target_size[0]), embedding_dim=embedding_dim)
        self.detector = PersonDetector(similarity_threshold=similarity_threshold, combine_features=True)
    
    def process_image(self, image_path: Optional[str] = None,
                     image: Optional[np.ndarray] = None) -> Dict:
        """
        Process single image through complete pipeline
        """
        # Step 1: Preprocessing
        preprocessed = self.preprocessor.preprocess(image_path=image_path, image=image)
        
        # Step 2: Segmentation
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
                    continue
                
                # Step 3: Filtering
                filtered_face = self.filter.apply_filter(face_img)
                
                # Resize filtered face to target size if needed
                if filtered_face.shape[:2] != self.preprocessor.target_size[::-1]:
                    filtered_face = cv2.resize(
                        filtered_face, 
                        self.preprocessor.target_size, 
                        interpolation=cv2.INTER_AREA
                    )
                
                # Step 4: Gabor extraction
                gabor_features = self.gabor_extractor.extract(filtered_face)
                
                # Step 5: Embedding extraction
                embedding = self.embedder.extract_embedding(filtered_face)
                
                # Step 6: Prediction (if detector is trained)
                prediction = None
                confidence = None
                if self.detector.is_trained:
                    try:
                        prediction, confidence = self.detector.predict_single(embedding, gabor_features)
                    except Exception as e:
                        print(f"Prediction error: {e}")
                
                results.append({
                    'face_image': filtered_face,
                    'bbox': face_data['bbox'],
                    'is_masked': face_data.get('is_masked', False),
                    'mask_confidence': face_data.get('mask_confidence', 0.0),
                    'gabor_features': gabor_features,
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
        """Process multiple images"""
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
        """Prepare training data from directory structure"""
        embeddings_list = []
        gabor_features_list = []
        labels_list = []
        
        for person_name in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            print(f"Processing {person_name}...")
            
            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                img_path = os.path.join(person_dir, img_file)
                try:
                    result = self.process_image(image_path=img_path)
                    if result['success'] and len(result['faces']) > 0:
                        face_data = result['faces'][0]
                        embeddings_list.append(face_data['embedding'])
                        gabor_features_list.append(face_data['gabor_features'])
                        labels_list.append(person_name)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        return np.array(embeddings_list), np.array(gabor_features_list), labels_list
    
    def train(self, train_dir: str, val_dir: Optional[str] = None):
        """Train the detector"""
        print("Preparing training data...")
        train_embeddings, train_gabor, train_labels = self.prepare_training_data(train_dir)
        
        print(f"Training on {len(train_labels)} samples...")
        self.detector.train(train_embeddings, train_gabor, train_labels)
        
        if val_dir:
            print("Preparing validation data...")
            val_embeddings, val_gabor, val_labels = self.prepare_training_data(val_dir)
            metrics = self.detector.evaluate(val_embeddings, val_gabor, val_labels)
            print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    
    def save_pipeline(self, model_dir: str):
        """Save all pipeline components"""
        os.makedirs(model_dir, exist_ok=True)
        self.embedder.save_model(os.path.join(model_dir, 'embedder.keras'))
        self.detector.save(os.path.join(model_dir, 'detector.pkl'))
        
    def load_pipeline(self, model_dir: str):
        """Load pipeline components"""
        self.embedder.load_model(os.path.join(model_dir, 'embedder.keras'))
        self.detector.load(os.path.join(model_dir, 'detector.pkl'))

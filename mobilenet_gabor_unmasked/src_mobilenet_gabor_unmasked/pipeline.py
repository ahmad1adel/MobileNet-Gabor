"""
Complete Face Recognition Pipeline (Unmasked Optimized)
End-to-end processing from image to person identification using MobileNet and Gabor features
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

from .preprocessing import ImagePreprocessor
from .segmentation import FaceSegmenter
from .gabor_extractor import GaborExtractor
from .embedding import FaceEmbedder
from .detector import PersonDetector


class FaceRecognitionPipeline:
    """Complete face recognition pipeline optimized for unmasked faces"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (256, 256),
                 remove_bg: bool = False,
                 detector_type: str = 'yolo',
                 embedding_dim: int = 128,
                 similarity_threshold: float = 0.55):
        """
        Initialize pipeline
        """
        self.preprocessor = ImagePreprocessor(target_size=target_size, remove_bg=remove_bg)
        self.segmenter = FaceSegmenter(detector_type=detector_type)
        self.gabor_extractor = GaborExtractor(orientations=4, scales=2)
        self.embedder = FaceEmbedder(input_size=(target_size[1], target_size[0]), embedding_dim=embedding_dim)
        self.detector = PersonDetector(similarity_threshold=similarity_threshold, combine_features=True)
    
    def process_image(self, image_path: Optional[str] = None,
                     image: Optional[np.ndarray] = None) -> Dict:
        """
        Process single image (Simplified for unmasked)
        """
        # Step 1: Preprocessing
        preprocessed = self.preprocessor.preprocess(image_path=image_path, image=image)
        
        # Step 2: Segmentation (Face only)
        segmentation_result = self.segmenter.segment_face(preprocessed, return_mask_info=False)
        
        if segmentation_result['num_faces'] == 0:
            return {'success': False, 'message': 'No faces detected'}
        
        results = []
        for face_data in segmentation_result['faces']:
            try:
                face_img = face_data['face_image']
                if face_img is None or face_img.size == 0: continue
                

                # Resize if needed
                if face_img.shape[:2] != self.preprocessor.target_size[::-1]:
                    face_img = cv2.resize(face_img, self.preprocessor.target_size, interpolation=cv2.INTER_AREA)
                
                # Step 3: Gabor extraction
                gabor_features = self.gabor_extractor.extract(face_img)
                
                # Step 4: Embedding extraction
                embedding = self.embedder.extract_embedding(face_img)
                
                # Step 5: Prediction
                prediction = None
                confidence = None
                if self.detector.is_trained:
                    prediction, confidence = self.detector.predict_single(embedding, gabor_features)
                
                results.append({
                    'face_image': face_img,
                    'bbox': face_data['bbox'],
                    'gabor_features': gabor_features,
                    'embedding': embedding,
                    'prediction': prediction,
                    'confidence': confidence
                })
            except Exception: continue
        
        return {
            'success': True,
            'num_faces': len(results),
            'faces': results
        }
    
    def prepare_training_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data"""
        embeddings_list = []
        gabor_features_list = []
        labels_list = []
        
        for person_name in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_dir): continue
            
            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                
                img_path = os.path.join(person_dir, img_file)
                try:
                    result = self.process_image(image_path=img_path)
                    if result['success'] and result['num_faces'] > 0:
                        face_data = result['faces'][0]
                        embeddings_list.append(face_data['embedding'])
                        gabor_features_list.append(face_data['gabor_features'])
                        labels_list.append(person_name)
                except Exception: continue
        
        return np.array(embeddings_list), np.array(gabor_features_list), labels_list
    
    def train(self, train_dir: str, val_dir: Optional[str] = None):
        """Train the detector"""
        train_embeddings, train_gabor, train_labels = self.prepare_training_data(train_dir)
        self.detector.train(train_embeddings, train_gabor, train_labels)
        
        if val_dir:
            val_embeddings, val_gabor, val_labels = self.prepare_training_data(val_dir)
            metrics = self.detector.evaluate(val_embeddings, val_gabor, val_labels)
            print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    
    def save_pipeline(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        self.embedder.save_model(os.path.join(model_dir, 'embedder.keras'))
        self.detector.save(os.path.join(model_dir, 'detector.pkl'))

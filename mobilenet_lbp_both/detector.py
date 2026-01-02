"""
Detector Module
Person identification using cosine similarity with stored reference embeddings
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import pickle
import os
from typing import Literal, Optional, Tuple, Dict


class PersonDetector:
    """Person identification using cosine similarity"""
    
    def __init__(self, similarity_threshold: float = 0.55,
                 combine_features: bool = True):
        """
        Initialize detector
        
        Args:
            similarity_threshold: Cosine similarity threshold for identification
            combine_features: Whether to combine embeddings and LBP features
        """
        self.similarity_threshold = similarity_threshold
        self.combine_features = combine_features
        self.scaler = StandardScaler()
        self.reference_signatures = {}  # Map person names to reference embeddings
        self.is_trained = False
    
    def combine_embedding_lbp(self, embeddings: np.ndarray, 
                               lbp_features: np.ndarray) -> np.ndarray:
        """Combine embedding and LBP features with normalization"""
        # Normalize each feature type
        embedding_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-7)
        lbp_norm = lbp_features / (np.linalg.norm(lbp_features, axis=1, keepdims=True) + 1e-7)
        
        # Concatenate
        combined = np.hstack([embedding_norm, lbp_norm])
        return combined

    def train(self, embeddings: np.ndarray, lbp_features: np.ndarray,
              labels: list, use_cross_validation: bool = True):
        """
        'Train' by storing average signatures for each person
        """
        if self.combine_features:
            features = self.combine_embedding_lbp(embeddings, lbp_features)
        else:
            features = embeddings
            
        # Standard scaling (though cosine similarity is often used on raw/normalized embeddings)
        features_scaled = self.scaler.fit_transform(features)
        
        # Calculate reference signature (mean) for each person
        unique_labels = np.unique(labels)
        self.reference_signatures = {}
        
        for label in unique_labels:
            mask = (np.array(labels) == label)
            person_features = features_scaled[mask]
            self.reference_signatures[label] = np.mean(person_features, axis=0)
            
        self.is_trained = True
        
        if use_cross_validation:
            # Simple internal evaluation
            preds, _ = self.predict(embeddings, lbp_features)
            acc = accuracy_score(labels, preds)
            print(f"Identification accuracy on training set: {acc:.4f}")

    def calculate_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        return dot_product / (norm1 * norm2 + 1e-7)

    def predict(self, embeddings: np.ndarray, lbp_features: np.ndarray) -> Tuple[list, np.ndarray]:
        """Predict using cosine similarity against stored references"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        if self.combine_features:
            features = self.combine_embedding_lbp(embeddings, lbp_features)
        else:
            features = embeddings
            
        features_scaled = self.scaler.transform(features)
        
        predictions = []
        confidences = []
        
        for feat in features_scaled:
            best_score = -1.0
            best_label = "Unknown"
            
            for label, ref_feat in self.reference_signatures.items():
                score = self.calculate_similarity(feat, ref_feat)
                if score > best_score:
                    best_score = score
                    best_label = label
            
            # Apply threshold
            if best_score < self.similarity_threshold:
                predictions.append("Unknown")
            else:
                predictions.append(best_label)
            confidences.append(max(0.0, min(1.0, best_score)))
            
        return predictions, np.array(confidences)

    def predict_single(self, embedding: np.ndarray, lbp_feature: np.ndarray) -> Tuple[str, float]:
        embedding = embedding.reshape(1, -1)
        lbp_feature = lbp_feature.reshape(1, -1)
        labels, confs = self.predict(embedding, lbp_feature)
        return labels[0], float(confs[0])

    def evaluate(self, embeddings: np.ndarray, lbp_features: np.ndarray,
                 true_labels: list) -> dict:
        predicted_labels, _ = self.predict(embeddings, lbp_features)
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1
        }

    def save(self, filepath: str):
        save_data = {
            'signatures': self.reference_signatures,
            'scaler': self.scaler,
            'threshold': self.similarity_threshold,
            'combine_features': self.combine_features,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
            
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        self.reference_signatures = save_data['signatures']
        self.scaler = save_data['scaler']
        self.similarity_threshold = save_data.get('threshold', 0.55)
        self.combine_features = save_data.get('combine_features', True)
        self.is_trained = save_data['is_trained']


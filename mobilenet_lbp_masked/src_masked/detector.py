"""
Detector Module - Masked Dataset with Cosine Similarity
Person identification using cosine similarity instead of classifiers
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Literal, Optional, Tuple, List


class PersonDetector:
    """Person identification detector using cosine similarity"""
    
    def __init__(self, similarity_threshold: float = 0.55, combine_features: bool = True):
        """
        Initialize detector with cosine similarity
        
        Args:
            similarity_threshold: Minimum similarity score for identification (0-1)
            combine_features: Whether to combine embeddings and LBP features
        """
        self.similarity_threshold = similarity_threshold
        self.combine_features = combine_features
        self.scaler = StandardScaler()
        self.label_encoder = {}  # Map person names to integer labels
        self.reverse_label_encoder = {}  # Map integer labels to person names
        self.is_trained = False
        
        # Storage for reference embeddings
        self.reference_embeddings = None  # Shape: (num_persons, embedding_dim)
        self.reference_labels = None  # Shape: (num_persons,)
        self.person_names = []  # List of unique person names
    
    def _encode_labels(self, labels: list) -> np.ndarray:
        """Encode string labels to integers"""
        unique_labels = sorted(list(set(labels)))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
        
        return np.array([self.label_encoder[label] for label in labels])
    
    def _decode_labels(self, encoded_labels: np.ndarray) -> list:
        """Decode integer labels to strings"""
        return [self.reverse_label_encoder[label] for label in encoded_labels]
    
    def combine_embedding_lbp(self, embeddings: np.ndarray, 
                              lbp_features: np.ndarray) -> np.ndarray:
        """Combine embedding and LBP features"""
        # Normalize each feature type
        embedding_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-7)
        lbp_norm = lbp_features / (np.linalg.norm(lbp_features, axis=1, keepdims=True) + 1e-7)
        
        # Concatenate
        combined = np.hstack([embedding_norm, lbp_norm])
        
        return combined
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-7)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-7)
        
        # Compute cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Convert to 0-1 range (from -1 to 1)
        similarity = (similarity + 1) / 2
        
        return similarity
    
    def train(self, embeddings: np.ndarray, lbp_features: np.ndarray,
              labels: list, use_cross_validation: bool = True):
        """
        Train the detector by storing reference embeddings
        
        Args:
            embeddings: Deep learning embeddings
            lbp_features: LBP feature vectors
            labels: Person labels
            use_cross_validation: Whether to evaluate using cross-validation
        """
        # Encode labels
        encoded_labels = self._encode_labels(labels)
        
        # Combine features if requested
        if self.combine_features:
            features = self.combine_embedding_lbp(embeddings, lbp_features)
        else:
            features = embeddings  # Use only embeddings
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Store one reference embedding per person (average of all their embeddings)
        self.person_names = sorted(list(set(labels)))
        self.reference_embeddings = []
        self.reference_labels = []
        
        for person_name in self.person_names:
            # Get all embeddings for this person
            person_mask = np.array(labels) == person_name
            person_embeddings = features_scaled[person_mask]
            
            # Average them to get reference embedding
            reference_embedding = np.mean(person_embeddings, axis=0)
            
            self.reference_embeddings.append(reference_embedding)
            self.reference_labels.append(person_name)
        
        self.reference_embeddings = np.array(self.reference_embeddings)
        self.reference_labels = np.array(self.reference_labels)
        self.is_trained = True
        
        print(f"✓ Trained cosine similarity detector with {len(self.person_names)} persons")
        print(f"  Similarity threshold: {self.similarity_threshold}")
        
        # Cross-validation if requested
        if use_cross_validation:
            self._evaluate_training(features_scaled, labels)
    
    def _evaluate_training(self, features: np.ndarray, labels: list):
        """Evaluate training using leave-one-out cross-validation"""
        correct = 0
        total = len(labels)
        
        for i in range(total):
            # Leave one out
            test_feature = features[i]
            true_label = labels[i]
            
            # Find most similar reference (excluding samples from same person)
            best_similarity = -1
            best_label = None
            
            for ref_embedding, ref_label in zip(self.reference_embeddings, self.reference_labels):
                similarity = self.cosine_similarity(test_feature, ref_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_label = ref_label
            
            # Check if prediction is correct
            if best_label == true_label and best_similarity >= self.similarity_threshold:
                correct += 1
        
        accuracy = correct / total
        print(f"  Cross-validation accuracy: {accuracy:.4f}")
    
    def predict(self, embeddings: np.ndarray, lbp_features: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Predict person identity using cosine similarity
        
        Args:
            embeddings: Deep learning embeddings
            lbp_features: LBP feature vectors
            
        Returns:
            Tuple of (predicted_labels, similarity_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Combine features if requested
        if self.combine_features:
            features = self.combine_embedding_lbp(embeddings, lbp_features)
        else:
            features = embeddings
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        predictions = []
        similarities = []
        
        for feature in features_scaled:
            # Calculate similarity with all reference embeddings
            best_similarity = -1
            best_label = "Unknown"
            
            for ref_embedding, ref_label in zip(self.reference_embeddings, self.reference_labels):
                similarity = self.cosine_similarity(feature, ref_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_label = ref_label
            
            # Only predict if similarity is above threshold
            if best_similarity < self.similarity_threshold:
                best_label = "Unknown"
                best_similarity = 0.0
            
            predictions.append(best_label)
            similarities.append(best_similarity)
        
        return predictions, np.array(similarities)
    
    def predict_single(self, embedding: np.ndarray, lbp_feature: np.ndarray) -> Tuple[str, float]:
        """
        Predict identity for a single sample
        
        Args:
            embedding: Single embedding vector
            lbp_feature: Single LBP feature vector
            
        Returns:
            Tuple of (predicted_label, similarity_score)
        """
        # Reshape for single sample
        embedding = embedding.reshape(1, -1)
        lbp_feature = lbp_feature.reshape(1, -1)
        
        labels, similarities = self.predict(embedding, lbp_feature)
        
        return labels[0], similarities[0]
    
    def evaluate(self, embeddings: np.ndarray, lbp_features: np.ndarray,
                 true_labels: list) -> dict:
        """
        Evaluate model performance
        
        Args:
            embeddings: Deep learning embeddings
            lbp_features: LBP feature vectors
            true_labels: True person labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        predicted_labels, similarities = self.predict(embeddings, lbp_features)
        
        # Filter out "Unknown" predictions for metrics
        valid_mask = np.array(predicted_labels) != "Unknown"
        
        if valid_mask.sum() > 0:
            filtered_pred = np.array(predicted_labels)[valid_mask]
            filtered_true = np.array(true_labels)[valid_mask]
            
            accuracy = accuracy_score(filtered_true, filtered_pred)
            precision = precision_score(filtered_true, filtered_pred, average='weighted', zero_division=0)
            recall = recall_score(filtered_true, filtered_pred, average='weighted', zero_division=0)
            f1 = f1_score(filtered_true, filtered_pred, average='weighted', zero_division=0)
        else:
            accuracy = precision = recall = f1 = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_similarity': np.mean(similarities),
            'unknown_rate': 1 - (valid_mask.sum() / len(predicted_labels))
        }
    
    def save(self, filepath: str):
        """Save detector to file"""
        save_data = {
            'reference_embeddings': self.reference_embeddings,
            'reference_labels': self.reference_labels,
            'person_names': self.person_names,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'reverse_label_encoder': self.reverse_label_encoder,
            'similarity_threshold': self.similarity_threshold,
            'combine_features': self.combine_features,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, filepath: str):
        """Load detector from file"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.reference_embeddings = save_data['reference_embeddings']
        self.reference_labels = save_data['reference_labels']
        self.person_names = save_data['person_names']
        self.scaler = save_data['scaler']
        self.label_encoder = save_data['label_encoder']
        self.reverse_label_encoder = save_data['reverse_label_encoder']
        self.similarity_threshold = save_data['similarity_threshold']
        self.combine_features = save_data['combine_features']
        self.is_trained = save_data['is_trained']


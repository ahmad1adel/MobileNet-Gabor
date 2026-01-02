"""
Embedding Module - Unmasked Dataset
Generates deep learning embeddings for face recognition
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Lambda
from tensorflow.keras.models import Model
import cv2
from typing import Optional, Tuple


class FaceEmbedder:
    """Generates face embeddings using deep learning"""
    
    def __init__(self, input_size: Tuple[int, int] = (224, 224), 
                 embedding_dim: int = 128, use_pretrained: bool = True):
        """
        Initialize face embedder
        
        Args:
            input_size: Input image size (height, width)
            embedding_dim: Dimension of output embedding
            use_pretrained: Whether to use pretrained weights
        """
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.model = None
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            self._build_model()
    
    def _build_model(self):
        """Build embedding model"""
        # Base model (MobileNetV2 for efficiency)
        base_model = MobileNetV2(
            input_shape=(*self.input_size, 3),
            include_top=False,
            weights='imagenet' if self.use_pretrained else None
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        embeddings = Dense(self.embedding_dim, activation=None)(x)
        
        # Note: Normalization will be done after prediction to avoid Lambda layer issues
        self.model = Model(inputs=base_model.input, outputs=embeddings)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Resize if needed
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Expand dimensions for batch
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from face image
        
        Args:
            image: Face image
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            self._build_model()
        
        # Preprocess image
        processed = self.preprocess_image(image)
        
        # Get embedding
        embedding = self.model.predict(processed, verbose=0)
        embedding = embedding[0]  # Remove batch dimension
        
        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-7)
        
        return embedding
    
    def extract_embeddings_batch(self, images: list) -> np.ndarray:
        """
        Extract embeddings from multiple images
        
        Args:
            images: List of face images
            
        Returns:
            Array of embeddings
        """
        if self.model is None:
            self._build_model()
        
        # Preprocess all images
        processed = np.array([self.preprocess_image(img)[0] for img in images])
        
        # Get embeddings
        embeddings = self.model.predict(processed, verbose=0)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-7)
        
        return embeddings
    
    def fine_tune_model(self, train_data: np.ndarray, train_labels: np.ndarray,
                       val_data: np.ndarray = None, val_labels: np.ndarray = None,
                       epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.0001):
        """
        Fine-tune the embedding model
        
        Args:
            train_data: Training images
            train_labels: Training labels
            val_data: Validation images
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            self._build_model()
        
        # Unfreeze some layers for fine-tuning
        for layer in self.model.layers[-10:]:
            layer.trainable = True
        
        # Add classification head for training
        x = self.model.layers[-1].output
        num_classes = len(np.unique(train_labels))
        classifier = Dense(num_classes, activation='softmax', name='classifier')(x)
        
        train_model = Model(inputs=self.model.input, outputs=classifier)
        
        train_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Prepare data
        train_processed = np.array([self.preprocess_image(img)[0] for img in train_data])
        
        callbacks = []
        if val_data is not None:
            val_processed = np.array([self.preprocess_image(img)[0] for img in val_data])
            callbacks.append(
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            )
        
        # Train
        train_model.fit(
            train_processed, train_labels,
            validation_data=(val_processed, val_labels) if val_data is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Remove classification head to keep embedding model
        # The base layers are already updated
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model is not None:
            # Use .keras format instead of .h5 to avoid deprecation warning
            if filepath.endswith('.h5'):
                filepath = filepath.replace('.h5', '.keras')
            self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        try:
            self.model = keras.models.load_model(filepath)
        except Exception as e:
            # If loading fails (e.g., due to Lambda layer issues), rebuild the model
            print(f"Warning: Could not load embedder model ({e})")
            print("Rebuilding embedder with pretrained weights...")
            self._build_model()
            # Try to load weights if available (skip Lambda layer)
            try:
                # Load weights from the old model, skipping incompatible layers
                old_model = keras.models.load_model(filepath, compile=False)
                # Copy compatible layer weights
                for i, layer in enumerate(self.model.layers):
                    if i < len(old_model.layers):
                        try:
                            old_layer = old_model.layers[i]
                            if layer.name == old_layer.name and hasattr(old_layer, 'get_weights'):
                                if len(layer.get_weights()) == len(old_layer.get_weights()):
                                    layer.set_weights(old_layer.get_weights())
                        except:
                            pass
            except:
                pass


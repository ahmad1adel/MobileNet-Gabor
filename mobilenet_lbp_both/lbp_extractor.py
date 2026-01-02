"""
LBP (Local Binary Pattern) Feature Extractor
Extracts texture features using Local Binary Patterns
"""

import numpy as np
from skimage import feature
from typing import Tuple, Optional


class LBPExtractor:
    """Extracts LBP features from images"""
    
    def __init__(self, num_points: int = 8, radius: int = 1, 
                 method: str = 'uniform', n_bins: int = 256):
        """
        Initialize LBP extractor
        
        Args:
            num_points: Number of circularly symmetric neighbor set points
            radius: Radius of circle (spatial resolution)
            method: Method to use ('default', 'ror', 'uniform', 'var')
            n_bins: Number of bins for histogram
        """
        self.num_points = num_points
        self.radius = radius
        self.method = method
        self.n_bins = n_bins
    
    def extract_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP pattern from image
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            LBP pattern image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        # Compute LBP
        lbp = feature.local_binary_pattern(
            gray, 
            self.num_points, 
            self.radius, 
            method=self.method
        )
        
        return lbp
    
    def extract_lbp_histogram(self, image: np.ndarray, 
                              normalize: bool = True) -> np.ndarray:
        """
        Extract LBP histogram features
        
        Args:
            image: Input image
            normalize: Whether to normalize histogram
            
        Returns:
            LBP histogram feature vector
        """
        # Extract LBP pattern
        lbp = self.extract_lbp(image)
        
        # Calculate histogram
        hist, _ = np.histogram(
            lbp.ravel(), 
            bins=self.n_bins, 
            range=(0, self.n_bins)
        )
        
        # Normalize if requested
        if normalize:
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)  # Avoid division by zero
        
        return hist
    
    def extract_multiscale_lbp(self, image: np.ndarray, 
                               radii: list = [1, 2, 3]) -> np.ndarray:
        """
        Extract multiscale LBP features
        
        Args:
            image: Input image
            radii: List of radii to use
            
        Returns:
            Concatenated multiscale LBP features
        """
        features = []
        
        for radius in radii:
            original_radius = self.radius
            self.radius = radius
            hist = self.extract_lbp_histogram(image)
            features.append(hist)
            self.radius = original_radius
        
        return np.concatenate(features)
    
    def extract_spatial_lbp(self, image: np.ndarray, 
                           grid_size: Tuple[int, int] = (4, 4)) -> np.ndarray:
        """
        Extract spatial LBP features by dividing image into grid
        
        Args:
            image: Input image
            grid_size: Grid dimensions (rows, cols)
            
        Returns:
            Concatenated spatial LBP features
        """
        h, w = image.shape[:2]
        cell_h = h // grid_size[0]
        cell_w = w // grid_size[1]
        
        features = []
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                y1 = i * cell_h
                y2 = (i + 1) * cell_h if i < grid_size[0] - 1 else h
                x1 = j * cell_w
                x2 = (j + 1) * cell_w if j < grid_size[1] - 1 else w
                
                cell = image[y1:y2, x1:x2]
                hist = self.extract_lbp_histogram(cell)
                features.append(hist)
        
        return np.concatenate(features)
    
    def extract_features(self, image: np.ndarray, 
                        use_multiscale: bool = False,
                        use_spatial: bool = False) -> np.ndarray:
        """
        Extract LBP features with various options
        
        Args:
            image: Input image
            use_multiscale: Whether to use multiscale LBP
            use_spatial: Whether to use spatial grid
            
        Returns:
            LBP feature vector
        """
        if use_multiscale:
            return self.extract_multiscale_lbp(image)
        elif use_spatial:
            return self.extract_spatial_lbp(image)
        else:
            return self.extract_lbp_histogram(image)


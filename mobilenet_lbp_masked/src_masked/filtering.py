"""
Filtering Module - Masked Dataset
Applies Gaussian Blur or Median Filter for noise reduction
"""

import cv2
import numpy as np
from typing import Literal


class ImageFilter:
    """Applies various filters to images"""
    
    def __init__(self, filter_type: Literal['gaussian', 'median'] = 'gaussian', 
                 kernel_size: int = 5):
        """
        Initialize filter
        
        Args:
            filter_type: Type of filter to apply
            kernel_size: Size of filter kernel (must be odd)
        """
        self.filter_type = filter_type
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    def apply_gaussian_blur(self, image: np.ndarray, sigma: float = None) -> np.ndarray:
        """
        Apply Gaussian blur to image
        
        Args:
            image: Input image
            sigma: Standard deviation (if None, calculated from kernel_size)
            
        Returns:
            Filtered image
        """
        if sigma is None:
            sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8
        
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), sigma)
    
    def apply_median_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply median filter to image
        
        Args:
            image: Input image
            
        Returns:
            Filtered image
        """
        return cv2.medianBlur(image, self.kernel_size)
    
    def apply_filter(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply the configured filter
        
        Args:
            image: Input image
            **kwargs: Additional filter parameters
            
        Returns:
            Filtered image
        """
        if self.filter_type == 'gaussian':
            sigma = kwargs.get('sigma', None)
            return self.apply_gaussian_blur(image, sigma)
        elif self.filter_type == 'median':
            return self.apply_median_filter(image)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
    
    def apply_adaptive_filter(self, image: np.ndarray, 
                             noise_level: float = 0.1) -> np.ndarray:
        """
        Apply adaptive filtering based on noise level
        
        Args:
            image: Input image
            noise_level: Estimated noise level (0-1)
            
        Returns:
            Filtered image
        """
        # Use Gaussian for low noise, Median for high noise
        if noise_level < 0.3:
            return self.apply_gaussian_blur(image)
        else:
            return self.apply_median_filter(image)


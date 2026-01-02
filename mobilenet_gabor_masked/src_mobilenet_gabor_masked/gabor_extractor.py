import numpy as np
import cv2
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

class GaborExtractor:
    """Texture feature extraction using a bank of Gabor filters"""
    
    def __init__(self, orientations=4, scales=2):
        """
        Initialize Gabor bank
        orientations: number of orientations (default 4: 0, 45, 90, 135)
        scales: number of scales (sigma values)
        """
        self.kernels = []
        for theta in range(orientations):
            theta = theta / orientations * np.pi
            for sigma in (1.0, 2.0)[:scales]:  # Using scales 1.0 and 2.0
                kernel = np.real(gabor_kernel(0.1, theta=theta, sigma_x=sigma, sigma_y=sigma))
                self.kernels.append(kernel)
                
    def extract(self, image):
        """
        Extract Gabor features from image
        Returns a 1D feature vector [mean1, std1, mean2, std2, ...]
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        features = []
        for kernel in self.kernels:
            filtered = ndi.convolve(gray, kernel, mode='wrap')
            features.extend([filtered.mean(), filtered.std()])
            
        return np.array(features)

    def extract_batch(self, images):
        """Extract features for a batch of images"""
        batch_features = []
        for img in images:
            batch_features.append(self.extract(img))
        return np.array(batch_features)

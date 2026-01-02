"""
Image Preprocessing Module
Handles background removal and image resizing to uniform dimensions
"""

import cv2
import numpy as np
from PIL import Image
import os
from rembg import remove
from typing import Tuple, Optional


class ImagePreprocessor:
    """Handles image preprocessing including background removal and resizing"""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256), remove_bg: bool = True):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image dimensions (width, height)
            remove_bg: Whether to remove background
        """
        self.target_size = target_size
        self.remove_bg = remove_bg
    
    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background from image using rembg
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Image with background removed
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        # Remove background
        output = remove(pil_image)
        
        # Convert back to numpy array
        output_array = np.array(output)
        
        # Convert RGBA to BGR if needed
        if output_array.shape[2] == 4:
            # Extract alpha channel for mask
            alpha = output_array[:, :, 3]
            rgb = output_array[:, :, :3]
            
            # Create white background
            white_bg = np.ones_like(rgb) * 255
            
            # Blend with white background using alpha
            result = (rgb * (alpha[:, :, np.newaxis] / 255.0) + 
                     white_bg * (1 - alpha[:, :, np.newaxis] / 255.0))
            
            return cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        return cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
    
    def resize_image(self, image: np.ndarray, maintain_aspect: bool = False) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            maintain_aspect: If True, maintain aspect ratio and pad
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            h, w = image.shape[:2]
            target_w, target_h = self.target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create padded image
            padded = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
            
            # Calculate padding offsets
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            # Place resized image in center
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return padded
        else:
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1]
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def preprocess(self, image_path: Optional[str] = None, 
                   image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to image file
            image: Image as numpy array (if image_path not provided)
            
        Returns:
            Preprocessed image
        """
        # Load image if path provided
        if image_path:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
        
        if image is None:
            raise ValueError("Either image_path or image must be provided")
        
        # Remove background if enabled
        if self.remove_bg:
            image = self.remove_background(image)
        
        # Resize to target size
        image = self.resize_image(image, maintain_aspect=True)
        
        return image
    
    def preprocess_batch(self, image_paths: list, output_dir: str) -> list:
        """
        Preprocess multiple images
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save processed images
            
        Returns:
            List of processed image paths
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_paths = []
        
        for img_path in image_paths:
            try:
                processed_img = self.preprocess(image_path=img_path)
                
                # Save processed image
                filename = os.path.basename(img_path)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, processed_img)
                processed_paths.append(output_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return processed_paths


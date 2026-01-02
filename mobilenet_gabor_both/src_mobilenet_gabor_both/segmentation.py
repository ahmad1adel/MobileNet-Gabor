"""
Segmentation Module
Handles face detection and mask detection
"""

import cv2
import numpy as np
from mtcnn import MTCNN
import mediapipe as mp
from typing import Tuple, List, Optional, Dict

try:
    # YOLO from ultralytics (for face/person detection)
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class FaceSegmenter:
    """Handles face detection and segmentation"""
    
    def __init__(self, detector_type: str = 'yolo', yolo_weights: str = 'yolov8n.pt'):
        """
        Initialize face segmenter
        
        Args:
            detector_type: 'yolo', 'mtcnn', or 'mediapipe'
            yolo_weights: YOLO weights path/name (for 'yolo' mode)
        """
        self.detector_type = detector_type
        self.yolo_model = None
        
        if detector_type == 'mtcnn':
            self.detector = MTCNN()
            self.mp_face = None
        elif detector_type == 'mediapipe':
            self.detector = None
            self.mp_face = mp.solutions.face_detection
            self.face_detection = self.mp_face.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        elif detector_type == 'yolo':
            if YOLO is None:
                raise ImportError(
                    "ultralytics is not installed. Install it with 'pip install ultralytics' "
                    "or switch detector_type to 'mtcnn' or 'mediapipe'."
                )
            # Load YOLO model (generic, class 0 = person). For face‑specific YOLO, change weights path.
            self.yolo_model = YOLO(yolo_weights)
            self.detector = None
            self.mp_face = None
        else:
            raise ValueError("detector_type must be 'yolo', 'mtcnn' or 'mediapipe'")
    
    def detect_face_mtcnn(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces using MTCNN
        
        Args:
            image: Input image
            
        Returns:
            List of face detections with bounding boxes
        """
        # Convert BGR to RGB for MTCNN
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb_image)
        return detections
    
    def detect_face_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces using MediaPipe
        
        Args:
            image: Input image
            
        Returns:
            List of face detections with bounding boxes
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        detections = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                detections.append({
                    'box': [x, y, width, height],
                    'confidence': detection.score[0]
                })
        
        return detections
    
    def detect_face_yolo(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces using YOLO (via ultralytics)

        Note: Here we use YOLO's 'person' class (class 0) and take the upper part
        of the bounding box as a proxy for the face. For best results you can
        plug in a YOLO model trained specifically for face detection.
        
        Args:
            image: Input image (BGR)

        Returns:
            List of detections with bounding boxes
        """
        if self.yolo_model is None:
            return []

        h, w = image.shape[:2]

        # YOLO expects RGB or BGR; ultralytics handles numpy arrays directly
        results = self.yolo_model.predict(source=image, verbose=False)

        detections = []
        if not results:
            return detections

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # For COCO weights, class 0 is 'person'
                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))

                width = max(0, x2 - x1)
                height = max(0, y2 - y1)

                if width <= 0 or height <= 0:
                    continue

                detections.append(
                    {
                        "box": [x1, y1, width, height],
                        "confidence": conf,
                    }
                )

        return detections
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image
        
        Args:
            image: Input image
            
        Returns:
            List of face detections
        """
        if self.detector_type == 'mtcnn':
            return self.detect_face_mtcnn(image)
        elif self.detector_type == 'mediapipe':
            return self.detect_face_mediapipe(image)
        elif self.detector_type == 'yolo':
            return self.detect_face_yolo(image)
        else:
            return []
    
    def extract_face_region(self, image: np.ndarray, bbox: List[int], 
                           padding: float = 0.2) -> np.ndarray:
        """
        Extract face region from image
        
        Args:
            image: Input image
            bbox: Bounding box [x, y, width, height]
            padding: Padding factor around face
            
        Returns:
            Cropped face region
        """
        h, w = image.shape[:2]
        x, y, width, height = bbox
        
        # Validate bounding box
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid bounding box dimensions: width={width}, height={height}")
        
        # Add padding
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        
        # Calculate crop coordinates
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + width + pad_x)
        y2 = min(h, y + height + pad_y)
        
        # Validate crop coordinates
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # Extract face
        face = image[y1:y2, x1:x2]
        
        # Validate extracted face
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            raise ValueError(f"Extracted face is empty: shape={face.shape}")
        
        # Ensure minimum size
        min_size = 32
        if face.shape[0] < min_size or face.shape[1] < min_size:
            # Resize if too small
            scale = max(min_size / face.shape[0], min_size / face.shape[1])
            new_h = max(min_size, int(face.shape[0] * scale))
            new_w = max(min_size, int(face.shape[1] * scale))
            face = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return face
    
    def detect_mask(self, face_image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if face is wearing a mask
        Simple heuristic-based approach (can be replaced with ML model)
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Tuple of (is_masked, confidence)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Get face dimensions
        h, w = gray.shape
        lower_face_region = gray[int(h * 0.5):, :]
        
        # Calculate variance in lower face region
        # Masked faces typically have lower variance in mouth/nose area
        variance = np.var(lower_face_region)
        
        # Threshold-based detection (can be improved with trained model)
        threshold = 500  # Adjust based on your data
        is_masked = variance < threshold
        confidence = min(1.0, abs(variance - threshold) / threshold)
        
        return is_masked, confidence
    
    def segment_face(self, image: np.ndarray, return_mask_info: bool = True) -> Dict:
        """
        Complete face segmentation pipeline
        
        Args:
            image: Input image
            return_mask_info: Whether to detect mask
            
        Returns:
            Dictionary with face regions and metadata
        """
        # Detect faces
        detections = self.detect_faces(image)
        
        if not detections:
            return {
                'faces': [],
                'num_faces': 0
            }
        
        faces_data = []
        for i, detection in enumerate(detections):
            try:
                # Extract bounding box
                if self.detector_type == 'mtcnn':
                    bbox = detection['box']
                else:
                    bbox = detection['box']
                
                # Validate bounding box
                if len(bbox) < 4:
                    continue
                if bbox[2] <= 0 or bbox[3] <= 0:  # width or height <= 0
                    continue
                
                # Extract face region
                try:
                    face_region = self.extract_face_region(image, bbox)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not extract face region: {e}")
                    continue
                
                # Validate face region
                if face_region is None or face_region.size == 0:
                    continue
                if len(face_region.shape) != 3 or face_region.shape[2] != 3:
                    continue
                if face_region.shape[0] < 32 or face_region.shape[1] < 32:
                    continue
                
                face_info = {
                    'face_image': face_region,
                    'bbox': bbox,
                    'confidence': detection.get('confidence', 1.0)
                }
                
                # Detect mask if requested
                if return_mask_info:
                    try:
                        is_masked, mask_confidence = self.detect_mask(face_region)
                        face_info['is_masked'] = is_masked
                        face_info['mask_confidence'] = mask_confidence
                    except Exception as e:
                        print(f"Warning: Mask detection failed: {e}")
                        face_info['is_masked'] = False
                        face_info['mask_confidence'] = 0.0
                
                faces_data.append(face_info)
            except Exception as e:
                print(f"Warning: Error processing detection {i}: {e}")
                continue
        
        return {
            'faces': faces_data,
            'num_faces': len(faces_data)
        }


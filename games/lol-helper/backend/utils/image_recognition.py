import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class ImageRecognition:
    
    def __init__(self):
        self.yolo = None
        self.ocr = None
    
    def load_yolo_model(self, model_path: str = "yolov8n.pt"):
        pass
    
    def load_ocr_model(self):
        pass
    
    def recognize_text(self, image: np.ndarray) -> List[str]:
        pass
    
    def recognize_health(self, image: np.ndarray) -> Optional[float]:
        pass
    
    def recognize_gold(self, image: np.ndarray) -> Optional[int]:
        pass
    
    def detect_objects(self, image: np.ndarray, 
                     classes: List[str] = None) -> List[Dict]:
        pass
    
    def track_positions(self, detections: List[Dict]) -> List[Tuple[int, int]]:
        pass
    
    def classify_hero(self, image: np.ndarray) -> Optional[str]:
        pass

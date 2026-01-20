import cv2
import numpy as np
from typing import Optional, Tuple
import time


class ScreenCapture:
    
    def __init__(self, window_name: str = "League of Legends"):
        self.window_name = window_name
        self.window = None
    
    def find_window(self) -> bool:
        pass
    
    def capture_full_screen(self) -> np.ndarray:
        pass
    
    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        pass
    
    def capture_game_window(self) -> np.ndarray:
        pass
    
    def resize_capture(self, image: np.ndarray, 
                      size: Tuple[int, int] = (320, 180)) -> np.ndarray:
        return cv2.resize(image, size)

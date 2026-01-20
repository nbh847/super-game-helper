import numpy as np
from typing import Dict, Any


class StateExtractor:
    
    def __init__(self):
        pass
    
    def extract(self, frame: Dict[str, Any]) -> np.ndarray:
        pass
    
    def normalize(self, state: np.ndarray) -> np.ndarray:
        pass
    
    def extract_hero_state(self, frame: Dict[str, Any]) -> Dict[str, float]:
        pass
    
    def extract_enemies_state(self, frame: Dict[str, Any]) -> np.ndarray:
        pass
    
    def extract_minions_state(self, frame: Dict[str, Any]) -> np.ndarray:
        pass
    
    def extract_tower_state(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        pass

import numpy as np
from typing import Dict, Any, List


class ActionExtractor:
    
    def __init__(self):
        pass
    
    def extract(self, frame_data: Dict[str, Any]) -> int:
        pass
    
    def encode(self, action: Dict[str, Any]) -> int:
        pass
    
    def decode(self, action_id: int) -> Dict[str, Any]:
        pass
    
    def get_action_space(self) -> int:
        pass
    
    def is_valid_action(self, action: Dict[str, Any]) -> bool:
        pass

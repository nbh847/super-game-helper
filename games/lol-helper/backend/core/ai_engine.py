import torch
from typing import Dict, Optional


class AIEngine:
    
    def __init__(self, model_path: str, hero_type: str):
        self.model = self._load_model(model_path)
        self.hero_type = hero_type
    
    def _load_model(self, model_path: str) -> Optional[torch.nn.Module]:
        pass
    
    def decide_action(self, game_state: Dict) -> Dict:
        pass
    
    def update_policy(self, reward: float) -> None:
        pass
    
    def set_model(self, model: torch.nn.Module) -> None:
        self.model = model
    
    def get_model(self) -> Optional[torch.nn.Module]:
        return self.model

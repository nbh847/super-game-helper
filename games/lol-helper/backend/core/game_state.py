import numpy as np
from typing import Dict, Optional, List, Tuple


class GameState:
    
    def __init__(self):
        self.hero_position: Optional[Tuple[int, int]] = None
        self.hero_health: Optional[float] = None
        self.hero_mana: Optional[float] = None
        self.skills_cooldown: List[bool] = []
        self.enemy_positions: List[Tuple[int, int]] = []
        self.minion_positions: List[Tuple[int, int]] = []
        self.tower_position: Optional[Tuple[int, int]] = None
        self.gold: Optional[int] = None
        self.level: Optional[int] = None
        self.kda: Dict[str, int] = {"kills": 0, "deaths": 0, "assists": 0}
    
    def update_from_screen(self, screenshot: np.ndarray) -> None:
        pass
    
    def to_tensor(self) -> np.ndarray:
        pass
    
    def get_hero_position(self) -> Optional[Tuple[int, int]]:
        return self.hero_position
    
    def get_health(self) -> Optional[float]:
        return self.hero_health
    
    def is_in_danger(self) -> bool:
        pass
    
    def get_nearest_enemy(self) -> Optional[Tuple[int, int]]:
        if not self.enemy_positions:
            return None
        return min(self.enemy_positions, key=lambda p: 
                  self._distance(self.hero_position, p))
    
    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

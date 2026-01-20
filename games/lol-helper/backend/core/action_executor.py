from typing import Optional, Tuple
import time
import random


class ActionExecutor:
    
    def __init__(self, human_behavior=None):
        self.human_behavior = human_behavior
    
    def move_to(self, target_pos: Tuple[int, int]) -> None:
        pass
    
    def attack_target(self, target_pos: Tuple[int, int]) -> None:
        pass
    
    def cast_skill(self, skill_key: str, 
                   target_pos: Optional[Tuple[int, int]] = None) -> None:
        pass
    
    def use_heal(self) -> None:
        pass
    
    def right_click(self, pos: Tuple[int, int]) -> None:
        pass
    
    def press_key(self, key: str) -> None:
        pass

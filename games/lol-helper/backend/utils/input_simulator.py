import pyautogui
from typing import Tuple, List
import time


class InputSimulator:
    
    def __init__(self):
        pyautogui.FAILSAFE = True
    
    def move_mouse(self, x: int, y: int, duration: Optional[float] = None) -> None:
        pass
    
    def click(self, x: Optional[int] = None, y: Optional[int] = None, 
              button: str = 'left') -> None:
        pass
    
    def press_key(self, key: str) -> None:
        pass
    
    def hotkey(self, *keys) -> None:
        pass
    
    def drag_to(self, x: int, y: int, duration: Optional[float] = None) -> None:
        pass
    
    def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        pass
    
    def scroll(self, clicks: int) -> None:
        pass

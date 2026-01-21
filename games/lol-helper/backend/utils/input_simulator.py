"""
输入模拟器模块
集成控制线程，提供统一的输入接口
"""

from typing import Optional
import time
import pyautogui

from .control_thread import ControlThread


class InputSimulator:
    def __init__(self):
        self.control_thread = ControlThread()
        self.control_thread.start()
        print("[输入模拟器] 已初始化")
    
    def start(self):
        """启动控制线程"""
        if not self.control_thread.control_running:
            self.control_thread.start()
    
    def stop(self):
        """停止控制线程"""
        self.control_thread.stop()
    
    def move_mouse(self, x: int, y: int, duration: Optional[float] = None) -> None:
        """移动鼠标到指定位置"""
        self.control_thread.move_mouse(x, y)
    
    def click(self, x: Optional[int] = None, y: Optional[int] = None, 
              button: str = 'left') -> None:
        """点击鼠标（左键/右键）"""
        self.control_thread.click(x, y, button)
    
    def press_key(self, key: str, duration: float = 0.15) -> None:
        """按下键盘按键"""
        self.control_thread.press_key(key, duration)
    
    def hotkey(self, *keys) -> None:
        """快捷键组合"""
        for key in keys:
            self.control_thread.press_key(key, 0.05)
    
    def drag_to(self, x: int, y: int, duration: Optional[float] = None) -> None:
        """拖拽到指定位置"""
        self.control_thread.drag_to(x, y, duration or 1)
    
    def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """右键点击"""
        self.control_thread.right_click(x, y)
    
    def scroll(self, clicks: int) -> None:
        """鼠标滚轮"""
        self.control_thread.scroll(clicks)


if __name__ == "__main__":
    print("开始测试输入模拟器...")
    
    sim = InputSimulator()
    
    print("\n测试1：移动鼠标")
    for i in range(5):
        sim.move_mouse(500 + i*30, 500 + i*30)
        time.sleep(0.1)
    
    print("\n测试2：点击")
    time.sleep(0.5)
    sim.click(650, 650)
    
    print("\n测试3：按键")
    time.sleep(0.5)
    sim.press_key('w', 0.2)
    
    print("\n测试4：右键点击")
    time.sleep(0.5)
    sim.right_click(600, 600)
    
    print("\n等待队列清空...")
    time.sleep(2)
    
    sim.stop()
    
    print("\n输入模拟器测试完成！")

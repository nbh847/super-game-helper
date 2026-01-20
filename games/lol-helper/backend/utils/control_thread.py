"""
控制线程模块
独立线程处理鼠标/键盘输入，避免阻塞主循环
"""

import time
import threading
import platform

import pyautogui

# pydirectinput仅在Windows上可用
IS_WINDOWS = platform.system() == 'Windows'
if IS_WINDOWS:
    import pydirectinput


class ControlThread:
    def __init__(self, queue_limit=10):
        self.control_queue = []
        self.control_thread = None
        self.control_running = False
        self.queue_limit = queue_limit
        self.exec_count = 0
        self.IS_WINDOWS = IS_WINDOWS
        
        if self.IS_WINDOWS:
            pydirectinput.PAUSE = 0.01
        pyautogui.PAUSE = 0.01
        pyautogui.FAILSAFE = False
    
    def start(self):
        self.control_running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        print("[控制线程] 已启动")
    
    def stop(self):
        self.control_running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        print("[控制线程] 已停止")
    
    def _control_loop(self):
        print("[控制线程] 开始运行...")
        
        while self.control_running:
            try:
                if self.control_queue:
                    cmd = self.control_queue.pop(0)
                    self._execute_command(cmd)
                    self.exec_count += 1
                    if self.exec_count % 100 == 0:
                        print(f"[控制线程] 已执行 {self.exec_count} 个指令，队列:{len(self.control_queue)}")
                
                time.sleep(0.005)
            
            except Exception as e:
                print(f"[控制线程错误] {e}")
                time.sleep(0.1)
    
    def _execute_command(self, cmd):
        cmd_type = cmd['type']
        
        if cmd_type == 'move_mouse':
            pyautogui.moveTo(cmd['x'], cmd['y'])
        
        elif cmd_type == 'click':
            if 'x' in cmd and 'y' in cmd:
                pyautogui.click(cmd['x'], cmd['y'], button=cmd.get('button', 'left'))
            else:
                pyautogui.click(button=cmd.get('button', 'left'))
        
        elif cmd_type == 'press_key':
            key = cmd['key']
            duration = cmd.get('duration', 0.15)
            if self.IS_WINDOWS:
                pydirectinput.keyDown(key)
                time.sleep(duration)
                pydirectinput.keyUp(key)
            else:
                pyautogui.keyDown(key)
                time.sleep(duration)
                pyautogui.keyUp(key)
        
        elif cmd_type == 'right_click':
            if 'x' in cmd and 'y' in cmd:
                pyautogui.rightClick(cmd['x'], cmd['y'])
            else:
                pyautogui.rightClick()
        
        elif cmd_type == 'drag_to':
            pyautogui.dragTo(cmd['x'], cmd['y'], duration=cmd.get('duration', 1))
        
        elif cmd_type == 'scroll':
            pyautogui.scroll(cmd['clicks'])
    
    def add_command(self, cmd_type, **kwargs):
        if len(self.control_queue) >= self.queue_limit:
            return False
        cmd = {'type': cmd_type, **kwargs}
        self.control_queue.append(cmd)
        return True
    
    def move_mouse(self, x, y):
        self.add_command('move_mouse', x=x, y=y)
    
    def click(self, x=None, y=None, button='left'):
        cmd = {'button': button}
        if x is not None:
            cmd['x'] = x
        if y is not None:
            cmd['y'] = y
        self.add_command('click', **cmd)
    
    def press_key(self, key, duration=0.15):
        self.add_command('press_key', key=key, duration=duration)
    
    def right_click(self, x=None, y=None):
        cmd = {}
        if x is not None:
            cmd['x'] = x
        if y is not None:
            cmd['y'] = y
        self.add_command('right_click', **cmd)
    
    def drag_to(self, x, y, duration=1):
        self.add_command('drag_to', x=x, y=y, duration=duration)
    
    def scroll(self, clicks):
        self.add_command('scroll', clicks=clicks)


if __name__ == "__main__":
    print("开始测试控制线程...")
    
    ct = ControlThread()
    ct.start()
    
    print("\n测试1：移动鼠标")
    for i in range(10):
        ct.move_mouse(500 + i*20, 500 + i*10)
        time.sleep(0.05)
    
    print("\n测试2：点击")
    time.sleep(0.5)
    ct.click(700, 550)
    
    print("\n测试3：按键")
    time.sleep(0.5)
    ct.press_key('a', 0.2)
    
    print("\n等待队列清空...")
    time.sleep(2)
    
    ct.stop()
    
    print("\n控制线程测试完成！")

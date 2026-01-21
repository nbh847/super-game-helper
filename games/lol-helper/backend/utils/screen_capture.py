"""
屏幕截取模块
使用mss库实现跨平台的高性能屏幕截图
"""

import mss
import numpy as np
import cv2
from typing import Optional, Tuple
import time
from threading import Thread


class ScreenCapture:
    """
    屏幕截取器
    
    使用mss库实现跨平台的高性能屏幕截图
    支持全屏/区域截图，自动缩放
    """
    
    def __init__(self, window_name: str = "League of Legends"):
        """
        初始化屏幕截取器
        
        Args:
            window_name: 窗口名称（用于Windows）
        """
        self.window_name = window_name
        self.window = None
        self.monitor = None
        
        # 性能统计
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        
        # 初始化mss
        self.sct = mss.mss()
        
        # 尝试找到窗口
        self.find_window()
    
    def find_window(self) -> bool:
        """
        查找游戏窗口
        
        Returns:
            是否找到窗口
        """
        # mss会自动查找所有显示器
        monitors = self.sct.monitors
        
        # 选择主显示器（通常是第一个非"global"的显示器）
        if len(monitors) > 1:
            self.monitor = monitors[1]  # 主显示器
            print(f"[ScreenCapture] 使用主显示器: {self.monitor}")
            return True
        else:
            print("[ScreenCapture] 未找到显示器")
            return False
    
    def capture_full_screen(self) -> np.ndarray:
        """
        截取全屏
        
        Returns:
            截图数组 (BGR格式)
        """
        if not self.monitor:
            self.find_window()
        
        # 截取主显示器
        screenshot = self.sct.grab(self.monitor)
        
        # 转换为numpy数组
        img = np.array(screenshot)
        
        # mss返回的是BGRA格式，转换为BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # 更新FPS统计
        self._update_fps()
        
        return img
    
    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        截取指定区域
        
        Args:
            region: (left, top, width, height)
            
        Returns:
            截图数组 (BGR格式)
        """
        left, top, width, height = region
        
        # 构建mss格式的区域
        capture_region = {
            "left": left,
            "top": top,
            "width": width,
            "height": height
        }
        
        # 截取区域
        screenshot = self.sct.grab(capture_region)
        
        # 转换为numpy数组
        img = np.array(screenshot)
        
        # mss返回的是BGRA格式，转换为BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # 更新FPS统计
        self._update_fps()
        
        return img
    
    def capture_game_window(self) -> Optional[np.ndarray]:
        """
        截取游戏窗口
        
        Returns:
            截图数组 (BGR格式)，未找到窗口返回None
        """
        if not self.monitor:
            if not self.find_window():
                return None
        
        return self.capture_full_screen()
    
    def resize_capture(self, image: np.ndarray, 
                      size: Tuple[int, int] = (320, 180)) -> np.ndarray:
        """
        缩放截图到目标尺寸
        
        Args:
            image: 输入图像
            size: 目标尺寸 (width, height)
            
        Returns:
            缩放后的图像
        """
        return cv2.resize(image, size)
    
    def _update_fps(self):
        """
        更新FPS统计
        """
        self.frame_count += 1
        current_time = time.time()
        
        # 每秒更新一次FPS
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
    
    def get_fps(self) -> float:
        """
        获取当前FPS
        
        Returns:
            FPS值
        """
        return self.fps
    
    def benchmark(self, duration: int = 10) -> dict:
        """
        性能测试
        
        Args:
            duration: 测试时长（秒）
            
        Returns:
            性能统计字典
        """
        print(f"[ScreenCapture] 开始性能测试，时长: {duration}秒")
        
        frame_times = []
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            # 测量单帧截取时间
            frame_start = time.time()
            self.capture_full_screen()
            frame_time = time.time() - frame_start
            
            frame_times.append(frame_time)
            frame_count += 1
            
            # 避免CPU占用过高
            time.sleep(0.01)
        
        # 计算统计数据
        avg_time = np.mean(frame_times)
        min_time = np.min(frame_times)
        max_time = np.max(frame_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        result = {
            "total_frames": frame_count,
            "duration": duration,
            "avg_fps": avg_fps,
            "avg_frame_time": avg_time * 1000,  # 毫秒
            "min_frame_time": min_time * 1000,  # 毫秒
            "max_frame_time": max_time * 1000,  # 毫秒
            "fps_percentiles": {
                "50th": np.percentile(frame_times, 50),
                "95th": np.percentile(frame_times, 95),
                "99th": np.percentile(frame_times, 99)
            }
        }
        
        print(f"[ScreenCapture] 性能测试完成:")
        print(f"  总帧数: {frame_count}")
        print(f"  平均FPS: {avg_fps:.2f}")
        print(f"  平均帧时间: {avg_time*1000:.2f}ms")
        print(f"  最小帧时间: {min_time*1000:.2f}ms")
        print(f"  最大帧时间: {max_time*1000:.2f}ms")
        
        return result


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("屏幕截取模块测试")
    print("=" * 60)
    
    # 创建截取器
    capture = ScreenCapture()
    
    # 测试截取
    print("\n[1/3] 测试全屏截取...")
    screenshot = capture.capture_full_screen()
    print(f"✓ 截图形状: {screenshot.shape}")
    print(f"✓ 数据类型: {screenshot.dtype}")
    
    # 测试缩放
    print("\n[2/3] 测试图像缩放...")
    resized = capture.resize_capture(screenshot, size=(320, 180))
    print(f"✓ 缩放后形状: {resized.shape}")
    
    # 性能测试
    print("\n[3/3] 性能测试（5秒）...")
    result = capture.benchmark(duration=5)
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)

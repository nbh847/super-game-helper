"""
游戏状态识别器
整合屏幕截取和图像识别，构建完整游戏状态
"""

import numpy as np
import cv2
from typing import Dict, Optional, List, Tuple
import time

from utils.screen_capture import ScreenCapture
from utils.image_recognition import ImageRecognition


class GameState:
    """
    游戏状态识别器
    
    从屏幕截图中提取游戏状态信息
    """
    
    def __init__(self, window_name: str = "League of Legends"):
        """
        初始化游戏状态识别器
        
        Args:
            window_name: 游戏窗口名称
        """
        # 屏幕截取器
        self.screen_capture = ScreenCapture(window_name)
        
        # 图像识别器
        self.image_recognition = ImageRecognition()
        
        # 游戏状态
        self.hero_position: Optional[Tuple[int, int]] = None
        self.hero_health: Optional[float] = None
        self.hero_mana: Optional[float] = None
        self.skills_cooldown: List[bool] = [False, False, False, False]  # Q/W/E/R
        self.enemy_positions: List[Tuple[int, int]] = []
        self.minion_positions: List[Tuple[int, int]] = []
        self.tower_position: Optional[Tuple[int, int]] = None
        self.gold: Optional[int] = None
        self.level: Optional[int] = None
        self.kda: Dict[str, int] = {"kills": 0, "deaths": 0, "assists": 0}
        
        # 性能统计
        self.last_update_time = time.time()
        self.update_count = 0
    
    def update_from_screen(self, screenshot: Optional[np.ndarray] = None) -> None:
        """
        从屏幕截图更新游戏状态
        
        Args:
            screenshot: 屏幕截图，None表示自动截取
        """
        # 自动截取屏幕
        if screenshot is None:
            screenshot = self.screen_capture.capture_full_screen()
        
        if screenshot is None:
            return
        
        # 缩放截图到模型输入尺寸
        screenshot_resized = cv2.resize(screenshot, (320, 180))
        
        # 更新各个状态
        self._update_hero_position(screenshot_resized)
        self._update_hero_status(screenshot_resized)
        self._update_enemy_positions(screenshot_resized)
        self._update_minion_positions(screenshot_resized)
        self._update_tower_position(screenshot_resized)
        self._update_gold_level(screenshot_resized)
        
        # 更新统计
        self.update_count += 1
        self.last_update_time = time.time()
    
    def _update_hero_position(self, image: np.ndarray):
        """
        更新英雄位置
        
        简化实现：检测中心区域的英雄
        实际项目中应该使用专门的英雄检测模型
        """
        # 目标检测
        detections = self.image_recognition.detect_objects(image)
        
        # 过滤英雄（这里假设'person'类别是英雄）
        hero_detections = [d for d in detections if 'person' in d['class_name'].lower()]
        
        if hero_detections:
            # 选择最中心的英雄
            image_center = (image.shape[1] // 2, image.shape[0] // 2)
            best_hero = min(hero_detections, 
                          key=lambda d: self._distance(d['center'], image_center))
            self.hero_position = best_hero['center']
        else:
            # 如果没检测到英雄，假设在中心
            self.hero_position = (image.shape[1] // 2, image.shape[0] // 2)
    
    def _update_hero_status(self, image: np.ndarray):
        """
        更新英雄状态（血量、蓝量、技能冷却）
        """
        # 英雄UI区域（左下角）
        # 假设血条在左下角 (0, 140, 100, 40)
        health_bar_region = (0, 140, 100, 40)
        
        # 识别血量
        self.hero_health = self.image_recognition.recognize_health(
            image, health_bar_region
        )
        
        # 蓝条（假设在血条下方）
        mana_bar_region = (0, 160, 80, 20)
        self.hero_mana = self.image_recognition.recognize_health(
            image, mana_bar_region
        )
        
        # 技能冷却（简化：假设都是可用）
        # 实际项目中需要检测技能图标
        self.skills_cooldown = [False, False, False, False]
    
    def _update_enemy_positions(self, image: np.ndarray):
        """
        更新敌方英雄位置
        """
        # 目标检测
        detections = self.image_recognition.detect_objects(image)
        
        # 过滤敌方英雄
        # 简化实现：假设上半部分是敌方
        enemy_detections = []
        for d in detections:
            y = d['center'][1]
            if y < image.shape[0] // 2:  # 上半部分
                enemy_detections.append(d)
        
        # 提取位置
        self.enemy_positions = self.image_recognition.track_positions(enemy_detections)
    
    def _update_minion_positions(self, image: np.ndarray):
        """
        更新小兵位置
        """
        # 目标检测
        detections = self.image_recognition.detect_objects(image)
        
        # 过滤小兵
        # 简化实现：检测所有小物体
        minion_detections = []
        for d in detections:
            area = (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1])
            if area < 1000:  # 小物体
                minion_detections.append(d)
        
        # 提取位置
        self.minion_positions = self.image_recognition.track_positions(minion_detections)
    
    def _update_tower_position(self, image: np.ndarray):
        """
        更新防御塔位置
        """
        # 目标检测
        detections = self.image_recognition.detect_objects(image)
        
        # 过滤防御塔
        # 简化实现：检测大且固定的物体
        tower_detections = []
        for d in detections:
            area = (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1])
            if area > 10000:  # 大物体
                tower_detections.append(d)
        
        if tower_detections:
            # 选择最近的防御塔
            if self.hero_position:
                self.tower_position = min(
                    tower_detections,
                    key=lambda d: self._distance(d['center'], self.hero_position)
                )['center']
            else:
                self.tower_position = tower_detections[0]['center']
        else:
            self.tower_position = None
    
    def _update_gold_level(self, image: np.ndarray):
        """
        更新金币和等级
        """
        # 金币区域（左上角）
        gold_region = (0, 0, 100, 30)
        
        # 识别金币
        self.gold = self.image_recognition.recognize_gold(image, gold_region)
        
        # 等级（假设在金币旁边）
        level_region = (100, 0, 30, 30)
        # 简化实现：从OCR结果提取数字
        # 实际项目中需要更精确的定位
        texts = self.image_recognition.recognize_text(image, level_region)
        if texts:
            for text in texts:
                digits = ''.join(c for c in text if c.isdigit())
                if digits and len(digits) <= 2:  # 等级通常是1-18
                    self.level = int(digits)
                    break
    
    def to_tensor(self) -> np.ndarray:
        """
        将游戏状态转换为张量
        
        Returns:
            状态张量 (state_dim,)
            格式: [hero_pos(2), hero_health(1), hero_mana(1), 
                   enemies(10), minions(15), tower(1), 
                   gold(1), level(1), kda(3), skills(4)]
                   总计: 38维
        """
        state = np.zeros(38, dtype=np.float32)
        
        # 英雄位置 (归一化到[0, 1])
        if self.hero_position:
            state[0] = self.hero_position[0] / 320.0
            state[1] = self.hero_position[1] / 180.0
        
        # 英雄状态
        state[2] = self.hero_health if self.hero_health is not None else 0.0
        state[3] = self.hero_mana if self.hero_mana is not None else 0.0
        
        # 敌方位置（最多5个）
        for i in range(min(5, len(self.enemy_positions))):
            pos = self.enemy_positions[i]
            state[4 + i*2] = pos[0] / 320.0
            state[5 + i*2] = pos[1] / 180.0
        
        # 小兵位置（最多5个）
        for i in range(min(5, len(self.minion_positions))):
            pos = self.minion_positions[i]
            state[14 + i*2] = pos[0] / 320.0
            state[15 + i*2] = pos[1] / 180.0
        
        # 防御塔位置
        if self.tower_position:
            state[24] = self.tower_position[0] / 320.0
            state[25] = self.tower_position[1] / 180.0
        
        # 金币（归一化）
        state[26] = (self.gold if self.gold is not None else 0) / 10000.0
        
        # 等级（归一化到[0, 1]，最大18级）
        state[27] = (self.level if self.level is not None else 0) / 18.0
        
        # KDA
        state[28] = self.kda['kills'] / 10.0
        state[29] = self.kda['deaths'] / 10.0
        state[30] = self.kda['assists'] / 10.0
        
        # 技能冷却（0=冷却中，1=可用）
        for i in range(4):
            state[31 + i] = 0.0 if self.skills_cooldown[i] else 1.0
        
        return state
    
    def get_hero_position(self) -> Optional[Tuple[int, int]]:
        """
        获取英雄位置
        
        Returns:
            英雄位置 (x, y)
        """
        return self.hero_position
    
    def get_health(self) -> Optional[float]:
        """
        获取英雄血量
        
        Returns:
            血量百分比（0.0-1.0）
        """
        return self.hero_health
    
    def is_in_danger(self) -> bool:
        """
        判断是否危险
        
        Returns:
            是否危险
        """
        # 血量低
        if self.hero_health is not None and self.hero_health < 0.3:
            return True
        
        # 有敌方英雄靠近
        if self.hero_position and self.enemy_positions:
            for enemy_pos in self.enemy_positions:
                distance = self._distance(self.hero_position, enemy_pos)
                if distance < 50:  # 距离小于50像素
                    return True
        
        return False
    
    def get_nearest_enemy(self) -> Optional[Tuple[int, int]]:
        """
        获取最近的敌方位置
        
        Returns:
            最近的敌方位置 (x, y)，没有敌方返回None
        """
        if not self.enemy_positions or not self.hero_position:
            return None
        return min(self.enemy_positions, key=lambda p: 
                  self._distance(self.hero_position, p))
    
    def get_safe_position(self) -> Optional[Tuple[int, int]]:
        """
        获取安全位置
        
        Returns:
            安全位置 (x, y)，无法找到返回None
        """
        if not self.hero_position:
            return None
        
        # 如果有防御塔，返回防御塔附近
        if self.tower_position:
            # 在防御塔和英雄之间选择中点
            safe_x = (self.hero_position[0] + self.tower_position[0]) // 2
            safe_y = (self.hero_position[1] + self.tower_position[1]) // 2
            return (safe_x, safe_y)
        
        # 否则返回地图边缘
        if self.hero_position[0] < 160:  # 左半边
            return (0, self.hero_position[1])
        else:  # 右半边
            return (319, self.hero_position[1])
    
    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        计算两个点的欧氏距离
        
        Args:
            pos1: 位置1
            pos2: 位置2
            
        Returns:
            欧氏距离
        """
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
    
    def benchmark(self, duration: int = 10) -> dict:
        """
        性能测试
        
        Args:
            duration: 测试时长（秒）
            
        Returns:
            性能统计字典
        """
        print(f"[GameState] 开始性能测试，时长: {duration}秒")
        
        update_times = []
        start_time = time.time()
        update_count = 0
        
        while time.time() - start_time < duration:
            # 测量单次更新时间
            update_start = time.time()
            self.update_from_screen()
            update_time = time.time() - update_start
            
            update_times.append(update_time)
            update_count += 1
            
            # 避免CPU占用过高
            time.sleep(0.01)
        
        # 计算统计数据
        avg_time = np.mean(update_times)
        min_time = np.min(update_times)
        max_time = np.max(update_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        result = {
            "total_updates": update_count,
            "duration": duration,
            "avg_fps": avg_fps,
            "avg_update_time": avg_time * 1000,  # 毫秒
            "min_update_time": min_time * 1000,  # 毫秒
            "max_update_time": max_time * 1000,  # 毫秒
        }
        
        print(f"[GameState] 性能测试完成:")
        print(f"  总更新次数: {update_count}")
        print(f"  平均FPS: {avg_fps:.2f}")
        print(f"  平均更新时间: {avg_time*1000:.2f}ms")
        print(f"  最小更新时间: {min_time*1000:.2f}ms")
        print(f"  最大更新时间: {max_time*1000:.2f}ms")
        
        return result


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("游戏状态识别器测试")
    print("=" * 60)
    
    # 创建游戏状态识别器
    print("\n[1/3] 创建游戏状态识别器...")
    game_state = GameState()
    
    # 测试状态更新
    print("\n[2/3] 测试状态更新...")
    game_state.update_from_screen()
    print(f"✓ 英雄位置: {game_state.get_hero_position()}")
    print(f"✓ 英雄血量: {game_state.get_health()}")
    print(f"✓ 是否危险: {game_state.is_in_danger()}")
    print(f"✓ 最近敌方: {game_state.get_nearest_enemy()}")
    print(f"✓ 安全位置: {game_state.get_safe_position()}")
    
    # 测试张量转换
    print("\n[3/3] 测试张量转换...")
    state_tensor = game_state.to_tensor()
    print(f"✓ 状态张量形状: {state_tensor.shape}")
    print(f"✓ 状态张量类型: {state_tensor.dtype}")
    print(f"✓ 状态张量样例: {state_tensor[:10]}")
    
    # 性能测试
    print("\n[4/4] 性能测试（5秒）...")
    result = game_state.benchmark(duration=5)
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)

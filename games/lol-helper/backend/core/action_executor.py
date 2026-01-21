"""
操作执行器模块
根据AI决策执行游戏操作（移动、攻击、技能等）
"""

import time
import random
from typing import Optional, Tuple
import numpy as np

from utils.input_simulator import InputSimulator
from utils.human_behavior import HumanBehaviorSimulator


class ActionExecutor:
    """
    操作执行器
    
    根据AI决策执行游戏操作，模拟人类行为
    """
    
    def __init__(self, human_behavior: Optional[HumanBehaviorSimulator] = None):
        """
        初始化操作执行器
        
        Args:
            human_behavior: 人类行为模拟器，None表示创建新的
        """
        # 输入模拟器
        self.input_simulator = InputSimulator()
        
        # 人类行为模拟器
        self.human_behavior = human_behavior or HumanBehaviorSimulator()
        
        # 游戏状态
        self.current_position = None
        self.last_action_time = 0
        self.action_count = 0
        
        print("[ActionExecutor] 操作执行器初始化成功")
    
    def move_to(self, target_pos: Tuple[int, int], 
                 speed: Optional[float] = None) -> None:
        """
        移动到目标位置（右键移动）
        
        Args:
            target_pos: 目标位置 (x, y)
            speed: 移动速度（秒），None表示自动计算
        """
        # 添加人类行为随机性
        speed = self.human_behavior.get_move_speed(speed)
        
        # 计算当前到目标的移动时间
        if self.current_position:
            distance = self._distance(self.current_position, target_pos)
            if speed is None:
                speed = distance / 500.0  # 假设500像素/秒
        
        # 模拟人类反应延迟
        self.human_behavior.add_delay()
        
        # 右键移动
        self.input_simulator.right_click(*target_pos)
        
        # 更新当前位置
        self.current_position = target_pos
        self.last_action_time = time.time()
        self.action_count += 1
        
        print(f"[ActionExecutor] 移动到: {target_pos}, 耗时: {speed:.2f}秒")
    
    def attack_target(self, target_pos: Tuple[int, int], 
                     follow: bool = True) -> None:
        """
        攻击目标（A+左键）
        
        Args:
            target_pos: 目标位置 (x, y)
            follow: 是否跟随目标
        """
        # 模拟人类反应延迟
        self.human_behavior.add_delay()
        
        # A键 + 左键
        self.input_simulator.press_key('a', 0.1)
        time.sleep(0.05)
        self.input_simulator.click(*target_pos, 'left')
        
        self.last_action_time = time.time()
        self.action_count += 1
        
        print(f"[ActionExecutor] 攻击目标: {target_pos}, 跟随: {follow}")
    
    def cast_skill(self, skill_key: str, 
                   target_pos: Optional[Tuple[int, int]] = None,
                   smart_cast: bool = True) -> None:
        """
        释放技能
        
        Args:
            skill_key: 技能键 (Q/W/E/R/D/F)
            target_pos: 目标位置，None表示朝鼠标方向
            smart_cast: 是否使用智能施法（Ctrl+技能键）
        """
        # 模拟人类反应延迟
        self.human_behavior.add_delay()
        
        # 智能施法（Ctrl+技能键）
        if smart_cast:
            self.input_simulator.press_key('ctrl_l', 0.05)
            self.input_simulator.press_key(skill_key.lower(), 0.1)
            self.input_simulator.press_key('ctrl_l', 0.05)
        else:
            # 普通施法（技能键 + 鼠标位置）
            if target_pos:
                self.input_simulator.move_mouse(*target_pos)
            self.input_simulator.press_key(skill_key.lower(), 0.1)
        
        self.last_action_time = time.time()
        self.action_count += 1
        
        print(f"[ActionExecutor] 释放技能: {skill_key}, 智能施法: {smart_cast}")
    
    def use_heal(self) -> None:
        """
        使用回血
        """
        # 模拟人类反应延迟
        self.human_behavior.add_delay()
        
        # 按下B键回城
        self.input_simulator.press_key('b', 0.2)
        
        self.last_action_time = time.time()
        self.action_count += 1
        
        print("[ActionExecutor] 使用回血")
    
    def stop(self) -> None:
        """
        停止移动（S键）
        """
        # 模拟人类反应延迟
        self.human_behavior.add_delay()
        
        # 按下S键
        self.input_simulator.press_key('s', 0.1)
        
        self.last_action_time = time.time()
        self.action_count += 1
        
        print("[ActionExecutor] 停止移动")
    
    def right_click(self, pos: Tuple[int, int]) -> None:
        """
        右键点击
        
        Args:
            pos: 点击位置 (x, y)
        """
        # 模拟人类反应延迟
        self.human_behavior.add_delay()
        
        # 右键点击
        self.input_simulator.right_click(*pos)
        
        self.last_action_time = time.time()
        self.action_count += 1
        
        print(f"[ActionExecutor] 右键点击: {pos}")
    
    def press_key(self, key: str) -> None:
        """
        按下键盘按键
        
        Args:
            key: 按键名称
        """
        # 模拟人类反应延迟
        self.human_behavior.add_delay()
        
        # 按下按键
        self.input_simulator.press_key(key, 0.15)
        
        self.last_action_time = time.time()
        self.action_count += 1
        
        print(f"[ActionExecutor] 按下按键: {key}")
    
    def execute_action_sequence(self, actions: list) -> None:
        """
        执行动作序列
        
        Args:
            actions: 动作列表
                每个动作是字典: {
                    'type': 'move'|'attack'|'skill'|'heal'|'stop',
                    'pos': (x, y),  # 可选
                    'skill': 'Q',    # 可选
                    'delay': 0.1     # 可选
                }
        """
        print(f"[ActionExecutor] 执行动作序列，共 {len(actions)} 个动作")
        
        for i, action in enumerate(actions):
            # 添加额外延迟
            if 'delay' in action:
                time.sleep(action['delay'])
            
            # 执行动作
            action_type = action['type']
            
            if action_type == 'move':
                pos = action.get('pos', self.current_position)
                if pos:
                    self.move_to(pos, action.get('speed'))
            
            elif action_type == 'attack':
                pos = action.get('pos')
                if pos:
                    self.attack_target(pos, action.get('follow', True))
            
            elif action_type == 'skill':
                skill_key = action.get('skill', 'Q')
                pos = action.get('pos')
                self.cast_skill(skill_key, pos, action.get('smart_cast', True))
            
            elif action_type == 'heal':
                self.use_heal()
            
            elif action_type == 'stop':
                self.stop()
            
            # 动作间随机延迟
            time.sleep(self.human_behavior.get_action_interval())
        
        print(f"[ActionExecutor] 动作序列执行完成，共执行 {self.action_count} 个操作")
    
    def update_position(self, pos: Tuple[int, int]) -> None:
        """
        更新当前位置（从游戏状态获取）
        
        Args:
            pos: 新位置 (x, y)
        """
        self.current_position = pos
    
    def get_action_count(self) -> int:
        """
        获取已执行的操作数量
        
        Returns:
            操作数量
        """
        return self.action_count
    
    def reset_action_count(self) -> None:
        """重置操作计数"""
        self.action_count = 0
    
    def _distance(self, pos1: Tuple[int, int], 
                  pos2: Tuple[int, int]) -> float:
        """
        计算两个点的欧氏距离
        
        Args:
            pos1: 位置1
            pos2: 位置2
            
        Returns:
            欧氏距离
        """
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
    
    def shutdown(self) -> None:
        """
        关闭操作执行器
        """
        print(f"[ActionExecutor] 关闭操作执行器，总操作数: {self.action_count}")
        self.input_simulator.stop()


# 测试代码
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    print("=" * 60)
    print("操作执行器测试")
    print("=" * 60)
    
    # 创建操作执行器
    print("\n[1/5] 创建操作执行器...")
    executor = ActionExecutor()
    print("✓ 操作执行器创建成功")
    
    # 测试移动
    print("\n[2/5] 测试移动...")
    time.sleep(1)
    executor.move_to((500, 500))
    time.sleep(1)
    executor.move_to((600, 600))
    
    # 测试攻击
    print("\n[3/5] 测试攻击...")
    time.sleep(1)
    executor.attack_target((650, 650))
    
    # 测试技能
    print("\n[4/5] 测试技能...")
    time.sleep(1)
    executor.cast_skill('q', (700, 700))
    
    # 测试动作序列
    print("\n[5/5] 测试动作序列...")
    time.sleep(1)
    actions = [
        {'type': 'move', 'pos': (550, 550)},
        {'type': 'attack', 'pos': (600, 600)},
        {'type': 'skill', 'skill': 'w', 'pos': (650, 650)},
        {'type': 'stop'}
    ]
    executor.execute_action_sequence(actions)
    
    # 关闭
    time.sleep(2)
    executor.shutdown()
    
    print("\n" + "=" * 60)
    print("✓ 所有测试完成！")
    print("=" * 60)

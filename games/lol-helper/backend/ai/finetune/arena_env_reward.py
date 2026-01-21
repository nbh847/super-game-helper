#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
复合奖励函数
用于计算英雄行动的奖励，综合考虑多个维度
"""

import numpy as np
from typing import Dict, Any
from pathlib import Path

import sys
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from utils.logger import logger


class RewardConfig:
    """奖励配置"""
    
    # 奖励权重
    REWARD_LAST_HIT = 1.2        # 补刀奖励（最高权重）
    REWARD_DAMAGE_DEALT = 0.01    # 造成伤害
    REWARD_ASSIST = 0.8           # 参与击杀
    REWARD_POSITION = 0.1          # 位置奖励（在兵线附近）
    
    # 惩罚权重
    PENALTY_DAMAGE_TAKEN = -0.015  # 承受伤害
    PENALTY_DEATH = -5.0            # 死亡惩罚（严重）
    PENALTY_IDLE = -0.01             # 空闲惩罚
    
    # 奖励上限
    MAX_REWARD_PER_FRAME = 10.0
    MAX_PENALTY_PER_FRAME = -10.0
    
    # 位置奖励配置
    MINION_LANE_DISTANCE = 300      # 兵线距离阈值
    TOWER_SAFE_DISTANCE = 800       # 防御塔安全距离


class CompositeReward:
    """
    复合奖励函数
    
    综合考虑多个维度计算奖励
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: 奖励配置（可选）
        """
        self.config = config or RewardConfig()
        
        # 统计信息
        self.rewards_history = []
        self.rewards_by_type = {
            'last_hit': [],
            'damage_dealt': [],
            'assist': [],
            'position': [],
            'damage_taken': [],
            'death': [],
            'idle': []
        }
        
        logger.info("CompositeReward初始化完成")
    
    def compute_reward(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any]) -> float:
        """
        计算奖励
        
        Args:
            state: 当前状态
                - hero_hp: 英雄血量 (0-100)
                - hero_mana: 英雄蓝量 (0-100)
                - gold: 当前金币
                - minions_killed: 补刀数
                - damage_dealt: 造成伤害累计
                - damage_taken: 承受伤害累计
                - kills: 击杀数
                - assists: 助攻数
                - deaths: 死亡数
                - hero_position: 英雄位置 (x, y)
                - minion_positions: 小兵位置列表 [(x, y), ...]
                - enemy_positions: 敌人位置列表 [(x, y), ...]
                - tower_position: 防御塔位置 (x, y)
            action: 执行的动作
            next_state: 下一状态
        
        Returns:
            reward: 奖励值
        """
        reward = 0.0
        
        # 1. 补刀奖励
        last_hit_reward = self._compute_last_hit_reward(state, next_state)
        reward += last_hit_reward
        self.rewards_by_type['last_hit'].append(last_hit_reward)
        
        # 2. 造成伤害奖励
        damage_reward = self._compute_damage_reward(state, next_state)
        reward += damage_reward
        self.rewards_by_type['damage_dealt'].append(damage_reward)
        
        # 3. 参与击杀奖励
        assist_reward = self._compute_assist_reward(state, next_state)
        reward += assist_reward
        self.rewards_by_type['assist'].append(assist_reward)
        
        # 4. 位置奖励
        position_reward = self._compute_position_reward(next_state)
        reward += position_reward
        self.rewards_by_type['position'].append(position_reward)
        
        # 5. 承受伤害惩罚
        damage_taken_penalty = self._compute_damage_taken_penalty(state, next_state)
        reward += damage_taken_penalty
        self.rewards_by_type['damage_taken'].append(damage_taken_penalty)
        
        # 6. 死亡惩罚
        death_penalty = self._compute_death_penalty(state, next_state)
        reward += death_penalty
        self.rewards_by_type['death'].append(death_penalty)
        
        # 7. 空闲惩罚
        idle_penalty = self._compute_idle_penalty(action)
        reward += idle_penalty
        self.rewards_by_type['idle'].append(idle_penalty)
        
        # 限制奖励范围
        reward = np.clip(reward, self.config.MAX_PENALTY_PER_FRAME, self.config.MAX_REWARD_PER_FRAME)
        
        # 记录历史
        self.rewards_history.append(reward)
        
        return reward
    
    def _compute_last_hit_reward(self, state: Dict[str, Any], next_state: Dict[str, Any]) -> float:
        """
        补刀奖励
        
        检测minions_killed是否增加
        """
        last_hit_reward = 0.0
        
        # 补刀数增加
        if next_state['minions_killed'] > state['minions_killed']:
            last_hit_reward = self.config.REWARD_LAST_HIT
        
        return last_hit_reward
    
    def _compute_damage_reward(self, state: Dict[str, Any], next_state: Dict[str, Any]) -> float:
        """
        造成伤害奖励
        
        检测damage_dealt是否增加
        """
        damage_dealt = next_state['damage_dealt'] - state['damage_dealt']
        
        # 避免异常值（如技能爆炸伤害）
        damage_dealt = min(damage_dealt, 1000)
        
        damage_reward = damage_dealt * self.config.REWARD_DAMAGE_DEALT
        
        return damage_reward
    
    def _compute_assist_reward(self, state: Dict[str, Any], next_state: Dict[str, Any]) -> float:
        """
        参与击杀奖励
        
        检测assists是否增加
        """
        assist_reward = 0.0
        
        # 助攻数增加
        if next_state['assists'] > state['assists']:
            assist_reward = self.config.REWARD_ASSIST
        
        return assist_reward
    
    def _compute_position_reward(self, state: Dict[str, Any]) -> float:
        """
        位置奖励
        
        英雄在兵线附近时给予奖励
        """
        hero_position = np.array(state['hero_position'])
        minion_positions = state['minion_positions']
        
        if not minion_positions:
            return 0.0
        
        # 计算到最近小兵的距离
        minion_positions = np.array(minion_positions)
        distances = np.linalg.norm(minion_positions - hero_position, axis=1)
        min_distance = np.min(distances)
        
        # 在兵线附近
        if min_distance < self.config.MINION_LANE_DISTANCE:
            position_reward = self.config.REWARD_POSITION
        else:
            position_reward = -self.config.REWARD_POSITION * 0.5
        
        return position_reward
    
    def _compute_damage_taken_penalty(self, state: Dict[str, Any], next_state: Dict[str, Any]) -> float:
        """
        承受伤害惩罚
        
        检测damage_taken是否增加
        """
        damage_taken = next_state['damage_taken'] - state['damage_taken']
        
        # 避免异常值
        damage_taken = min(damage_taken, 1000)
        
        damage_taken_penalty = damage_taken * self.config.PENALTY_DAMAGE_TAKEN
        
        return damage_taken_penalty
    
    def _compute_death_penalty(self, state: Dict[str, Any], next_state: Dict[str, Any]) -> float:
        """
        死亡惩罚
        
        检测deaths是否增加
        """
        death_penalty = 0.0
        
        # 死亡数增加
        if next_state['deaths'] > state['deaths']:
            death_penalty = self.config.PENALTY_DEATH
        
        return death_penalty
    
    def _compute_idle_penalty(self, action: int) -> float:
        """
        空闲惩罚
        
        动作为7（等待）时给予小惩罚
        """
        # 7 = 等待
        if action == 7:
            idle_penalty = self.config.PENALTY_IDLE
        else:
            idle_penalty = 0.0
        
        return idle_penalty
    
    def get_statistics(self):
        """获取奖励统计"""
        if not self.rewards_history:
            return {
                'total_frames': 0,
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0,
                'rewards_by_type': {}
            }
        
        stats = {
            'total_frames': len(self.rewards_history),
            'mean_reward': np.mean(self.rewards_history),
            'std_reward': np.std(self.rewards_history),
            'min_reward': np.min(self.rewards_history),
            'max_reward': np.max(self.rewards_history),
            'rewards_by_type': {}
        }
        
        # 统计各类型奖励
        for reward_type, values in self.rewards_by_type.items():
            if values:
                stats['rewards_by_type'][reward_type] = {
                    'mean': np.mean(values),
                    'sum': np.sum(values)
                }
            else:
                stats['rewards_by_type'][reward_type] = {
                    'mean': 0.0,
                    'sum': 0.0
                }
        
        return stats
    
    def reset(self):
        """重置统计"""
        self.rewards_history = []
        self.rewards_by_type = {
            'last_hit': [],
            'damage_dealt': [],
            'assist': [],
            'position': [],
            'damage_taken': [],
            'death': [],
            'idle': []
        }


def test_reward():
    """测试奖励函数"""
    print("=" * 50)
    print("测试复合奖励函数")
    print("=" * 50)
    
    # 创建奖励函数
    reward_fn = CompositeReward()
    
    # 模拟状态
    def create_state(frame=0):
        return {
            'hero_hp': 100 - frame * 5,  # 血量逐渐减少
            'hero_mana': 100 - frame * 3,
            'gold': 1000 + frame * 10,
            'minions_killed': 0,
            'damage_dealt': 0,
            'damage_taken': 0,
            'kills': 0,
            'assists': 0,
            'deaths': 0,
            'hero_position': [100, 200],
            'minion_positions': [[120, 200], [150, 220], [180, 200]],
            'enemy_positions': [[300, 400], [350, 450]],
            'tower_position': [0, 0]
        }
    
    # 模拟游戏过程
    print("\n模拟游戏过程...")
    for i in range(10):
        state = create_state(i)
        
        # 模拟不同动作
        if i == 2:
            # 补刀
            action = 4  # 攻击小兵
            next_state = create_state(i + 1)
            next_state['minions_killed'] = 1
            next_state['damage_dealt'] = 50
            print(f"帧 {i}: 补刀")
        elif i == 5:
            # 造成伤害
            action = 5  # 攻击英雄
            next_state = create_state(i + 1)
            next_state['damage_dealt'] = 100
            print(f"帧 {i}: 攻击英雄")
        elif i == 8:
            # 死亡
            action = 3  # 移动右
            next_state = create_state(i + 1)
            next_state['damage_taken'] = 500
            next_state['deaths'] = 1
            next_state['hero_hp'] = 0
            print(f"帧 {i}: 死亡")
        else:
            # 正常移动
            action = 2  # 移动左
            next_state = create_state(i + 1)
            print(f"帧 {i}: 移动")
        
        # 计算奖励
        reward = reward_fn.compute_reward(state, action, next_state)
        print(f"  奖励: {reward:.4f}")
    
    # 统计信息
    stats = reward_fn.get_statistics()
    print("\n奖励统计:")
    print(f"  总帧数: {stats['total_frames']}")
    print(f"  平均奖励: {stats['mean_reward']:.4f}")
    print(f"  标准差: {stats['std_reward']:.4f}")
    print(f"  最小奖励: {stats['min_reward']:.4f}")
    print(f"  最大奖励: {stats['max_reward']:.4f}")
    print(f"\n各类型奖励:")
    for reward_type, type_stats in stats['rewards_by_type'].items():
        print(f"  {reward_type}: mean={type_stats['mean']:.4f}, sum={type_stats['sum']:.4f}")
    
    print("\n✅ 所有测试通过！")


if __name__ == '__main__':
    test_reward()

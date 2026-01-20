import numpy as np
from typing import Dict, Any, Tuple
from enum import Enum


class ActionType(Enum):
    """动作类型枚举"""
    MOVE = 0          # 移动（8个方向）
    ATTACK = 8         # 普攻
    SKILL_Q = 9        # Q技能
    SKILL_W = 10       # W技能
    SKILL_E = 11       # E技能
    SKILL_R = 12       # R技能
    SUMMONER_D = 13    # D召唤师技能
    SUMMONER_F = 14    # F召唤师技能
    HEAL = 15          # 治疗
    RECALL = 16        # 回城
    STOP = 17          # 停止操作


class StateExtractor:
    """状态特征提取器
    
    从游戏帧中提取状态特征
    """
    
    def __init__(self):
        self.state_dim = 128  # 状态向量维度
    
    def extract(self, frame: Dict[str, Any]) -> np.ndarray:
        """提取状态特征
        
        Args:
            frame: 游戏帧数据
            
        Returns:
            state: 状态向量
        """
        # 提取各类特征
        hero_state = self.extract_hero_state(frame)
        enemies_state = self.extract_enemies_state(frame)
        minions_state = self.extract_minions_state(frame)
        tower_state = self.extract_tower_state(frame)
        
        # 合并所有特征
        state = np.concatenate([
            hero_state,
            enemies_state,
            minions_state,
            tower_state
        ])
        
        # 归一化
        state = self.normalize(state)
        
        return state
    
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """归一化状态向量
        
        将所有特征归一化到[0, 1]范围
        """
        # 简化版本：除以最大值
        return np.clip(state / 3000.0, 0, 1)
    
    def extract_hero_state(self, frame: Dict[str, Any]) -> np.ndarray:
        """提取英雄状态
        
        包括：位置、血量、蓝量、等级、金币、技能冷却等
        """
        # 位置 (x, y) - 归一化到[0, 1]
        position = frame.get('hero_position', (0, 0))
        normalized_pos = [position[0] / 1920, position[1] / 1080]
        
        # 血量 - 归一化到[0, 1]
        health = frame.get('health', 0) / 5000.0
        health = np.clip(health, 0, 1)
        
        # 蓝量 - 归一化到[0, 1]
        mana = frame.get('mana', 0) / 3000.0
        mana = np.clip(mana, 0, 1)
        
        # 等级 - 归一化到[0, 1]
        level = frame.get('level', 1) / 18.0
        level = np.clip(level, 0, 1)
        
        # 金币 - 归一化到[0, 1]
        gold = frame.get('gold', 0) / 20000.0
        gold = np.clip(gold, 0, 1)
        
        # 技能冷却 (Q, W, E, R) - 0=可用, 1=冷却中
        skills = [
            frame.get('skill_q_cd', 0),
            frame.get('skill_w_cd', 0),
            frame.get('skill_e_cd', 0),
            frame.get('skill_r_cd', 0)
        ]
        skills = np.array(skills, dtype=np.float32)
        
        # KDA - 归一化
        kills = frame.get('kills', 0) / 20.0
        deaths = frame.get('deaths', 0) / 10.0
        assists = frame.get('assists', 0) / 20.0
        
        kda = np.clip([kills, deaths, assists], 0, 1)
        
        # 合并所有特征
        hero_features = np.concatenate([
            normalized_pos,  # 2
            [health, mana, level, gold],  # 4
            skills,  # 4
            kda  # 3
        ])
        
        return hero_features  # 总共13个特征
    
    def extract_enemies_state(self, frame: Dict[str, Any]) -> np.ndarray:
        """提取敌方英雄状态
        
        包括最多5个敌方英雄的位置、血量等信息
        """
        enemies = frame.get('enemies', [])
        
        # 最多考虑5个敌人
        max_enemies = 5
        enemy_features = []
        
        for i in range(max_enemies):
            if i < len(enemies):
                enemy = enemies[i]
                # 位置 (x, y)
                pos = enemy.get('position', (0, 0))
                normalized_pos = [pos[0] / 1920, pos[1] / 1080]
                
                # 血量
                health = enemy.get('health', 0) / 5000.0
                health = np.clip(health, 0, 1)
                
                # 距离（相对于己方英雄）
                my_pos = frame.get('hero_position', (0, 0))
                distance = self._calculate_distance(my_pos, pos) / 2000.0
                distance = np.clip(distance, 0, 1)
                
                enemy_features.extend(normalized_pos)
                enemy_features.extend([health, distance])
            else:
                # 用0填充不存在的敌人
                enemy_features.extend([0, 0, 0, 0])
        
        return np.array(enemy_features, dtype=np.float32)  # 5 * 4 = 20个特征
    
    def extract_minions_state(self, frame: Dict[str, Any]) -> np.ndarray:
        """提取小兵状态
        
        包括最近的小兵位置和数量
        """
        minions = frame.get('minions', [])
        
        # 最多考虑10个小兵
        max_minions = 10
        minion_features = []
        
        # 计算己方小兵和敌方小兵
        friendly_minions = [m for m in minions if not m.get('is_enemy', True)]
        enemy_minions = [m for m in minions if m.get('is_enemy', True)]
        
        my_pos = frame.get('hero_position', (0, 0))
        
        # 按距离排序
        friendly_minions.sort(key=lambda m: self._calculate_distance(my_pos, m.get('position', (0, 0))))
        enemy_minions.sort(key=lambda m: self._calculate_distance(my_pos, m.get('position', (0, 0))))
        
        # 取最近的小兵
        for i in range(max_minions // 2):
            # 己方小兵
            if i < len(friendly_minions):
                pos = friendly_minions[i].get('position', (0, 0))
                normalized_pos = [pos[0] / 1920, pos[1] / 1080]
                distance = self._calculate_distance(my_pos, pos) / 1000.0
                distance = np.clip(distance, 0, 1)
                minion_features.extend(normalized_pos)
                minion_features.append(distance)
            else:
                minion_features.extend([0, 0, 0])
            
            # 敌方小兵
            if i < len(enemy_minions):
                pos = enemy_minions[i].get('position', (0, 0))
                normalized_pos = [pos[0] / 1920, pos[1] / 1080]
                distance = self._calculate_distance(my_pos, pos) / 1000.0
                distance = np.clip(distance, 0, 1)
                minion_features.extend(normalized_pos)
                minion_features.append(distance)
            else:
                minion_features.extend([0, 0, 0])
        
        return np.array(minion_features, dtype=np.float32)  # 10 * 3 = 30个特征
    
    def extract_tower_state(self, frame: Dict[str, Any]) -> np.ndarray:
        """提取防御塔状态
        
        包括最近的防御塔位置和血量
        """
        tower = frame.get('tower', {})
        
        if not tower:
            return np.zeros(5, dtype=np.float32)
        
        # 位置
        pos = tower.get('position', (0, 0))
        normalized_pos = [pos[0] / 1920, pos[1] / 1080]
        
        # 血量
        health = tower.get('health', 0) / 5000.0
        health = np.clip(health, 0, 1)
        
        # 距离（相对于己方英雄）
        my_pos = frame.get('hero_position', (0, 0))
        distance = self._calculate_distance(my_pos, pos) / 2000.0
        distance = np.clip(distance, 0, 1)
        
        # 是否是敌方防御塔
        is_enemy = 1.0 if tower.get('is_enemy', False) else 0.0
        
        tower_features = np.array(
            normalized_pos + [health, distance, is_enemy],
            dtype=np.float32
        )
        
        return tower_features  # 5个特征
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """计算两点之间的欧氏距离"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

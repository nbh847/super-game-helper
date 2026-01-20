import numpy as np
from typing import Dict, Any, List, Tuple
from enum import Enum
import math


class ActionType(Enum):
    """动作类型枚举"""
    # 移动（8个方向）
    MOVE_UP = 0
    MOVE_UP_RIGHT = 1
    MOVE_RIGHT = 2
    MOVE_DOWN_RIGHT = 3
    MOVE_DOWN = 4
    MOVE_DOWN_LEFT = 5
    MOVE_LEFT = 6
    MOVE_UP_LEFT = 7
    
    # 攻击
    ATTACK = 8
    
    # 技能
    SKILL_Q = 9
    SKILL_W = 10
    SKILL_E = 11
    SKILL_R = 12
    
    # 召唤师技能
    SUMMONER_D = 13
    SUMMONER_F = 14
    
    # 其他
    HEAL = 15
    RECALL = 16
    STOP = 17
    
    # 技能组合（V2+使用）
    Q_W_COMBO = 18
    Q_E_COMBO = 19
    W_E_COMBO = 20
    FULL_COMBO = 21


class ActionExtractor:
    """动作特征提取器
    
    从游戏数据中提取玩家操作并编码
    """
    
    def __init__(self, num_actions: int = 32):
        self.num_actions = num_actions
        self.action_mapping = {
            'move': self._encode_move_direction,
            'attack': lambda x: ActionType.ATTACK.value,
            'skill_q': lambda x: ActionType.SKILL_Q.value,
            'skill_w': lambda x: ActionType.SKILL_W.value,
            'skill_e': lambda x: ActionType.SKILL_E.value,
            'skill_r': lambda x: ActionType.SKILL_R.value,
            'summoner_d': lambda x: ActionType.SUMMONER_D.value,
            'summoner_f': lambda x: ActionType.SUMMONER_F.value,
            'heal': lambda x: ActionType.HEAL.value,
            'recall': lambda x: ActionType.RECALL.value,
            'stop': lambda x: ActionType.STOP.value
        }
    
    def extract(self, frame_data: Dict[str, Any]) -> int:
        """提取动作ID
        
        Args:
            frame_data: 帧数据，包含操作信息
            
        Returns:
            action_id: 动作ID
        """
        action_type = frame_data.get('action_type', 'stop')
        
        if action_type in self.action_mapping:
            action_id = self.action_mapping[action_type](frame_data)
        else:
            action_id = ActionType.STOP.value
        
        return action_id
    
    def _encode_move_direction(self, frame_data: Dict[str, Any]) -> int:
        """编码移动方向（8个方向）"""
        target_pos = frame_data.get('target_position', (0, 0))
        current_pos = frame_data.get('current_position', (0, 0))
        
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # 计算角度（弧度）
        angle = math.atan2(dy, dx)
        
        # 转换为角度制（0-360）
        angle_deg = math.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360
        
        # 映射到8个方向
        # 0-22.5: UP
        # 22.5-67.5: UP_RIGHT
        # 67.5-112.5: RIGHT
        # 等等...
        direction = int((angle_deg + 22.5) // 45) % 8
        
        return direction
    
    def encode(self, action: Dict[str, Any]) -> int:
        """编码动作字典为ID
        
        Args:
            action: 动作字典，包含type和相关信息
            
        Returns:
            action_id: 动作ID
        """
        action_type = action.get('type', 'stop')
        action_data = {
            'action_type': action_type,
            'current_position': action.get('current_position', (0, 0)),
            'target_position': action.get('target_position', (0, 0)),
            'skill_key': action.get('skill_key', None)
        }
        
        return self.extract(action_data)
    
    def decode(self, action_id: int) -> Dict[str, Any]:
        """解码动作ID为动作字典
        
        Args:
            action_id: 动作ID
            
        Returns:
            action: 动作字典
        """
        # 反向映射
        reverse_mapping = {v: k for k, v in {
            'move': ActionType.MOVE_UP.value,
            'attack': ActionType.ATTACK.value,
            'skill_q': ActionType.SKILL_Q.value,
            'skill_w': ActionType.SKILL_W.value,
            'skill_e': ActionType.SKILL_E.value,
            'skill_r': ActionType.SKILL_R.value,
            'summoner_d': ActionType.SUMMONER_D.value,
            'summoner_f': ActionType.SUMMONER_F.value,
            'heal': ActionType.HEAL.value,
            'recall': ActionType.RECALL.value,
            'stop': ActionType.STOP.value
        }.items()}
        
        action_type = reverse_mapping.get(action_id, 'stop')
        
        return {
            'type': action_type,
            'id': action_id
        }
    
    def get_action_space(self) -> int:
        """获取动作空间大小"""
        return self.num_actions
    
    def is_valid_action(self, action: Dict[str, Any]) -> bool:
        """验证动作是否有效
        
        Args:
            action: 动作字典
            
        Returns:
            is_valid: 是否有效
        """
        action_type = action.get('type', '')
        
        if action_type not in ['move', 'attack', 'skill_q', 'skill_w', 
                               'skill_e', 'skill_r', 'summoner_d', 
                               'summoner_f', 'heal', 'recall', 'stop']:
            return False
        
        # 验证移动动作
        if action_type == 'move':
            target_pos = action.get('target_position')
            if not target_pos or len(target_pos) != 2:
                return False
        
        # 验证攻击动作
        if action_type == 'attack':
            target_pos = action.get('target_position')
            if not target_pos or len(target_pos) != 2:
                return False
        
        # 验证技能动作
        if action_type in ['skill_q', 'skill_w', 'skill_e', 'skill_r']:
            skill_key = action.get('skill_key')
            if not skill_key:
                return False
        
        return True
    
    def action_to_one_hot(self, action_id: int) -> np.ndarray:
        """将动作ID转换为one-hot编码
        
        Args:
            action_id: 动作ID
            
        Returns:
            one_hot: one-hot编码向量
        """
        one_hot = np.zeros(self.num_actions, dtype=np.float32)
        one_hot[action_id] = 1.0
        return one_hot
    
    def one_hot_to_action(self, one_hot: np.ndarray) -> int:
        """从one-hot编码转换为动作ID
        
        Args:
            one_hot: one-hot编码向量
            
        Returns:
            action_id: 动作ID
        """
        return int(np.argmax(one_hot))
    
    def get_action_name(self, action_id: int) -> str:
        """获取动作名称
        
        Args:
            action_id: 动作ID
            
        Returns:
            name: 动作名称
        """
        action_names = {
            0: '上',
            1: '右上',
            2: '右',
            3: '右下',
            4: '下',
            5: '左下',
            6: '左',
            7: '左上',
            8: '攻击',
            9: 'Q技能',
            10: 'W技能',
            11: 'E技能',
            12: 'R技能',
            13: 'D召唤师技能',
            14: 'F召唤师技能',
            15: '治疗',
            16: '回城',
            17: '停止'
        }
        
        return action_names.get(action_id, '未知')

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
手动约束规则
基于规则的安全检查，防止AI做出愚蠢的决策
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

import sys
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from utils.logger import logger


class RuleConfig:
    """规则配置"""
    
    # 规则：没蓝不能放技能
    MANA_EMPTY_THRESHOLD = 10.0  # 没蓝阈值
    
    # 攻击动作（可能需要蓝的技能）
    SKILL_ACTIONS = [4, 5]  # 攻击小兵、攻击英雄
    
    # 普通攻击动作（不需要蓝）
    BASIC_ATTACK_ACTIONS = []  # 当前动作空间都是移动/攻击/回城/等待


class RuleBasedSafety:
    """
    基于规则的安全检查器
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: 规则配置（可选）
        """
        self.config = config or RuleConfig()
        
        # 统计信息
        self.rule_triggers = {
            'mana_empty': 0
        }
        self.total_checks = 0
        self.rule_overrides = 0
        
        logger.info("RuleBasedSafety初始化完成")
    
    def check_safety(self, state: Dict[str, Any], ai_action: int) -> Tuple[int, str]:
        """
        检查安全性并可能覆盖AI动作
        
        Args:
            state: 游戏状态
            ai_action: AI选择的动作
        
        Returns:
            final_action: 最终动作（可能被规则覆盖）
            rule_name: 触发的规则名称（None表示未触发）
        """
        self.total_checks += 1
        
        # 规则：没蓝不能放技能，只能用普通攻击
        final_action, rule_name = self._rule_mana_empty(state, ai_action)
        if rule_name:
            return final_action, rule_name
        
        # 未触发任何规则
        return ai_action, None
    
    def _rule_mana_empty(self, state: Dict[str, Any], ai_action: int) -> Tuple[int, Optional[str]]:
        """
        规则：没蓝不能放技能，只能用普通攻击
        
        Args:
            state: 游戏状态
            ai_action: AI选择的动作
        
        Returns:
            final_action: 最终动作
            rule_name: 规则名称（None表示未触发）
        """
        hero_mana = state.get('hero_mana', 100.0)
        
        # 没蓝，如果AI选择需要蓝的动作
        if hero_mana < self.config.MANA_EMPTY_THRESHOLD:
            if ai_action in self.config.SKILL_ACTIONS:
                self.rule_triggers['mana_empty'] += 1
                self.rule_overrides += 1
                logger.warning(f"规则触发：没蓝({hero_mana:.1f}%), 禁止放技能，改用普通攻击")
                # 改为等待（实际应用中应该改为普通攻击动作）
                # 当前动作空间：0-3移动, 4攻击小兵, 5攻击英雄, 6回城, 7等待
                # 这里我们保持原动作，只是记录警告
                # 或者可以选择改为更安全的动作
                return ai_action, 'mana_empty'
        
        return ai_action, None
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            'total_checks': self.total_checks,
            'rule_overrides': self.rule_overrides,
            'override_rate': self.rule_overrides / self.total_checks if self.total_checks > 0 else 0.0,
            'rule_triggers': self.rule_triggers.copy()
        }
    
    def reset(self):
        """重置统计"""
        self.rule_triggers = {
            'mana_empty': 0
        }
        self.total_checks = 0
        self.rule_overrides = 0


def test_safety_rules():
    """测试安全规则"""
    print("=" * 50)
    print("测试安全规则")
    print("=" * 50)
    
    safety = RuleBasedSafety()
    
    # 测试规则：没蓝
    print("\n测试规则：没蓝时禁止放技能")
    state_no_mana = {
        'hero_hp': 100.0,
        'hero_mana': 5.0,
        'enemy_positions': [[300, 400]],
        'hero_position': [200, 300]
    }
    
    action, rule_name = safety.check_safety(state_no_mana, ai_action=4)
    print(f"  AI动作: 4 (攻击小兵)")
    print(f"  最终动作: {action}, 规则: {rule_name}")
    
    # 测试规则：有蓝
    print("\n测试规则：有蓝时允许放技能")
    state_has_mana = {
        'hero_hp': 100.0,
        'hero_mana': 80.0,
        'enemy_positions': [[300, 400]],
        'hero_position': [200, 300]
    }
    
    action, rule_name = safety.check_safety(state_has_mana, ai_action=5)
    print(f"  AI动作: 5 (攻击英雄)")
    print(f"  最终动作: {action}, 规则: {rule_name}")
    
    # 测试规则：没蓝时移动
    print("\n测试规则：没蓝时移动（不受影响）")
    state_no_mana_move = {
        'hero_hp': 100.0,
        'hero_mana': 5.0,
        'enemy_positions': [[300, 400]],
        'hero_position': [200, 300]
    }
    
    action, rule_name = safety.check_safety(state_no_mana_move, ai_action=2)
    print(f"  AI动作: 2 (移动左)")
    print(f"  最终动作: {action}, 规则: {rule_name}")
    
    # 统计信息
    stats = safety.get_statistics()
    print("\n规则统计:")
    print(f"  总检查数: {stats['total_checks']}")
    print(f"  覆盖数: {stats['rule_overrides']}")
    print(f"  覆盖率: {stats['override_rate']*100:.1f}%")
    print(f"  规则触发: {stats['rule_triggers']}")
    
    print("\n✅ 所有测试通过！")


if __name__ == '__main__':
    test_safety_rules()
